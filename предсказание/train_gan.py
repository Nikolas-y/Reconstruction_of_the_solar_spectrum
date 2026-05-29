import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Определения генератора и дискриминатора (если их нет в model_1.py)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.act(self.linear1(x))
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.linear2(x)
        return residual + self.dropout(x)

class SpectralGenerator(nn.Module):
    def __init__(self, input_len=9, output_len=1904, hidden_dim=512, n_blocks=4, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_len, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_len)
        )
        self.shortcut = nn.Linear(input_len, output_len)

    def forward(self, x):
        # x: [B, 1, 9] -> [B, 9]
        x_flat = x.view(x.size(0), -1)
        base = self.shortcut(x_flat)
        feat = self.input_layer(x_flat)
        for blk in self.res_blocks:
            feat = blk(feat)
        detail = self.output_layer(feat)
        return (base + detail).unsqueeze(1)

class SpectralDiscriminator(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=64, n_layers=3):
        super().__init__()
        layers = []
        in_ch = input_channels
        out_ch = hidden_channels
        for i in range(n_layers):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)
        layers.append(nn.Conv1d(in_ch, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class SpectralGAN(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.gen = generator
        self.disc = discriminator

    def forward(self, x):
        return self.gen(x)

# Вспомогательные функции потерь для GAN
# надо заботать GAN
def gan_loss_discriminator(real_pred, fake_pred):
    """Hinge loss для дискриминатора (стабилен)"""
    real_loss = torch.mean(F.relu(1.0 - real_pred))
    fake_loss = torch.mean(F.relu(1.0 + fake_pred))
    return real_loss + fake_loss

def gan_loss_generator(fake_pred):
    """Потеря генератора: -mean(fake_pred) для Hinge loss"""
    return -torch.mean(fake_pred)

# Загрузка данных как в models.py
def load_spectral_data(data_path):
    X_list, y_list, wl_list = [], [], []
    for fname in os.listdir(data_path):
        if not (fname.startswith('paired_') and fname.endswith('.txt')):
            continue
        with open(os.path.join(data_path, fname), 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if len(lines) < 4:
            continue
        multi_vals = list(map(float, lines[1].strip().split()))
        spec_wl    = list(map(float, lines[2].strip().split()))
        spec_vals  = list(map(float, lines[3].strip().split()))
        if len(multi_vals) != 9 or len(spec_vals) != 1904:
            continue
        X_list.append(multi_vals)
        y_list.append(spec_vals)
        wl_list.append(spec_wl)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    wavelengths = np.array(wl_list[0])
    print(f"Загружено образцов: {X.shape[0]}")
    print(f"Длина спектра: {y.shape[1]} точек")
    print(f"Диапазон длин волн: {wavelengths[0]:.1f} – {wavelengths[-1]:.1f} нм")
    return X, y, wavelengths

def preprocess_data(X_train, y_train, X_val=None, y_val=None,
                    x_method='raw', y_method='raw',
                    to_tensor=False, device='cpu'):
    scaler_x = None
    if x_method == 'standard':
        scaler_x = StandardScaler()
        X_train_proc = scaler_x.fit_transform(X_train)
        X_val_proc = scaler_x.transform(X_val) if X_val is not None else None
    elif x_method == 'robust':
        scaler_x = RobustScaler()
        X_train_proc = scaler_x.fit_transform(X_train)
        X_val_proc = scaler_x.transform(X_val) if X_val is not None else None
    elif x_method == 'log':
        scaler_x = 'log'
        X_train_proc = np.log1p(np.maximum(X_train, 0))
        X_val_proc = np.log1p(np.maximum(X_val, 0)) if X_val is not None else None
    else:
        X_train_proc = X_train.copy()
        X_val_proc = X_val.copy() if X_val is not None else None

    if y_method == 'log':
        y_train_proc = np.log1p(np.maximum(y_train, 0))
        y_val_proc = np.log1p(np.maximum(y_val, 0)) if y_val is not None else None
        y_inverse = lambda x: np.expm1(x)
    elif y_method == 'standard':
        scaler_y = StandardScaler()
        y_train_proc = scaler_y.fit_transform(y_train)
        y_val_proc = scaler_y.transform(y_val) if y_val is not None else None
        y_inverse = lambda x: scaler_y.inverse_transform(x)
    elif y_method == 'robust':
        scaler_y = RobustScaler()
        y_train_proc = scaler_y.fit_transform(y_train)
        y_val_proc = scaler_y.transform(y_val) if y_val is not None else None
        y_inverse = lambda x: scaler_y.inverse_transform(x)
    else:
        y_train_proc = y_train.copy()
        y_val_proc = y_val.copy() if y_val is not None else None
        y_inverse = lambda x: x

    if to_tensor:
        X_train_proc = torch.tensor(X_train_proc, dtype=torch.float32, device=device)
        y_train_proc = torch.tensor(y_train_proc, dtype=torch.float32, device=device)
        if X_val_proc is not None:
            X_val_proc = torch.tensor(X_val_proc, dtype=torch.float32, device=device)
            y_val_proc = torch.tensor(y_val_proc, dtype=torch.float32, device=device)

    return (X_train_proc, y_train_proc,
            X_val_proc, y_val_proc,
            scaler_x, y_inverse, scaler_y)

# Функция оценки и графиков models.py точно такая же
def evaluate_and_plot(model, loader, y_inverse_func, y_raw_test,
                      method_name, wavelengths, save_dir, device):
    model.eval()
    all_preds_norm = []
    all_true_norm = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred_norm = model(xb)
            all_preds_norm.append(pred_norm.cpu().numpy())
            all_true_norm.append(yb.cpu().numpy())
    preds_norm = np.vstack(all_preds_norm)[:, 0, :]
    true_norm = np.vstack(all_true_norm)[:, 0, :]
    preds_phys = y_inverse_func(preds_norm)
    preds_phys = np.maximum(preds_phys, 0)
    trues_phys = y_raw_test

    mse = mean_squared_error(trues_phys, preds_phys)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(trues_phys - preds_phys))
    r2 = r2_score(trues_phys.flatten(), preds_phys.flatten())

    norm_p = np.linalg.norm(preds_phys, axis=1, keepdims=True)
    norm_t = np.linalg.norm(trues_phys, axis=1, keepdims=True)
    dot = np.sum(preds_phys * trues_phys, axis=1, keepdims=True)
    cos = dot / (norm_p * norm_t + 1e-8)
    sam = np.mean(np.arccos(np.clip(cos, -1, 1))) * (180 / np.pi)
    rel_rmse = rmse / (np.mean(trues_phys) + 1e-8)

    total_irr = np.sum(trues_phys, axis=1)
    sample_rmse = np.sqrt(np.mean((preds_phys - trues_phys)**2, axis=1))
    mean_sample = np.mean(trues_phys, axis=1)
    sample_rel_rmse = sample_rmse / (mean_sample + 1e-8)

    os.makedirs(save_dir, exist_ok=True)

    # Средний спектр
    plt.figure(figsize=(12,5))
    plt.plot(wavelengths, trues_phys.mean(0), 'b-', label='True')
    plt.plot(wavelengths, preds_phys.mean(0), 'r--', label='Pred')
    plt.title(f"Средний спектр – {method_name}")
    plt.xlabel("Длина волны (нм)")
    plt.ylabel("Интенсивность")
    plt.legend(); plt.grid(alpha=0.25)
    plt.savefig(os.path.join(save_dir, "mean_spectrum.png"), dpi=200)
    plt.close()

    # Ratio
    plt.figure(figsize=(12,5))
    ratio = preds_phys.mean(0) / (trues_phys.mean(0) + 1e-8)
    plt.plot(wavelengths, ratio, 'r')
    plt.axhline(1.0, color='blue', linestyle='--')
    plt.ylim(0.85,1.15)
    plt.title(f"Ratio – {method_name}")
    plt.grid(alpha=0.25)
    plt.savefig(os.path.join(save_dir, "ratio.png"), dpi=200)
    plt.close()

    # Относительная RMSE по длинам волн
    rmse_wvl = np.sqrt(np.mean((preds_phys - trues_phys)**2, axis=0))
    rel_rmse_wvl = rmse_wvl / (np.mean(trues_phys, axis=0) + 1e-8)
    plt.figure(figsize=(12,5))
    plt.plot(wavelengths, rel_rmse_wvl, 'r')
    plt.title(f"Относительная RMSE по λ – {method_name}")
    plt.xlabel("Длина волны (нм)")
    plt.grid(alpha=0.25)
    plt.savefig(os.path.join(save_dir, "error_per_wvl.png"), dpi=200)
    plt.close()

    # Относительная ошибка vs яркость
    plt.figure(figsize=(8,6))
    plt.scatter(total_irr, sample_rel_rmse, alpha=0.5, c='steelblue', edgecolors='k')
    plt.xlabel("Суммарная яркость")
    plt.ylabel("Относительная RMSE образца")
    plt.title(f"RelRMSE vs яркость – {method_name}")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "rel_rmse_vs_irr.png"), dpi=200)
    plt.close()

    # Абсолютная RMSE vs яркость
    plt.figure(figsize=(8,6))
    plt.scatter(total_irr, sample_rmse, alpha=0.5, c='steelblue', edgecolors='k')
    plt.xlabel("Суммарная яркость")
    plt.ylabel("RMSE образца")
    plt.title(f"RMSE vs Total Irradiance – {method_name}")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "rmse_vs_irr.png"), dpi=200)
    plt.close()

    # Пример спектра
    idx = np.random.randint(0, len(trues_phys))
    plt.figure(figsize=(12,5))
    plt.plot(wavelengths, trues_phys[idx], 'b-', label='True')
    plt.plot(wavelengths, preds_phys[idx], 'r--', label='Pred')
    plt.title(f"Пример (idx={idx}, RMSE={sample_rmse[idx]:.4f})")
    plt.legend(); plt.grid(alpha=0.25)
    plt.savefig(os.path.join(save_dir, "sample_example.png"), dpi=200)
    plt.close()

    print("="*70)
    print(f"РЕЗУЛЬТАТЫ НА ТЕСТЕ – {method_name}")
    print("-"*70)
    print(f"RMSE      = {rmse:.4f}")
    print(f"MAE       = {mae:.4f}")
    print(f"Rel RMSE  = {rel_rmse:.4f} ({rel_rmse*100:.2f}%)")
    print(f"SAM       = {sam:.3f}°")
    print(f"R²        = {r2:.5f}")
    print("="*70)

    return {'RMSE': rmse, 'SAM': sam, 'R2': r2, 'RelRMSE': rel_rmse, 'MAE': mae}


# Функция обучения GAN
def train_gan(generator, discriminator, train_loader, val_loader, test_loader,
              y_inverse_func, y_raw_test, method_name, wavelengths, save_dir, device,
              epochs=150, lr_gen=1e-4, lr_disc=1e-4, lambda_l1=100, n_critic=1, print_every=10):
    gen = generator.to(device)
    disc = discriminator.to(device)

    opt_g = optim.AdamW(gen.parameters(), lr=lr_gen, betas=(0.5, 0.9), weight_decay=1e-4)
    opt_d = optim.AdamW(disc.parameters(), lr=lr_disc, betas=(0.5, 0.9), weight_decay=1e-4)
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs, eta_min=1e-6)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    best_state = None

    print("="*70)
    print(f"Обучение GAN: {method_name} | Устройство: {device} | Эпох: {epochs}")
    header = f"{'Epoch':>5s}  {'G_Loss':>10s}  {'D_Loss':>10s}  {'ValRMSE':>10s}  {'Time(s)':>8s}"
    print("-"*len(header))
    print(header)
    print("-"*len(header))
    total_start = time.time()

    for epoch in range(1, epochs+1):
        epoch_start = time.time()
        gen.train()
        disc.train()

        d_loss_total = 0.0
        g_loss_total = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # Обучение дискриминатора
            for _ in range(n_critic):
                opt_d.zero_grad()
                with torch.no_grad():
                    fake = gen(xb)
                real_pred = disc(yb)
                fake_pred = disc(fake)
                d_loss = gan_loss_discriminator(real_pred, fake_pred)
                d_loss.backward()
                opt_d.step()
                d_loss_total += d_loss.item() * xb.size(0)

            # Обучение генератора
            opt_g.zero_grad()
            fake = gen(xb)
            fake_pred = disc(fake)
            adv_loss = gan_loss_generator(fake_pred)
            l1_loss = F.l1_loss(fake, yb)
            g_loss = adv_loss + lambda_l1 * l1_loss
            g_loss.backward()
            opt_g.step()
            g_loss_total += g_loss.item() * xb.size(0)

        d_loss_total /= len(train_loader.dataset)
        g_loss_total /= len(train_loader.dataset)
        scheduler_g.step()
        scheduler_d.step()

        # Оценка RMSE на валидации
        gen.eval()
        val_rmse = float('nan')
        with torch.no_grad():
            xb, yb = next(iter(val_loader))
            xb, yb = xb.to(device), yb.to(device)
            pred = gen(xb)
            pred_np = y_inverse_func(pred.cpu().numpy().reshape(pred.size(0), -1))
            true_np = y_inverse_func(yb.cpu().numpy().reshape(yb.size(0), -1))
            if not (np.any(np.isnan(pred_np)) or np.any(np.isinf(pred_np))):
                val_rmse = np.sqrt(mean_squared_error(true_np, pred_np))

        # Сохранение лучшего генератора по d_loss или val_rmse
        if d_loss_total < best_val_loss:
            best_val_loss = d_loss_total
            best_state = copy.deepcopy(gen.state_dict())

        epoch_time = time.time() - epoch_start
        if epoch % print_every == 0 or epoch == 1:
            print(f"{epoch:5d}  {g_loss_total:10.4f}  {d_loss_total:10.4f}  {val_rmse:10.4f}  {epoch_time:8.2f}")

    total_time = time.time() - total_start
    print("-"*len(header))
    print(f"Обучение завершено за {total_time:.1f} сек ({total_time/60:.1f} мин)")

    gen.load_state_dict(best_state)
    # Сохраняем тренировочные кривые

    metrics = evaluate_and_plot(gen, test_loader, y_inverse_func, y_raw_test,
                                method_name, wavelengths, save_dir, device)
    return metrics

# Главный блок
if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("PyTorch version:", torch.__version__)
    print(f"Используется устройство: {DEVICE}")

    DATA_PATH = r"D:\scrap-heap\Reconstruction_of_the_solar_spectrum\data_set\paired_data"
    BASE_OUT_DIR = "gan_experiments"
    os.makedirs(BASE_OUT_DIR, exist_ok=True)

    # Загрузка и сплит
    X, y, wl = load_spectral_data(DATA_PATH)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)

    # Предобработка (x=standard, y=standard) – можно оставить как в PCA+MLP
    (X_tr_p, y_tr_p, X_val_p, y_val_p,
     _, inv_y, _) = preprocess_data(
        X_tr, y_tr, X_val, y_val,
        x_method='standard', y_method='standard',
        to_tensor=True, device=DEVICE
    )
    X_te_p, y_te_p, _, _, _, _, _ = preprocess_data(
        X_te, y_te,
        x_method='standard', y_method='standard',
        to_tensor=True, device=DEVICE
    )

    # DataLoaders
    train_ds = TensorDataset(X_tr_p.unsqueeze(1), y_tr_p.unsqueeze(1))
    val_ds   = TensorDataset(X_val_p.unsqueeze(1), y_val_p.unsqueeze(1))
    test_ds  = TensorDataset(X_te_p.unsqueeze(1), y_te_p.unsqueeze(1))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Создание генератора, дискриминатора и GAN
    generator = SpectralGenerator(input_len=9, output_len=1904, hidden_dim=512, n_blocks=4)
    discriminator = SpectralDiscriminator(input_channels=1, hidden_channels=64, n_layers=3)
    gan_model = SpectralGAN(generator, discriminator)  # не используется напрямую, но можно сохранить

    # Обучение GAN
    method_name = "SpectralGAN_standard_standard"
    save_dir = os.path.join(BASE_OUT_DIR, method_name)
    os.makedirs(save_dir, exist_ok=True)

    metrics = train_gan(
        generator, discriminator,
        train_loader, val_loader, test_loader,
        inv_y, y_te,
        method_name=method_name,
        wavelengths=wl,
        save_dir=save_dir,
        device=DEVICE,
        epochs=150,
        lr_gen=1e-4,
        lr_disc=1e-4,
        lambda_l1=50,
        n_critic=1,
        print_every=5
    )

    # Сохранение метрик в CSV
    csv_path = os.path.join(BASE_OUT_DIR, "summary.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("model,x_method,y_method,RMSE,SAM_deg,R2,RelRMSE,MAE\n")
    with open(csv_path, 'a', encoding='utf-8') as f:
        f.write(f"SpectralGAN,standard,standard,{metrics['RMSE']:.6f},{metrics['SAM']:.4f},"
                f"{metrics['R2']:.6f},{metrics['RelRMSE']:.6f},{metrics['MAE']:.6f}\n")

    print(f"\nРезультаты сохранены в {csv_path}")
    print("Обучение GAN завершено!")