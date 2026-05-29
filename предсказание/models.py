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
import pandas as pd

# Импорт всех архитектур
from architecture.model_1 import SpectralGAN
from architecture.model_2 import SpectralUNet
from architecture.model_3 import SpectralTransformer
from architecture.model_4 import LightSpectralMamba
from architecture.model_5_cascade import SpectralResNet

# 1. load data
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
    X = np.array(X_list)
    y = np.array(y_list)
    wavelengths = np.array(wl_list[0])
    print(f"Загружено образцов: {X.shape[0]}")
    print(f"Длина спектра: {y.shape[1]} точек")
    print(f"Диапазон длин волн: {wavelengths[0]:.1f} – {wavelengths[-1]:.1f} нм")
    return X, y, wavelengths

# 2 Предобработка
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
    else:  # raw
        X_train_proc = X_train.copy()
        X_val_proc = X_val.copy() if X_val is not None else None

    scaler_y = None
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
    else:  # raw
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
# mb nado chtoto eche dobavit
# ladno i tak poidet

def ultra_low_brightness_loss(pred, target, sam_weight=0.3, eps=1e-8):
    if pred.dim() == 3 and pred.size(1) == 1:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
    offset = 2.0
    log_pred = torch.log(torch.abs(pred) + offset)
    log_target = torch.log(torch.abs(target) + offset)
    msle = F.mse_loss(log_pred, log_target)
    cos_sim = F.cosine_similarity(pred, target, dim=-1)
    sam_loss = torch.acos(torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)).mean()
    return (1 - sam_weight) * msle + sam_weight * sam_loss

def weighted_l1_sam_loss(pred, target, alpha=0.5):
    if pred.dim() == 3 and pred.size(1) == 1:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
    num_points = pred.size(-1)
    weights = torch.ones(num_points, device=pred.device)
    start_tail_idx = 1523
    if num_points > start_tail_idx:
        tail_len = num_points - start_tail_idx
        weights[start_tail_idx:] = torch.linspace(1.0, 5.0, steps=tail_len, device=pred.device)
    l1_loss = (torch.abs(pred - target) * weights).mean()
    cos_sim = F.cosine_similarity(pred, target, dim=-1).mean()
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
    sam_loss = torch.acos(cos_sim)
    return (1 - alpha) * l1_loss + alpha * (sam_loss / torch.pi)

def spectral_mamba_loss(pred, target, sam_weight=0.5, eps=1e-8):
    if pred.dim() == 3 and pred.size(1) == 1:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
    rel_error = ((pred - target) ** 2) / (target ** 2 + eps)
    rel_mse = rel_error.mean()
    cos_sim = F.cosine_similarity(pred, target, dim=-1)
    sam_loss = torch.acos(torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)).mean()
    return (1 - sam_weight) * rel_mse + sam_weight * sam_loss

logcosh = lambda pred, target: torch.log(torch.cosh(pred - target)).mean()
mse_l1 = lambda pred, target: 0.5 * nn.MSELoss()(pred, target) + 0.5 * nn.L1Loss()(pred, target)

loss_dict = {
    'Huber_delta1.0': nn.HuberLoss(delta=1.0),
    'Huber_delta0.5': nn.HuberLoss(delta=0.5),
    'Huber_delta5.0': nn.HuberLoss(delta=5.0),
    'MSE': nn.MSELoss(),
    'L1': nn.L1Loss(),
    'SmoothL1': nn.SmoothL1Loss(),
    'LogCosh': logcosh,
    'MSE+L1': mse_l1,
    'UltraLowBrightness': ultra_low_brightness_loss,
    'WeightedL1_SAM': weighted_l1_sam_loss,
    'SpectralMambaLoss': spectral_mamba_loss,
}
# dobavit Huber_delta WeightedL1_SAM i dlya mambi SpectralMambaLoss loss

# 4. визуал и метрики
def evaluate_and_plot(model, loader, y_inverse_func, y_raw_test,
                      method_name, wavelengths, save_dir, device):
    model.eval()
    all_preds_norm = []
    all_true_norm = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred_norm = model(xb)                 # [B,1,1904]
            all_preds_norm.append(pred_norm.cpu().numpy())
            all_true_norm.append(yb.cpu().numpy())

    preds_norm = np.vstack(all_preds_norm)[:, 0, :]   # (n, 1904)
    true_norm = np.vstack(all_true_norm)[:, 0, :]

    preds_phys = y_inverse_func(preds_norm)
    preds_phys = np.maximum(preds_phys, 0)
    trues_phys = y_raw_test

    # ----- Метрики -----
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

    # ----- Дополнительные величины для графиков -----
    total_irr = np.sum(trues_phys, axis=1)                     # суммарная яркость
    sample_rmse = np.sqrt(np.mean((preds_phys - trues_phys)**2, axis=1))
    mean_sample = np.mean(trues_phys, axis=1)
    sample_rel_rmse = sample_rmse / (mean_sample + 1e-8)

    # ----- Сохранение графиков -----
    os.makedirs(save_dir, exist_ok=True)

    # 1. Средний спектр
    plt.figure(figsize=(12, 5))
    plt.plot(wavelengths, trues_phys.mean(0), 'b-', label='True')
    plt.plot(wavelengths, preds_phys.mean(0), 'r--', label='Predicted')
    plt.title(f"Средний спектр – {method_name}")
    plt.xlabel("Длина волны (нм)")
    plt.ylabel("Интенсивность")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.savefig(os.path.join(save_dir, "mean_spectrum.png"), dpi=200)
    plt.close()

    # 2. Отношение предсказание / истина
    plt.figure(figsize=(12, 5))
    ratio = preds_phys.mean(0) / (trues_phys.mean(0) + 1e-8)
    plt.plot(wavelengths, ratio, 'r')
    plt.axhline(1.0, color='blue', linestyle='--')
    plt.ylim(0.85, 1.15)
    plt.title(f"Отношение предсказание / истина – {method_name}")
    plt.xlabel("Длина волны (нм)")
    plt.grid(alpha=0.25)
    plt.savefig(os.path.join(save_dir, "ratio.png"), dpi=200)
    plt.close()

    # 3. Относительная RMSE по длинам волн
    rmse_per_wvl = np.sqrt(np.mean((preds_phys - trues_phys)**2, axis=0))
    rel_rmse_curve = rmse_per_wvl / (np.mean(trues_phys, axis=0) + 1e-8)
    plt.figure(figsize=(12, 5))
    plt.plot(wavelengths, rel_rmse_curve, 'r')
    plt.title(f"Относительная RMSE по длинам волн – {method_name}")
    plt.xlabel("Длина волны (нм)")
    plt.grid(alpha=0.25)
    plt.savefig(os.path.join(save_dir, "error_per_wvl.png"), dpi=200)
    plt.close()

    # 4. Относительная RMSE образца vs суммарная яркость
    plt.figure(figsize=(8, 6))
    plt.scatter(total_irr, sample_rel_rmse, alpha=0.5, c='steelblue', edgecolors='k', linewidth=0.3)
    plt.xlabel("Суммарная яркость истинного спектра")
    plt.ylabel("Относительная RMSE образца (RMSE / средняя яркость)")
    plt.title(f"Относительная RMSE vs Total Irradiance – {method_name}")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "rel_rmse_vs_irr.png"), dpi=200)
    plt.close()

    # 5. Абсолютная RMSE образца vs суммарная яркость (тот график, который вы просили)
    plt.figure(figsize=(8, 6))
    plt.scatter(total_irr, sample_rmse, alpha=0.5, c='steelblue', edgecolors='k', linewidth=0.3)
    plt.xlabel("Суммарная яркость истинного спектра")
    plt.ylabel("RMSE образца")
    plt.title(f"RMSE vs Total Irradiance – {method_name}")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "rmse_vs_irr.png"), dpi=200)
    plt.close()

    # 6. Пример одного спектра
    idx = np.random.randint(0, len(trues_phys))
    plt.figure(figsize=(12, 5))
    plt.plot(wavelengths, trues_phys[idx], 'b-', label='True')
    plt.plot(wavelengths, preds_phys[idx], 'r--', label='Predicted')
    plt.title(f"Пример спектра (idx={idx}, RMSE={sample_rmse[idx]:.4f}, RelRMSE={sample_rel_rmse[idx]:.4f}) – {method_name}")
    plt.xlabel("Длина волны (нм)")
    plt.ylabel("Интенсивность")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.savefig(os.path.join(save_dir, "sample_example.png"), dpi=200)
    plt.close()

    print("=" * 70)
    print(f"РЕЗУЛЬТАТЫ НА ТЕСТЕ – {method_name}")
    print("-" * 70)
    print(f"RMSE      = {rmse:.4f}")
    print(f"MAE       = {mae:.4f}")
    print(f"Rel RMSE  = {rel_rmse:.4f} ({rel_rmse*100:.2f}%)")
    print(f"SAM       = {sam:.3f}°")
    print(f"R²        = {r2:.5f}")
    print("=" * 70)

    metrics = {
        'RMSE': rmse,
        'SAM': sam,
        'R2': r2,
        'RelRMSE': rel_rmse,
        'MAE': mae
    }
    return metrics


# 5. обучение комбинации (модель + loss + предобработка)
def train_one_config(model_class, train_loader, val_loader, test_loader,
                     y_inverse_func, y_raw_test, method_name,
                     wavelengths, save_dir, criterion, device,
                     epochs=150, lr=1e-3, print_every=30):
    model = model_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        optimizer, mode='min', factor=0.5, patience=10
    )
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_loss = float('inf')
    best_state = None
    train_losses, val_losses = [], []

    print("="*70)
    print(f"Обучение: {method_name} | Устройство: {device} | Эпох: {epochs}")
    header = f"{'Epoch':>5s}  {'TrainLoss':>10s}  {'ValLoss':>10s}  {'lr':>10s}  {'ValRMSE':>10s}  {'Time(s)':>8s}  {'Tot(min)':>8s}"
    print("-"*len(header))
    print(header)
    print("-"*len(header))
    total_start = time.time()

    for epoch in range(1, epochs+1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)           # [B,1,1904]
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        # валидационный RMSE для вывода
        val_rmse = float('nan')
        if epoch % print_every == 0 or epoch == 1:
            with torch.no_grad():
                xb, yb = next(iter(val_loader))
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                pred_np = y_inverse_func(pred.cpu().numpy().reshape(pred.size(0), -1))
                true_np = y_inverse_func(yb.cpu().numpy().reshape(yb.size(0), -1))
                if np.any(np.isnan(pred_np)) or np.any(np.isinf(pred_np)):
                    print(f"WARNING: NaN/Inf ВСЕ ПЛОХЛ В  {epoch}")
                    val_rmse = np.nan
                else:
                    val_rmse = np.sqrt(mean_squared_error(true_np, pred_np))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        epoch_time = time.time() - epoch_start
        total_elapsed = (time.time() - total_start) / 60.0
        current_lr = scheduler.get_last_lr()[0]
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % print_every == 0 or epoch == 1:
            print(f"{epoch:5d}  {train_loss:10.6f}  {val_loss:10.6f}  {current_lr:10.2e}  {val_rmse:10.4f}  {epoch_time:8.2f}  {total_elapsed:8.1f}")

    total_time = time.time() - total_start
    print("-"*len(header))
    print(f"Обучение завершено за {total_time:.1f} сек ({total_time/60:.1f} мин)")
    print(f"Лучшая val_loss: {best_val_loss:.6f}\n")

    model.load_state_dict(best_state)
    # Кривая обучения
    plt.figure(figsize=(10,6))
    plt.plot(range(1, epochs+1), train_losses, 'b-', label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, 'r-', label='Val Loss')
    plt.xlabel('Эпоха'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f"Кривые обучения – {method_name}")
    plt.savefig(os.path.join(save_dir, "training_curve.png"), dpi=200)
    plt.close()

    metrics = evaluate_and_plot(model, test_loader, y_inverse_func, y_raw_test,
                                method_name, wavelengths, save_dir, device)
    return metrics

# 6 Перебор, лосс, предобработка
if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("PyTorch version:", torch.__version__)
    print(f"Используется устройство: {DEVICE}")

    DATA_PATH = r"D:\scrap-heap\Reconstruction_of_the_solar_spectrum\data_set\paired_data"
    BASE_OUT_DIR = "models_grid_search"
    os.makedirs(BASE_OUT_DIR, exist_ok=True)

    # Загрузка и сплит
    X, y, wl = load_spectral_data(DATA_PATH)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)

    model_classes = [
        #LightSpectralMamba,
        #SpectralResNet
        # SpectralRefinerNetV2,
         SpectralTransformer
    ]

    x_methods = ['standart', 'raw']
    # робаст робаст а не робуст хотя норм
    y_methods = ['robust','standart']
    selected_losses = ['Huber_delta0.5', 'MSE+L1']

    # CSV-файл для сводки

    csv_path = os.path.join(BASE_OUT_DIR, "summary.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("model,x_method,y_method,loss,RMSE,SAM_deg,R2,RelRMSE,MAE\n")

    total_runs = len(model_classes) * len(x_methods) * len(y_methods) * len(selected_losses)
    run_idx = 0

    for model_cls in model_classes:
        for xm in x_methods:
            for ym in y_methods:
                for loss_name in selected_losses:
                    run_idx += 1
                    print(f"\n{'#'*80}")
                    print(f"Запуск {run_idx}/{total_runs}: модель={model_cls.__name__}, x={xm}, y={ym}, loss={loss_name}")
                    print(f"{'#'*80}")

                    # Предобработка
                    (X_tr_p, y_tr_p, X_val_p, y_val_p,
                     _, inv_y, _) = preprocess_data(
                        X_tr, y_tr, X_val, y_val,
                        x_method=xm, y_method=ym,
                        to_tensor=True, device=DEVICE
                    )
                    X_te_p, y_te_p, _, _, _, _, _ = preprocess_data(
                        X_te, y_te,
                        x_method=xm, y_method=ym,
                        to_tensor=True, device=DEVICE
                    )

                    #починиьт тензоры
                    train_ds = TensorDataset(X_tr_p.unsqueeze(1), y_tr_p.unsqueeze(1))
                    val_ds   = TensorDataset(X_val_p.unsqueeze(1), y_val_p.unsqueeze(1))
                    test_ds  = TensorDataset(X_te_p.unsqueeze(1), y_te_p.unsqueeze(1))
                    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
                    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
                    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

                    # выбор loss
                    criterion = loss_dict[loss_name]

                    # Имя для папки
                    safe_name = f"{model_cls.__name__}_x{xm}_y{ym}_{loss_name}".replace('.', '_')
                    combo_dir = os.path.join(BASE_OUT_DIR, safe_name)
                    os.makedirs(combo_dir, exist_ok=True)

                    # Обучение
                    metrics = train_one_config(
                        model_cls, train_loader, val_loader, test_loader,
                        inv_y, y_te,
                        method_name=safe_name,
                        wavelengths=wl,
                        save_dir=combo_dir,
                        criterion=criterion,
                        device=DEVICE,
                        epochs=180,
                        lr=1e-3,
                        print_every=5
                    )

                    # запись в CSV
                    with open(csv_path, 'a', encoding='utf-8') as f:
                        f.write(f"{model_cls.__name__},{xm},{ym},{loss_name},"
                                f"{metrics['RMSE']:.6f},{metrics['SAM']:.4f},{metrics['R2']:.6f},"
                                f"{metrics['RelRMSE']:.6f},{metrics['MAE']:.6f}\n")

    # Итоговая таблица
    print("\n" + "="*80)
    print("ПОЛНАЯ СВОДКА РЕЗУЛЬТАТОВ (отсортировано по RMSE)")
    print("="*80)
    df = pd.read_csv(csv_path)
    print(df.sort_values('RMSE').to_string(index=False))
    print(f"\nРезультаты сохранены в: {csv_path}")

# sdelati normalnii vivod a to niche ne ponatno
# починить gan
# забить на gan и затестить unet resnet и трансформер
# починить unet resnet, а трансформер идет гулять оч долго
"""
Напомнить себе никогда больше не браться за обратные задачи и задачи с физикой и гиперкамерой
где много всего а все равно ничего не понятно. уволиться и пойти делать туркменам лабы 
и курсовые за бабки. 
"""
# dopilit graviki
# сделать перебор предобработак
# попридумывать всякие еще доп loss
# vinesti v novii proekt se
# zakinuti v ii dlya udaleni moix commentov i dlya normalnogo naiminga
"""
допилить неймниг и сделать побольше понятных выводов. Сделать всю оптимизацию по коду.
"""
# сделать отчет и reamde и на гит