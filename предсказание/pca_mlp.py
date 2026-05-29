import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd


# 1 Загрузка данных
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
        if len(multi_vals) != 9 or len(spec_vals) == 0:
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
    else:
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

# 3 Модель MLP
class SimpleMLP(nn.Module):
    #simpledimpal papl papl scvit
    def __init__(self, input_dim=9, output_dim=64, hidden=[512,512,512], dropout=0.1):
        super().__init__()
        layers = []
        dim = input_dim
        for h in hidden:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            dim = h
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# 4 Оценка и графики
def evaluate_and_plot(model, loader,
                      pca_mean, pca_comp, y_inverse_func,
                      y_raw_test, method_name,
                      wavelengths, save_dir, device):
    model.eval()
    all_preds_proc = []
    all_true_proc = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            coeffs = model(xb)
            pred_prep = coeffs @ pca_comp.to(device) + pca_mean.to(device)
            all_preds_proc.append(pred_prep.cpu().numpy())
            all_true_proc.append(yb.cpu().numpy())

    preds_proc = np.vstack(all_preds_proc)
    trues_proc = np.vstack(all_true_proc)
    preds_phys = y_inverse_func(preds_proc)
    preds_phys = np.maximum(preds_phys, 0)
    trues_phys = y_raw_test

    # Метрики
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

    # Величины для графиков
    total_irr = np.sum(trues_phys, axis=1)
    sample_rmse = np.sqrt(np.mean((preds_phys - trues_phys)**2, axis=1))
    mean_sample = np.mean(trues_phys, axis=1)
    sample_rel_rmse = sample_rmse / (mean_sample + 1e-8)

    os.makedirs(save_dir, exist_ok=True)

    # 1. Средний спектр
    plt.figure(figsize=(12, 5))
    plt.plot(wavelengths, trues_phys.mean(0), 'b-', label='True')
    plt.plot(wavelengths, preds_phys.mean(0), 'r--', label='Predicted')
    plt.title(f"Средний спектр – {method_name}")
    plt.xlabel("Длина волны (нм)"); plt.ylabel("Интенсивность")
    plt.legend(); plt.grid(alpha=0.25)
    plt.savefig(os.path.join(save_dir, "mean_spectrum.png"), dpi=200)
    plt.close()

    # 2. Отношение предсказание / истина
    plt.figure(figsize=(12, 5))
    ratio = preds_phys.mean(0) / (trues_phys.mean(0) + 1e-8)
    plt.plot(wavelengths, ratio, 'r')
    plt.axhline(1.0, color='blue', linestyle='--')
    plt.ylim(0.85, 1.15)
    plt.title(f"Отношение предсказание / истина – {method_name}")
    plt.xlabel("Длина волны (нм)"); plt.grid(alpha=0.25)
    plt.savefig(os.path.join(save_dir, "ratio.png"), dpi=200)
    plt.close()

    # 3. Относительная ошибка по длинам волн
    rmse_per_wvl = np.sqrt(np.mean((preds_phys - trues_phys)**2, axis=0))
    rel_rmse_curve = rmse_per_wvl / (np.mean(trues_phys, axis=0) + 1e-8)
    plt.figure(figsize=(12, 5))
    plt.plot(wavelengths, rel_rmse_curve, 'r')
    plt.title(f"Относительная RMSE по длинам волн – {method_name}")
    plt.xlabel("Длина волны (нм)"); plt.grid(alpha=0.25)
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

    # 5. Абсолютная RMSE образца vs суммарная яркость
    plt.figure(figsize=(8, 6))
    plt.scatter(total_irr, sample_rmse, alpha=0.5, c='steelblue', edgecolors='k', linewidth=0.3)
    plt.xlabel("Суммарная яркость истинного спектра")
    plt.ylabel("RMSE образца")
    plt.title(f"RMSE vs Total Irradiance – {method_name}")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "rmse_vs_irr.png"), dpi=200)
    plt.close()

    # 6. Пример одного тестового спектра
    idx = np.random.randint(0, len(trues_phys))
    plt.figure(figsize=(12, 5))
    plt.plot(wavelengths, trues_phys[idx], 'b-', label='True')
    plt.plot(wavelengths, preds_phys[idx], 'r--', label='Predicted')
    plt.title(f"Пример спектра (idx={idx}, RMSE={sample_rmse[idx]:.4f}, RelRMSE={sample_rel_rmse[idx]:.4f}) – {method_name}")
    plt.xlabel("Длина волны (нм)"); plt.ylabel("Интенсивность")
    plt.legend(); plt.grid(alpha=0.25)
    plt.savefig(os.path.join(save_dir, "sample_example.png"), dpi=200)
    plt.close()

    # Вывод метрик
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

# 5 Обучение
def train_mlp(model, train_loader, val_loader, test_loader,
              pca_mean, pca_comp, y_inverse_func,
              y_raw_test, method_name, wavelengths,
              save_dir, criterion, epochs=150, lr=1e-3, device='cpu', print_every=30):
    model = model.to(device)
    pca_mean = pca_mean.to(device)
    pca_comp = pca_comp.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    best_state = None

    # История потерь
    train_losses = []
    val_losses = []

    print("=" * 70)
    print(f"Обучение: {method_name} | Устройство: {device} | Эпох: {epochs}")
    header = f"{'Epoch':>5s}  {'TrainLoss':>10s}  {'ValLoss':>10s}  {'lr':>10s}  {'ValRMSE':>10s}  {'Time(s)':>8s}  {'Tot(min)':>8s}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    total_start = time.time()

    for epoch in range(1, epochs+1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            coeffs = model(xb)
            pred_y = coeffs @ pca_comp + pca_mean
            loss = criterion(pred_y, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                coeffs = model(xb)
                pred_y = coeffs @ pca_comp + pca_mean
                val_loss += criterion(pred_y, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        # Быстрый RMSE на одном батче валидации
        val_rmse = float('nan')
        if epoch % print_every == 0 or epoch == 1:
            with torch.no_grad():
                xb, yb = next(iter(val_loader))
                xb, yb = xb.to(device), yb.to(device)
                coeffs = model(xb)
                pred_prep = coeffs @ pca_comp + pca_mean
                pred_np = y_inverse_func(pred_prep.cpu().numpy())
                true_np = y_inverse_func(yb.cpu().numpy())
                val_rmse = np.sqrt(mean_squared_error(true_np, pred_np))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        epoch_time = time.time() - epoch_start
        total_elapsed = (time.time() - total_start) / 60.0
        current_lr = scheduler.get_last_lr()[0]

        # Сохраняем историю потерь
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % print_every == 0 or epoch == 1:
            print(f"{epoch:5d}  {train_loss:10.6f}  {val_loss:10.6f}  {current_lr:10.2e}  {val_rmse:10.4f}  {epoch_time:8.2f}  {total_elapsed:8.1f}")

    total_time = time.time() - total_start
    print("-" * len(header))
    print(f"Обучение завершено за {total_time:.1f} сек ({total_time/60:.1f} мин)")
    print(f"Лучшая val_loss: {best_val_loss:.6f}\n")

    model.load_state_dict(best_state)

    # График процесса обучения
    epochs_range = np.arange(1, epochs+1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs_range, val_losses, 'r-', label='Val Loss')
    plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f"Кривые обучения – {method_name}")
    plt.xlabel('Эпоха')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curve.png"), dpi=200)
    plt.close()

    # Финальная оценка с графиками
    metrics = evaluate_and_plot(
        model, test_loader,
        pca_mean, pca_comp, y_inverse_func,
        y_raw_test, method_name,
        wavelengths, save_dir, device
    )
    return metrics

# 6 перебор преобработак
if __name__ == "__main__":
    # Методы предобработки
    x_methods = ['standart','raw']
    y_methods = ['standard','robust']

    logcosh = lambda pred, target: torch.log(torch.cosh(pred - target)).mean()
    mse_l1 = lambda pred, target: 0.5 * nn.MSELoss()(pred, target) + 0.5 * nn.L1Loss()(pred, target)

    loss_dict = {
        'Huber(delta=1.0)': nn.HuberLoss(delta=1.0),
        'Huber(delta=0.5)': nn.HuberLoss(delta=0.5),
        'L1': nn.L1Loss(),
        'SmoothL1': nn.SmoothL1Loss(),
        'MSE+L1': mse_l1
    }

    # Параметры обучения и модели
    N_COMP = 5               # число PCA компонент
    HIDDEN = [1024, 1024, 1024]   # архитектура MLP
    EPOCHS = 150
    LR = 1e-3
    PRINT_EVERY = 10           # каждые N эпох выводить подробную информацию

    # Пути
    DATA_PATH = (r"D:\scrap-heap\Reconstruction_of_the_solar_spectrum\data_set\paired_data")
    BASE_OUT_DIR = "mlp_grid_search"


    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("PyTorch version:", torch.__version__)
    print(f"Используется устройство: {DEVICE}")

    os.makedirs(BASE_OUT_DIR, exist_ok=True)
    X, y, wl = load_spectral_data(DATA_PATH)

    # Разделение данных
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)

    # Файл для сводки
    csv_path = os.path.join(BASE_OUT_DIR, "summary.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("x_method,y_method,loss,RMSE,SAM_deg,R2,RelRMSE,MAE\n")

    # Тройной цикл: x-метод, y-метод, loss
    total_runs = len(x_methods) * len(y_methods) * len(loss_dict)
    run = 0
    for xm in x_methods:
        for ym in y_methods:
            for loss_name, criterion in loss_dict.items():
                run += 1
                print(f"\n{'#'*80}")
                print(f"Запуск {run}/{total_runs}: x={xm}, y={ym}, loss={loss_name}")
                print(f"{'#'*80}")

                # Имя комбинации для папки
                safe_name = loss_name.replace('=', '_').replace('(', '').replace(')', '').replace(' ', '_').replace('.', '_')
                combo_name = f"x{xm}_y{ym}_{safe_name}"
                combo_dir = os.path.join(BASE_OUT_DIR, combo_name)
                os.makedirs(combo_dir, exist_ok=True)

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

                # PCA на обучающих y
                pca = PCA(n_components=N_COMP, random_state=42)
                y_tr_np = y_tr_p.cpu().numpy()
                pca.fit(y_tr_np)
                explained = pca.explained_variance_ratio_.sum()
                print(f"PCA: {N_COMP} компонент объясняют {explained:.4f}")
                pca_mean = torch.tensor(pca.mean_, dtype=torch.float32, device=DEVICE)
                pca_comp = torch.tensor(pca.components_, dtype=torch.float32, device=DEVICE)

                # DataLoaders
                train_loader = DataLoader(TensorDataset(X_tr_p, y_tr_p), batch_size=64, shuffle=True)
                val_loader = DataLoader(TensorDataset(X_val_p, y_val_p), batch_size=64, shuffle=False)
                test_loader = DataLoader(TensorDataset(X_te_p, y_te_p), batch_size=64, shuffle=False)

                # Модель
                model = SimpleMLP(input_dim=9, output_dim=N_COMP, hidden=HIDDEN).to(DEVICE)

                # Обучение с заданной loss-функцией
                metrics = train_mlp(
                    model, train_loader, val_loader, test_loader,
                    pca_mean, pca_comp, inv_y,
                    y_te,                     # оригинальные y теста
                    f"{combo_name}",          # method_name для графиков
                    wl,
                    combo_dir,                # сюда сохранятся все 6+1 графиков
                    criterion,
                    epochs=EPOCHS,
                    lr=LR,
                    device=DEVICE,
                    print_every=PRINT_EVERY
                )

                # Запись в CSV
                with open(csv_path, 'a', encoding='utf-8') as f:
                    f.write(f"{xm},{ym},{loss_name},{metrics['RMSE']:.6f},{metrics['SAM']:.4f},"
                            f"{metrics['R2']:.6f},"
                            f"{metrics['RelRMSE']:.6f},{metrics['MAE']:.6f}\n")

    # Итоговая таблица
    print("\n" + "="*80)
    print("ПОЛНАЯ СВОДКА РЕЗУЛЬТАТОВ")
    print("="*80)
    df = pd.read_csv(csv_path)
    print(df.sort_values('RMSE').to_string(index=False))
    print(f"\nРезультаты сохранены в: {csv_path}")
"""
ПОЛНАЯ СВОДКА РЕЗУЛЬТАТОВ
================================================================================
x_method y_method             loss      RMSE  SAM_deg       R2  MAPE_percent  RelRMSE       MAE
standard standard Huber(delta=1.0) 34.996386   0.9735 0.996827  8737772.5129 0.048828 18.254247
standard standard Huber(delta=5.0) 37.519731   0.9973 0.996353  8722504.2146 0.052349 19.341696
standard standard              MSE 37.949482   1.0003 0.996269  8895711.7746 0.052948 19.366505
standard standard          LogCosh 38.267538   1.0056 0.996206  8631955.5106 0.053392 19.110020
standard standard         SmoothL1 40.279104   1.0252 0.995797  8893479.7903 0.056199 19.686569
standard standard           MSE+L1 45.349402   1.0603 0.994672  8458723.5658 0.063273 20.002099
standard standard Huber(delta=0.5) 46.292737   1.0612 0.994448  8666897.6022 0.064589 19.855439
standard standard               L1 56.156641   1.1493 0.991831  8308843.5256 0.078351 21.098787
"""