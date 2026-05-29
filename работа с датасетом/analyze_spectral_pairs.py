import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import seaborn as sns

def load_spectral_data(data_path):
    X_list, y_list, wavelengths_list = [], [], []
    files = [f for f in os.listdir(data_path) if f.startswith('paired_') and f.endswith('.txt')]

    for filename in files:
        file_path = os.path.join(data_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if len(lines) < 4:
            print(f"Пропуск {filename}: недостаточно строк")
            continue

        # Парсим строки
        multi_wl = list(map(float, lines[0].strip().split()))   # можно не использовать, т.к. фиксированы
        multi_vals = list(map(float, lines[1].strip().split()))
        spec_wl = list(map(float, lines[2].strip().split()))
        spec_vals = list(map(float, lines[3].strip().split()))

        if len(multi_vals) != 9:
            print(f"Пропуск {filename}: ожидалось 9 мультиспектральных значений, получено {len(multi_vals)}")
            continue

        if len(spec_vals) == 0:
            print(f"Пропуск {filename}: пустой спектр")
            continue

        X_list.append(multi_vals)
        y_list.append(spec_vals)
        wavelengths_list.append(spec_wl)

    if not wavelengths_list:
        raise ValueError("Не найдено подходящих paired_*.txt файлов")

    # проверка одинаковая длина волн
    ref_wl = wavelengths_list[0]
    for wl in wavelengths_list[1:]:
        if len(wl) != len(ref_wl) or not np.allclose(wl, ref_wl, rtol=1e-5, atol=1e-8):
            print("Внимание: длины волн спектрометра различаются между файлами.")

    X = np.array(X_list)
    y = np.array(y_list)
    wavelengths = np.array(ref_wl)

    print(f"Загружено: {X.shape[0]} образцов, {X.shape[1]} признаков, {y.shape[1]} точек спектра.")
    return X, y, wavelengths


# 2. Вспомогательная функция
def create_non_overlapping_bounds(wavelengths, n_channels=9):
    w_min, w_max = wavelengths.min(), wavelengths.max()
    bounds = []
    step = (w_max - w_min) / n_channels
    for i in range(n_channels):
        low = w_min + i * step
        high = w_min + (i + 1) * step
        bounds.append((low, high))
    return bounds


# 3 Отдельные функции для каждого графика / блока анализа
def plot_total_irradiance_distribution(y, wavelengths, save_dir):
    """График 1: распределение суммарной яркости спектров"""
    total_irr = np.sum(y, axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(total_irr, bins=80, color='skyblue', edgecolor='black', alpha=0.85)
    plt.title("Распределение суммарной яркости спектров")
    plt.xlabel("Суммарная яркость")
    plt.ylabel("Количество спектров")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "01_total_irradiance_distribution.png"), dpi=220, bbox_inches='tight')
    plt.show()
    return total_irr


def plot_mean_spectrum_with_std(y, wavelengths, save_dir):
    """График 2: средний спектр ± стандартное отклонение"""
    mean_spec = y.mean(axis=0)
    std_spec = y.std(axis=0)

    plt.figure(figsize=(12, 7))
    plt.plot(wavelengths, mean_spec, 'b', lw=2.2, label='Средний спектр')
    plt.fill_between(wavelengths, mean_spec - std_spec, mean_spec + std_spec,
                     color='blue', alpha=0.25, label='±1 std')
    plt.title("Средний спектр ± стандартное отклонение")
    plt.xlabel("Длина волны (нм)")
    plt.ylabel("Интенсивность")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "02_mean_spectrum_with_std.png"), dpi=220, bbox_inches='tight')
    plt.show()
    return mean_spec, std_spec


def print_pca_by_ranges(y, wavelengths):
    """Текстовый анализ: PCA по нескольким спектральным диапазонам"""
    ranges = [(400, 450), (450, 500), (500, 600), (600, 700), (700, 800), (800, 900)]
    print("\n=== PCA АНАЛИЗ — ПЕРВЫЕ 5 КОМПОНЕНТ ПО ДИАПАЗОНАМ ===\n")
    for start, end in ranges:
        mask = (wavelengths >= start) & (wavelengths <= end)
        if np.sum(mask) < 2:
            print(f"Диапазон {start}-{end} нм: слишком мало точек, пропускаем")
            continue
        y_range = y[:, mask]
        pca = PCA(n_components=5)
        pca.fit(y_range)
        print(f"Диапазон {start:3d}-{end:3d} нм  ({np.sum(mask)} точек):")
        for i in range(5):
            explained = pca.explained_variance_ratio_[i] * 100
            print(f"   PC{i+1:2d}: {explained:6.2f}%")
        print("-" * 50)


def plot_top10_bright_dark_with_mean(y, wavelengths, save_dir):
    """График 4: топ-10 самых ярких и тусклых + средний спектр"""
    total_irr = np.trapezoid(y, x=wavelengths, axis=1)
    brightest_idx = np.argsort(total_irr)[-10:][::-1]
    darkest_idx = np.argsort(total_irr)[:10]

    plt.figure(figsize=(13, 7))
    plt.plot(wavelengths, y.mean(axis=0), 'black', lw=2.8, label='Средний спектр', alpha=0.95)
    for idx in brightest_idx:
        plt.plot(wavelengths, y[idx], color='orange', lw=0.85, alpha=0.65)
    for idx in darkest_idx:
        plt.plot(wavelengths, y[idx], color='blue', lw=0.85, alpha=0.65)

    plt.title("Топ-10 самых ярких (оранжевые) + топ-10 самых тусклых (синие) + средний спектр (чёрный)")
    plt.xlabel("Длина волны (нм)")
    plt.ylabel("Интенсивность")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "04_top10_bright_dark_plus_mean.png"), dpi=220, bbox_inches='tight')
    plt.show()


def plot_coefficient_of_variation(y, wavelengths, save_dir):
    """График 5: коэффициент вариации (CV) по длинам волн"""
    mean_spec = y.mean(axis=0)
    std_spec = y.std(axis=0)
    cv = std_spec / (mean_spec + 1e-8)

    plt.figure(figsize=(12, 6))
    plt.plot(wavelengths, cv, 'darkred', lw=2)
    plt.title("Коэффициент вариации (std / mean) по длинам волн")
    plt.xlabel("Длина волны (нм)")
    plt.ylabel("CV = std / mean")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "05_coefficient_of_variation.png"), dpi=220, bbox_inches='tight')
    plt.show()


def plot_pca_full(y, wavelengths, save_dir):
    """График 6: PCA всего спектра (кумулятивная дисперсия и нагрузки)"""
    N = y.shape[0]
    pca_full = PCA(n_components = 10)
    pca_full.fit(y)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)

    # Вывод текста
    n_show = min(10, pca_full.n_components_)
    print("\n=== ОБЩИЙ PCA ПО ВСЕМУ СПЕКТРУ ===\n")
    print("Объяснённая дисперсия по компонентам:")
    for i in range(n_show):
        print(f"   PC{i+1:2d}: {pca_full.explained_variance_ratio_[i]*100:6.2f}%  "
              f"(кумулятивно {cumsum[i]*100:.2f}%)")

    # График кумулятивной дисперсии
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumsum)+1), cumsum, 'bo-', markersize=4)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% explained')
    plt.axhline(y=0.99, color='g', linestyle='--', label='99% explained')
    plt.title("Кумулятивная объяснённая дисперсия — полный спектр")
    plt.xlabel("Число главных компонент")
    plt.ylabel("Доля объяснённой дисперсии")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "06_pca_full_cumulative_variance.png"), dpi=220, bbox_inches='tight')
    plt.show()

    # Нагрузки первых трёх компонент
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for i in range(3):
        axes[i].plot(wavelengths, pca_full.components_[i], color=f'C{i}')
        axes[i].set_ylabel(f'PC{i+1} нагрузка')
        axes[i].grid(alpha=0.3)
    axes[-1].set_xlabel('Длина волны (нм)')
    fig.suptitle('Первые три главные компоненты (нагрузки) полного спектра', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "06_pca_full_loadings.png"), dpi=220, bbox_inches='tight')
    plt.show()


def plot_pca_X(X, save_dir):
    """График 7: PCA для входных признаков (мультиспектральных значений)"""
    if X is None:
        print("X не передан, PCA для входных признаков пропущен.")
        return
    X = np.asarray(X, dtype=np.float32)
    n_features = X.shape[1]
    pca_X = PCA(n_components=n_features)
    pca_X.fit(X)
    cumsum_X = np.cumsum(pca_X.explained_variance_ratio_)

    print("\n=== PCA ДЛЯ ВХОДНЫХ ПРИЗНАКОВ X ===\n")
    print("Объяснённая дисперсия по компонентам (входные признаки):")
    for i in range(n_features):
        print(f"   PC{i+1:2d}: {pca_X.explained_variance_ratio_[i]*100:6.2f}%  "
              f"(кумулятивно {cumsum_X[i]*100:.2f}%)")

    # График кумулятивной дисперсии
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_features+1), cumsum_X, 'mo-', markersize=6)
    plt.title("Кумулятивная объяснённая дисперсия — входные признаки")
    plt.xlabel("Число главных компонент")
    plt.ylabel("Доля объяснённой дисперсии")
    plt.grid(alpha=0.3)
    plt.xticks(range(1, n_features+1))
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(save_dir, "07_pca_X_cumulative_variance.png"), dpi=220, bbox_inches='tight')
    plt.show()

    # Нагрузки первых двух компонент
    plt.figure(figsize=(10, 5))
    comp1 = pca_X.components_[0]
    comp2 = pca_X.components_[1]
    plt.bar(range(1, n_features+1), comp1, alpha=0.7, label='PC1')
    plt.bar(range(1, n_features+1), comp2, alpha=0.7, label='PC2')
    plt.title("Нагрузки первых двух главных компонент (входные признаки)")
    plt.xlabel("Номер входного признака")
    plt.ylabel("Значение нагрузки")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "07_pca_X_loadings.png"), dpi=220, bbox_inches='tight')
    plt.show()


def plot_correlation_analysis(X, y, wavelengths, save_dir, n_channels=9, n_pca_per_channel=3):
    """
    Полный корреляционный анализ:
      - Карты корреляции между X и PCA-компонентами каналов
      - Сводная тепловая карта средней абсолютной корреляции
      - Корреляция X с каждой длиной волны
    """
    if X is None:
        return

    bounds = create_non_overlapping_bounds(wavelengths, n_channels=n_channels)
    print("\n=== АНАЛИЗ КОРРЕЛЯЦИЙ X И PCA-КОМПОНЕНТ КАНАЛОВ ===\n")
    print("Непересекающиеся каналы для корреляционного анализа:")
    for i, (low, high) in enumerate(bounds):
        mask = (wavelengths >= low) & (wavelengths <= high)
        print(f"  Канал {i}: {low:.1f}-{high:.1f} нм, {mask.sum()} точек")

    # Используем существующую функцию для детальных карт
    all_correlations = plot_correlation_maps(
        X, y, wavelengths, bounds, save_dir,
        n_pca_per_channel=n_pca_per_channel
    )

    # Дополнительные сводные графики уже внутри plot_correlation_maps,
    # поэтому здесь просто оставим вызов. Можно было бы вынести отдельно,
    # но для простоты оставлен оригинальный функционал.
    return all_correlations


def plot_correlation_maps(X, y, wavelengths, bounds, save_dir, n_pca_per_channel=3):
    """
    Создаёт карты корреляции между X и PCA-компонентами для каждого канала.
    (функция сохранена из исходного кода с небольшими правками под новый save_dir)
    """
    os.makedirs(save_dir, exist_ok=True)
    n_channels = len(bounds)
    feature_names = [f'X{i+1}' for i in range(X.shape[1])]

    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 4 * n_channels))
    if n_channels == 1:
        axes = [axes]

    all_correlations = []

    for ch_idx, ((low, high), ax) in enumerate(zip(bounds, axes)):
        mask = (wavelengths >= low) & (wavelengths <= high)
        y_channel = y[:, mask]

        if y_channel.shape[1] < 3:
            print(f"Канал {ch_idx}: {low:.1f}-{high:.1f} нм - слишком мало точек, пропускаем")
            ax.text(0.5, 0.5, f'Канал {ch_idx}: недостаточно точек',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        n_comp = min(n_pca_per_channel, y_channel.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(y_channel)
        coeffs = pca.transform(y_channel)

        corr_matrix = np.zeros((X.shape[1], n_comp))
        p_values = np.zeros((X.shape[1], n_comp))

        for i in range(X.shape[1]):
            for j in range(n_comp):
                corr, p_val = pearsonr(X[:, i], coeffs[:, j])
                corr_matrix[i, j] = corr
                p_values[i, j] = p_val

        all_correlations.append({
            'channel': ch_idx,
            'bounds': (low, high),
            'n_points': y_channel.shape[1],
            'correlations': corr_matrix,
            'p_values': p_values,
            'explained_variance': pca.explained_variance_ratio_[:n_comp]
        })

        im = ax.imshow(corr_matrix.T, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

        for i in range(X.shape[1]):
            for j in range(n_comp):
                ax.text(i, j, f'{corr_matrix[i, j]:.2f}',
                        ha="center", va="center",
                        color="black" if abs(corr_matrix[i, j]) < 0.5 else "white",
                        fontsize=9)

        ax.set_xticks(range(X.shape[1]))
        ax.set_xticklabels(feature_names)
        ax.set_yticks(range(n_comp))
        ax.set_yticklabels([f'PC{j+1}\n({pca.explained_variance_ratio_[j]*100:.1f}%)'
                            for j in range(n_comp)])
        ax.set_title(f'Канал {ch_idx}: {low:.1f} - {high:.1f} нм ({y_channel.shape[1]} точек)',
                     fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Корреляция Пирсона')

    plt.suptitle('Корреляция между входными признаками и PCA-компонентами спектральных каналов',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "08_correlation_maps_all_channels.png"),
                dpi=250, bbox_inches='tight')
    plt.show()

    # Вывод значимых корреляций
    print("\n" + "=" * 100)
    print("ЗНАЧИМЫЕ КОРРЕЛЯЦИИ МЕЖДУ X И PCA-КОМПОНЕНТАМИ (|r| > 0.3, p < 0.05)")
    print("=" * 100)
    for ch_data in all_correlations:
        ch_idx = ch_data['channel']
        low, high = ch_data['bounds']
        corr_matrix = ch_data['correlations']
        p_values = ch_data['p_values']
        n_comp = corr_matrix.shape[1]
        print(f"\nКанал {ch_idx}: {low:.1f}-{high:.1f} нм")
        print("-" * 80)
        for j in range(n_comp):
            pc_variance = ch_data['explained_variance'][j]
            significant = []
            for i in range(X.shape[1]):
                if abs(corr_matrix[i, j]) > 0.3 and p_values[i, j] < 0.05:
                    significant.append((i, corr_matrix[i, j]))
            if significant:
                print(f"  PC{j+1} (объясняет {pc_variance*100:.1f}% дисперсии):")
                for feat_idx, corr_val in sorted(significant, key=lambda x: abs(x[1]), reverse=True):
                    print(f"    X{feat_idx+1}: r = {corr_val:+.3f}")

    # Средняя абсолютная корреляция по каналам
    avg_abs_corr = np.zeros((X.shape[1], len(all_correlations)))
    channel_labels = []
    for ch_idx, ch_data in enumerate(all_correlations):
        low, high = ch_data['bounds']
        channel_labels.append(f'{low:.0f}-{high:.0f}')
        corr_matrix = ch_data['correlations']
        avg_abs_corr[:, ch_idx] = np.mean(np.abs(corr_matrix), axis=1)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    im2 = ax2.imshow(avg_abs_corr, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(channel_labels)))
    ax2.set_xticklabels(channel_labels, rotation=45, ha='right')
    ax2.set_yticks(range(X.shape[1]))
    ax2.set_yticklabels(feature_names)
    ax2.set_xlabel('Спектральный канал (нм)')
    ax2.set_ylabel('Входной признак')
    ax2.set_title('Средняя абсолютная корреляция между входными признаками и PCA-компонентами')
    for i in range(X.shape[1]):
        for j in range(len(channel_labels)):
            ax2.text(j, i, f'{avg_abs_corr[i, j]:.2f}',
                     ha="center", va="center",
                     color="black" if avg_abs_corr[i, j] < 0.5 else "white",
                     fontsize=9)
    plt.colorbar(im2, ax=ax2, label='Средняя |корреляция|')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "09_average_correlation_summary.png"), dpi=250, bbox_inches='tight')
    plt.show()

    # Корреляция X с исходными длинами волн
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    corr_X_wavelengths = np.zeros((X.shape[1], len(wavelengths)))
    for i in range(X.shape[1]):
        for j in range(len(wavelengths)):
            corr_X_wavelengths[i, j], _ = pearsonr(X[:, i], y[:, j])

    im3 = ax3.imshow(corr_X_wavelengths, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1,
                     extent=[wavelengths[0], wavelengths[-1], X.shape[1]-0.5, -0.5])
    ax3.set_yticks(range(X.shape[1]))
    ax3.set_yticklabels(feature_names)
    ax3.set_xlabel('Длина волны (нм)')
    ax3.set_ylabel('Входной признак')
    ax3.set_title('Корреляция входных признаков с интенсивностью на каждой длине волны')
    for low, high in bounds:
        ax3.axvline(x=low, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
        ax3.axvline(x=high, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
    plt.colorbar(im3, ax=ax3, label='Корреляция Пирсона')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "10_correlation_X_vs_wavelengths.png"), dpi=250, bbox_inches='tight')
    plt.show()

    return all_correlations

# ----------------------------------------------------------------------
# Дополнительные функции анализа спектров (сохранение в подпапку spectr)
# ----------------------------------------------------------------------
def plot_top5_brightest_darkest(y, wavelengths, save_dir):
    """
    Создаёт два отдельных графика:
    - 5 самых ярких спектров (по суммарной яркости)
    - 5 самых тусклых спектров
    Сохраняет в подпапку spectr.
    """
    spectr_dir = os.path.join(save_dir, "spectr")
    os.makedirs(spectr_dir, exist_ok=True)

    total_irr = np.trapezoid(y, x=wavelengths, axis=1)
    brightest_idx = np.argsort(total_irr)[-5:][::-1]
    darkest_idx = np.argsort(total_irr)[:5]

    # График 5 самых ярких
    plt.figure(figsize=(12, 7))
    for idx in brightest_idx:
        plt.plot(wavelengths, y[idx], lw=1.2, label=f'#{idx} (sum={total_irr[idx]:.1f})')
    plt.title("5 самых ярких спектров")
    plt.xlabel("Длина волны (нм)")
    plt.ylabel("Интенсивность")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(spectr_dir, "05_brightest_5_spectra.png"), dpi=250, bbox_inches='tight')
    plt.show()

    # График 5 самых тусклых
    plt.figure(figsize=(12, 7))
    for idx in darkest_idx:
        plt.plot(wavelengths, y[idx], lw=1.2, label=f'#{idx} (sum={total_irr[idx]:.1f})')
    plt.title("5 самых тусклых спектров")
    plt.xlabel("Длина волны (нм)")
    plt.ylabel("Интенсивность")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(spectr_dir, "06_darkest_5_spectra.png"), dpi=250, bbox_inches='tight')
    plt.show()


def plot_all_spectra_overlay(y, wavelengths, save_dir):
    """
    Отображает все спектры на одном графике (overlay) с низкой прозрачностью.
    Сохраняет в подпапку spectr.
    """
    spectr_dir = os.path.join(save_dir, "spectr")
    os.makedirs(spectr_dir, exist_ok=True)

    plt.figure(figsize=(14, 8))
    for i in range(y.shape[0]):
        plt.plot(wavelengths, y[i], lw=0.5, alpha=0.3, color='gray')
    # Для наглядности добавим средний спектр
    mean_spec = y.mean(axis=0)
    plt.plot(wavelengths, mean_spec, 'r', lw=2.5, label='Средний спектр', alpha=0.95)
    plt.title(f"Совмещение всех спектров (N = {y.shape[0]})")
    plt.xlabel("Длина волны (нм)")
    plt.ylabel("Интенсивность")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(spectr_dir, "07_all_spectra_overlay.png"), dpi=250, bbox_inches='tight')
    plt.show()

# ----------------------------------------------------------------------
# 4. Главная управляющая функция
# ----------------------------------------------------------------------
def analyze_spectral_dataset(X, y, wavelengths, save_dir="dataset_description"):
    """
    Запускает полный цикл анализа спектральных данных с сохранением графиков.
    """
    os.makedirs(save_dir, exist_ok=True)
    spectr_dir = os.path.join(save_dir, "spectr")
    os.makedirs(spectr_dir, exist_ok=True)          # <-- создаём подпапку заранее

    print(f"Анализируем датасет: {y.shape[0]} спектров, {y.shape[1]} точек\n")

    # 1. Распределение суммарной яркости
    total_irr = plot_total_irradiance_distribution(y, wavelengths, save_dir)

    # 2. Средний спектр ± std
    mean_spec, std_spec = plot_mean_spectrum_with_std(y, wavelengths, save_dir)

    # 3. PCA по диапазонам (только текст)
    print_pca_by_ranges(y, wavelengths)

    # 4. Топ-10 ярких/тусклых + средний (старый)
    plot_top10_bright_dark_with_mean(y, wavelengths, save_dir)

    # 5. Коэффициент вариации
    plot_coefficient_of_variation(y, wavelengths, save_dir)

    # 6. PCA полного спектра
    plot_pca_full(y, wavelengths, save_dir)

    # 7. PCA для входных признаков (если есть)
    if X is not None:
        plot_pca_X(X, save_dir)

    # 8. Корреляционный анализ (если есть X)
    if X is not None:
        plot_correlation_analysis(X, y, wavelengths, save_dir)

    # ===== НОВЫЕ ГРАФИКИ (сохраняются в подпапку spectr) =====
    # 5 самых ярких / 5 самых тусклых (отдельные графики)
    plot_top5_brightest_darkest(y, wavelengths, save_dir)

    # Наложение всех спектров
    plot_all_spectra_overlay(y, wavelengths, save_dir)

    # Итоговая статистика
    print("\n=== Краткая статистика датасета ===")
    print(f"Количество спектров:           {y.shape[0]}")
    print(f"Диапазон длин волн:            {wavelengths.min():.1f} — {wavelengths.max():.1f} нм")
    print(f"Средняя суммарная яркость:     {total_irr.mean():.2f} ± {total_irr.std():.2f}")
    if X is not None:
        print(f"Количество входных признаков:  {X.shape[1]}")
        print(f"Размерность X:                 {X.shape}")
    print(f"\nВсе графики сохранены в папку: {save_dir}/")
    print(f"Дополнительные графики спектров: {spectr_dir}/")


# ----------------------------------------------------------------------
# 5. Точка входа
# ----------------------------------------------------------------------
if __name__ == "__main__":
    DATA_PATH = r"D:\scrap-heap\Reconstruction_of_the_solar_spectrum\data_set\paired_data_cleaned\\"
    SAVE_DIR = "dataset_description/paired_data_cleaned"

    X_raw, y_raw, wavelengths = load_spectral_data(DATA_PATH)
    analyze_spectral_dataset(X_raw, y_raw, wavelengths, save_dir=SAVE_DIR)