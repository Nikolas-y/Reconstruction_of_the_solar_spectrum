import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_spectral_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if len(lines) < 4:
        raise ValueError(f"Файл {file_path} имеет меньше 4 строк")
    x = np.array(list(map(float, lines[1].strip().split())))
    wavelengths = np.array(list(map(float, lines[2].strip().split())))
    y = np.array(list(map(float, lines[3].strip().split())))
    return x, y, wavelengths, os.path.basename(file_path)


def group_files_by_x(data_dir, tolerance=1e-6):
    files = [f for f in os.listdir(data_dir) if f.startswith('paired_') and f.endswith('.txt')]
    if not files:
        return []

    x_list = []
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        try:
            x, _, _, _ = load_spectral_file(fpath)
            x_list.append((fpath, x))
        except Exception as e:
            print(f"Ошибка при загрузке {fname}: {e}")
            continue

    # Группировка
    used = [False] * len(x_list)
    groups = []
    for i in range(len(x_list)):
        if used[i]:
            continue
        group = [x_list[i][0]]
        for j in range(i+1, len(x_list)):
            if used[j]:
                continue
            if np.allclose(x_list[i][1], x_list[j][1], atol=tolerance, rtol=0):
                group.append(x_list[j][0])
                used[j] = True
        used[i] = True
        if len(group) > 1:
            groups.append(group)
    return groups

def plot_group_comparisons(group_paths, output_dir, wavelengths, tolerance=1e-6):
    if len(group_paths) < 2:
        return

    data = []
    for fpath in group_paths:
        _, y, wl, fname = load_spectral_file(fpath)
        if data and not np.allclose(wl, data[0]['wl']):
            print(f"Предупреждение: длины волн в {fname} отличаются от первого файла, использую общие.")
        data.append({'y': y, 'fname': fname})
    wl_common = wavelengths if wavelengths is not None else data[0]['wl']

    x_first, _, _, _ = load_spectral_file(group_paths[0])
    x_str = '_'.join([f"{val:.4f}" for val in x_first[:3]])  # сокращённое имя
    group_dir = os.path.join(output_dir, f"group_{x_str}_size{len(group_paths)}")
    os.makedirs(group_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    for idx, d in enumerate(data):
        plt.plot(wl_common, d['y'], color=colors[idx], linewidth=1.5,
                 label=f"{d['fname']}")
    plt.xlabel("Длина волны (нм)")
    plt.ylabel("Интенсивность")
    plt.title(f"Все спектры группы (X: {x_first})")
    plt.legend(fontsize=8, loc='best')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(group_dir, "all_spectra.png"), dpi=200, bbox_inches='tight')
    plt.close()

    first = data[0]
    for idx in range(1, len(data)):
        other = data[idx]
        plt.figure(figsize=(12, 6))
        plt.plot(wl_common, first['y'], color='black', linestyle='-', linewidth=2,
                 label=f"{first['fname']}")
        plt.plot(wl_common, other['y'], color='red', linestyle='--', linewidth=2,
                 label=f"{other['fname']}")
        plt.xlabel("Длина волны (нм)")
        plt.ylabel("Интенсивность")
        plt.title(f"Сравнение: {first['fname']} (чёрный) vs {other['fname']} (красный пунктир)")
        plt.legend()
        plt.grid(alpha=0.3)
        out_name = f"compare_{first['fname'].replace('.txt','')}_vs_{other['fname'].replace('.txt','')}.png"
        plt.savefig(os.path.join(group_dir, out_name), dpi=200, bbox_inches='tight')
        plt.close()

    print(f"Группа из {len(group_paths)} файлов обработана, сохранено в {group_dir}")

# 4. Основная функция
def find_all_duplicate_x_and_plot(data_dir, output_dir, tolerance=1e-6):

    if not os.path.exists(data_dir):
        print(f"Папка {data_dir} не существует!")
        return
    os.makedirs(output_dir, exist_ok=True)

    print("Группировка файлов по X...")
    groups = group_files_by_x(data_dir, tolerance)
    print(f"Найдено групп с повторяющимися X: {len(groups)}")

    if not groups:
        print("Нет групп. Завершение.")
        return

    # Для загрузки длин волн (они одинаковы для всех файлов)
    _, _, wavelengths, _ = load_spectral_file(groups[0][0])

    for idx, group in enumerate(groups):
        print(f"Обработка группы {idx+1}/{len(groups)} (размер {len(group)})")
        plot_group_comparisons(group, output_dir, wavelengths, tolerance)

    print(f"Все графики сохранены в {output_dir}")

# main
if __name__ == "__main__":
    DATA_DIR = r"D:\scrap-heap\Reconstruction_of_the_solar_spectrum\data_set\paired_data"
    OUT_DIR = r"D:\scrap-heap\Reconstruction_of_the_solar_spectrum\data_set\duplicate_x_full_analysis"

    find_all_duplicate_x_and_plot(DATA_DIR, OUT_DIR, tolerance=1e-6)
    print("Анализ завершён!")