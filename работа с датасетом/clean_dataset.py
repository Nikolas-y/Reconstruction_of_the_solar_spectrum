import os
import numpy as np
from sympy import false


def load_spectral_data_correct(data_path):
    X_list = []
    y_list = []
    wavelengths = None
    filenames = []

    files = [f for f in os.listdir(data_path) if f.startswith('paired_') and f.endswith('.txt')]
    for fname in files:
        with open(os.path.join(data_path, fname), 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if len(lines) < 4:
            print(f"Пропущен {fname}: меньше 4 строк")
            continue

        # Строка 1 (индекс 0) - длины волн каналов (пропускаем)
        # Строка 2 (индекс 1) - значения каналов (X)
        try:
            x_vals = list(map(float, lines[1].strip().split()))
        except:
            print(f"Ошибка в строке X файла {fname}")
            continue

        # Строка 3 (индекс 2) - длины волн спектра
        try:
            wl_vals = list(map(float, lines[2].strip().split()))
        except:
            print(f"Ошибка в строке длин волн файла {fname}")
            continue

        # Строка 4 (индекс 3) - значения спектра (y)
        try:
            y_vals = list(map(float, lines[3].strip().split()))
        except:
            print(f"Ошибка в строке спектра файла {fname}")
            continue

        if len(x_vals) != 9 or len(y_vals) != 1904:
            print(f"Пропущен {fname}: неверная размерность (x={len(x_vals)}, y={len(y_vals)})")
            continue

        X_list.append(x_vals)
        y_list.append(y_vals)
        filenames.append(fname)

        if wavelengths is None:
            wavelengths = np.array(wl_vals)
        else:
            # Проверка, что длины волн совпадают (опционально)
            if not np.allclose(wavelengths, wl_vals):
                print(f"Предупреждение: длины волн в {fname} отличаются от первого файла")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"Загружено {len(X)} образцов, длина спектра: {y.shape[1]}")
    return X, y, wavelengths, filenames


def save_cleaned_data(output_dir, filenames, X_clean, y_clean, channel_wavelengths, spectral_wavelengths):
    """
    Сохраняет очищенные данные в исходном формате (4 строки).
    channel_wavelengths: длины волн каналов (9 значений) – берём из первого файла.
    spectral_wavelengths: длины волн спектра (1904 значения) – из load.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Преобразуем channel_wavelengths в строку
    channel_wl_str = ' '.join(f'{w:.6f}' for w in channel_wavelengths)
    spectral_wl_str = ' '.join(f'{w:.6f}' for w in spectral_wavelengths)

    for fname, x_row, y_row in zip(filenames, X_clean, y_clean):
        out_path = os.path.join(output_dir, fname)
        with open(out_path, 'w', encoding='utf-8') as f:
            # Строка 1: длины волн каналов (9)
            f.write(channel_wl_str + '\n')
            # Строка 2: значения каналов (X)
            f.write(' '.join(f'{val:.6f}' for val in x_row) + '\n')
            # Строка 3: длины волн спектра
            f.write(spectral_wl_str + '\n')
            # Строка 4: значения спектра (y)
            f.write(' '.join(f'{val:.6f}' for val in y_row) + '\n')

    print(f"Сохранено {len(filenames)} файлов в {output_dir}")


def clean_dataset_by_intensity(
        input_dir,
        output_dir,
        low_threshold=100000,
        high_threshold=550000,
        use_integral=False
):

    print(f"Загрузка данных из: {input_dir}")
    X, y, spectral_wl, filenames = load_spectral_data_correct(input_dir)
    print(f"Загружено образцов: {X.shape[0]}")

    # Вычисляем суммарную яркость
    if use_integral:
        # Используем метод трапеций (учитываем неравномерную сетку, если нужна точность)
        total_intensity = np.trapz(y, x=spectral_wl, axis=1)
    else:
        total_intensity = np.sum(y, axis=1)

    # Статистика
    print(
        f"Суммарная яркость: min={total_intensity.min():.2f}, max={total_intensity.max():.2f}, mean={total_intensity.mean():.2f}")
    print(f"Нижний порог: {low_threshold}, верхний порог: {high_threshold}")

    # Фильтрация
    mask = (total_intensity >= low_threshold) & (total_intensity <= high_threshold)
    n_removed = np.sum(~mask)
    print(f"Удалено образцов: {n_removed} ({n_removed / len(mask) * 100:.1f}%)")

    X_clean = X[mask]
    y_clean = y[mask]
    filenames_clean = [filenames[i] for i in range(len(filenames)) if mask[i]]

    # Сохраняем
    # Чтобы получить длины волн каналов (9 значений), возьмём их из первого файла исходной папки
    # Просто прочитаем первый попавшийся файл и извлечём первую строку
    first_file = None
    for f in os.listdir(input_dir):
        if f.startswith('paired_') and f.endswith('.txt'):
            first_file = f
            break
    if first_file is None:
        raise ValueError("Не найден ни один файл в исходной папке")
    with open(os.path.join(input_dir, first_file), 'r') as f:
        channel_wl_line = f.readline().strip()
    channel_wavelengths = np.array(list(map(float, channel_wl_line.split())))
    if len(channel_wavelengths) != 9:
        raise ValueError(f"Длины волн каналов имеют размер {len(channel_wavelengths)}, ожидалось 9")

    save_cleaned_data(output_dir, filenames_clean, X_clean, y_clean, channel_wavelengths, spectral_wl)

    # Лог удалённых файлов
    removed_log = os.path.join(output_dir, "removed_files.txt")
    with open(removed_log, 'w', encoding='utf-8') as f:
        for i, keep in enumerate(mask):
            if not keep:
                f.write(f"{filenames[i]}\t{total_intensity[i]:.2f}\n")
    print(f"Список удалённых файлов сохранён в {removed_log}")

    return X_clean, y_clean, spectral_wl


if __name__ == "__main__":
    INPUT_DIR = r"D:\scrap-heap\Reconstruction_of_the_solar_spectrum\data_set\paired_data"
    OUTPUT_DIR = r"D:\scrap-heap\Reconstruction_of_the_solar_spectrum\data_set\paired_data_cleaned"

    clean_dataset_by_intensity(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        low_threshold=500000,
        high_threshold=2250000,
        use_integral=False  # True для численного интегрирования трапецией, False для суммы
    )
    print("Очистка завершена!")