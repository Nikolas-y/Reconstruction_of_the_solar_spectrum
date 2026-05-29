import numpy as np
import os
from datetime import datetime

# Пути к папкам
path_type1 = r"D:\scrap-heap\Reconstruction_of_the_solar_spectrum\data_set\FSR05_spectr\\"
path_type2 = r"D:\scrap-heap\Reconstruction_of_the_solar_spectrum\data_set\multi_spectr\\"

# Папка для результатов
output_path = r"D:\scrap-heap\Reconstruction_of_the_solar_spectrum\data_set\paired_data\\"
os.makedirs(output_path, exist_ok=True)

# Функции парсинга времени
def parse_time_type1(filename):
    base_name = filename.split('#')[0]
    parts = base_name.split('_')

    date_str = parts[0]
    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    hour = int(parts[1])
    minute = int(parts[2])
    second = int(parts[3])
    ms = int(parts[4])

    return datetime(year, month, day, hour, minute, second, ms * 1000)


def parse_time_type2(filename):
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')

    if len(parts) < 4 or parts[0] != 'spectrum':
        return None

    date_str = parts[1]
    time_str = parts[2]
    us_str = parts[3]

    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    hour = int(time_str[0:2])
    minute = int(time_str[2:4])
    second = int(time_str[4:6])

    microseconds = int(us_str)

    return datetime(year, month, day, hour, minute, second, microseconds)


# Собираем и сортируем файлы
files_type1 = [f for f in os.listdir(path_type1) if f.lower().endswith('.asc')]
files_type2 = [f for f in os.listdir(path_type2) if os.path.isfile(os.path.join(path_type2, f))]

type1_list = []
for fname in files_type1:
    try:
        t = parse_time_type1(fname)
        type1_list.append((t, fname))
    except Exception as e:
        print(f"Ошибка парсинга типа 1 ({fname}): {e}")

type1_list.sort(key=lambda x: x[0])

type2_list = []
for fname in files_type2:
    try:
        t = parse_time_type2(fname)
        if t:
            type2_list.append((t, fname))
    except Exception as e:
        print(f"Ошибка парсинга типа 2 ({fname}): {e}")

type2_list.sort(key=lambda x: x[0])

available = set(range(len(type1_list)))

# поиск пар 0.5 = 0.5 секунд разница по времени
pairs = []

for t2, f2 in type2_list:
    min_diff = float('inf')
    best_idx = None

    for idx in list(available):
        t1, _ = type1_list[idx]
        diff = abs((t2 - t1).total_seconds())
        if diff < 0.5 and diff < min_diff:
            min_diff = diff
            best_idx = idx

    if best_idx is not None:
        pairs.append((type1_list[best_idx][1], f2, type1_list[best_idx][0]))
        available.remove(best_idx)

print(f"Найдено пар: {len(pairs)}\n")

# гаусианы
MULTI_WAVELENGTHS = [415, 445, 480, 515, 555, 590, 630, 680, 912]

for f1, f2, time1 in pairs:
    print(f"Обрабатываем: {f1} → {f2}")

    # Загружаем спектр из FSR05
    data1 = np.loadtxt(os.path.join(path_type1, f1))
    wavelengths = data1[:, 0]
    intensities = data1[:, 1]

    # Обрезаем диапазон 400–900 нм
    mask = (wavelengths >= 400) & (wavelengths <= 900)

    wl_trimmed = wavelengths[mask]
    src_trimmed = intensities[mask]

    if len(wl_trimmed) == 0:
        print(f"   Ошибка: после обрезки не осталось точек → пропускаем")
        continue

    print(f"   После обрезки: {len(wl_trimmed)} точек "
          f"({wl_trimmed[0]:.2f} – {wl_trimmed[-1]:.2f} нм)")

    # Читаем мультиспектральный файл
    with open(os.path.join(path_type2, f2), 'r', encoding='utf-8') as file:
        lines = file.readlines()

    multi_values = []
    for line in lines:
        line = line.strip()
        if 'nm:' in line:
            try:
                # Извлекаем числовое значение после двоеточия
                value = float(line.split(':')[1].strip())
                multi_values.append(int(round(value)))
            except:
                continue

    if len(multi_values) != 9:
        print(f"   Предупреждение: найдено {len(multi_values)} значений вместо 9 → пропускаем")
        continue

    # Формируем имя файла
    date_str = time1.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_filename = f"paired_{date_str}.txt"
    out_path = os.path.join(output_path, out_filename)

    # Запись в новом порядке
    with open(out_path, 'w', encoding='utf-8') as f:
        # 1. Длины волн мультиспектра
        f.write(' '.join(map(str, MULTI_WAVELENGTHS)) + '\n')
        # 2. Значения мультиспектра
        f.write(' '.join(map(str, multi_values)) + '\n')
        # 3. Длины волн спектрометра (обрезанные)
        f.write(' '.join(f"{w:.6f}" for w in wl_trimmed) + '\n')
        # 4. Интенсивности спектрометра
        f.write(' '.join(map(str, src_trimmed.astype(int))) + '\n')

    print(f"   ✓ Создан: {out_filename}  ({len(wl_trimmed)} точек)")

print(f"\nГотово! Обработано пар: {len(pairs)}")