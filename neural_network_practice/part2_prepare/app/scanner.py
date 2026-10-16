"""Рекурсивный поиск файлов и подготовка массивов NumPy."""

from __future__ import annotations

import csv
import logging
import os
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np


LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)

SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | {".csv"}


class PreparationError(ValueError):
    """Ошибка поиска, чтения или преобразования данных."""


def find_files(
    input_dir: Path,
    extensions: tuple[str, ...],
) -> list[Path]:
    """Рекурсивно найти файлы с указанными расширениями."""
    input_dir = Path(input_dir)

    if not input_dir.is_dir():
        raise PreparationError(
            f"Каталог данных не найден: {input_dir}"
        )

    normalized_extensions = set()

    for extension in extensions:
        extension = extension.strip().lower()

        if not extension:
            continue

        if not extension.startswith("."):
            extension = f".{extension}"

        if extension not in SUPPORTED_EXTENSIONS:
            raise PreparationError(
                f"Неподдерживаемое расширение: {extension}"
            )

        normalized_extensions.add(extension)

    if not normalized_extensions:
        raise PreparationError(
            "Не указаны расширения для поиска."
        )

    files = []

    def handle_walk_error(error: OSError) -> None:
        raise PreparationError(
            f"Не удалось прочитать каталог: {error}"
        ) from error

    # Не переходим по символьным ссылкам на каталоги.
    for directory, subdirectories, filenames in os.walk(
        input_dir,
        followlinks=False,
        onerror=handle_walk_error,
    ):
        subdirectories.sort()

        for filename in sorted(filenames):
            path = Path(directory) / filename

            if path.suffix.lower() in normalized_extensions:
                files.append(path)

    if not files:
        raise PreparationError(
            f"В каталоге {input_dir} не найдены файлы "
            f"с расширениями: {', '.join(sorted(normalized_extensions))}"
        )

    return files


def read_csv_file(
    path: Path,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Прочитать CSV с заголовком и числовыми значениями."""
    records = []
    line_number = 1

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.reader(file, delimiter=",", strict=True)
            header_row = next(reader, None)

            if header_row is None:
                raise PreparationError(
                    f"CSV-файл пуст: {path}"
                )

            header = tuple(value.strip() for value in header_row)

            if len(header) < 2:
                raise PreparationError(
                    f"{path}: нужны минимум два столбца — "
                    "признак и целевая переменная."
                )

            if any(not name for name in header):
                raise PreparationError(
                    f"{path}: заголовки столбцов не должны быть пустыми."
                )

            if len(set(header)) != len(header):
                raise PreparationError(
                    f"{path}: названия столбцов должны быть уникальными."
                )

            for row in reader:
                line_number = reader.line_num

                # Разрешаем полностью пустые строки.
                if not row or all(not value.strip() for value in row):
                    continue

                if len(row) != len(header):
                    raise PreparationError(
                        f"{path}, строка {line_number}: "
                        f"ожидалось {len(header)} столбцов, "
                        f"получено {len(row)}."
                    )

                try:
                    numeric_row = [float(value) for value in row]
                except ValueError as exc:
                    raise PreparationError(
                        f"{path}, строка {line_number}: "
                        "найдены нечисловые или пустые значения."
                    ) from exc

                records.append(numeric_row)

    except (OSError, UnicodeError, csv.Error) as exc:
        raise PreparationError(
            f"Не удалось прочитать CSV {path}: {exc}"
        ) from exc

    if not records:
        raise PreparationError(
            f"{path}: нет строк с данными."
        )

    try:
        array = np.asarray(records, dtype=np.float64)
    except (ValueError, OverflowError) as exc:
        raise PreparationError(
            f"{path}: не удалось преобразовать числа в массив."
        ) from exc

    if not np.isfinite(array).all():
        raise PreparationError(
            f"{path}: данные содержат NaN или бесконечность."
        )

    return header, array


def prepare_csv_files(files: list[Path]) -> np.ndarray:
    """Объединить таблицы и нормализовать признаки.

    Последний столбец сохраняется без нормализации.
    """
    expected_header = None
    tables = []

    for path in files:
        header, table = read_csv_file(path)

        if expected_header is None:
            expected_header = header
        elif header != expected_header:
            raise PreparationError(
                f"Заголовки CSV не совпадают: {path}. "
                f"Ожидается {expected_header}, получено {header}."
            )

        tables.append(table)
        LOGGER.info("CSV: %s; форма: %s", path, table.shape)

    combined = np.concatenate(tables, axis=0)

    features = combined[:, :-1]
    targets = combined[:, -1:]

    minimum = features.min(axis=0)
    maximum = features.max(axis=0)

    # Защита от переполнения для экстремальных числовых значений.
    with np.errstate(over="ignore", invalid="ignore"):
        ranges = maximum - minimum

    if not np.isfinite(ranges).all():
        raise PreparationError(
            "Слишком большой диапазон значений CSV для нормализации."
        )

    # Постоянный признак преобразуется в нулевой столбец.
    safe_ranges = np.where(ranges == 0, 1.0, ranges)

    normalized_features = (features - minimum) / safe_ranges

    with np.errstate(over="ignore", invalid="ignore"):
        result = np.concatenate(
            [normalized_features, targets],
            axis=1,
        ).astype(np.float32)

    if not np.isfinite(result).all():
        raise PreparationError(
            "Данные нельзя представить конечными числами float32."
        )

    return result


def get_image_size() -> tuple[int, int]:
    """Получить размер изображений из окружения.

    PIL использует порядок (ширина, высота).
    """
    try:
        width = int(os.getenv("PREPARE_IMAGE_WIDTH", "64"))
        height = int(os.getenv("PREPARE_IMAGE_HEIGHT", "64"))
    except ValueError as exc:
        raise PreparationError(
            "PREPARE_IMAGE_WIDTH и PREPARE_IMAGE_HEIGHT "
            "должны быть целыми числами."
        ) from exc

    if width <= 0 or height <= 0:
        raise PreparationError(
            "Ширина и высота изображений должны быть положительными."
        )

    return width, height


def prepare_image_files(files: list[Path]) -> np.ndarray:
    """Прочитать изображения и вернуть массив (N, H, W, 3)."""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise PreparationError(
            "Для изображений установите Pillow: pip install Pillow"
        ) from exc

    size = get_image_size()
    images = []

    for path in files:
        try:
            with Image.open(path) as source:
                original_size = source.size

                # Учитываем ориентацию фотографии из EXIF.
                oriented = ImageOps.exif_transpose(source)
                rgb = oriented.convert("RGB")
                resized = rgb.resize(
                    size,
                    resample=Image.Resampling.BILINEAR,
                )

                array = np.asarray(
                    resized,
                    dtype=np.float32,
                ) / 255.0

        except (
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ) as exc:
            raise PreparationError(
                f"Не удалось обработать изображение {path}: {exc}"
            ) from exc

        images.append(array)

        LOGGER.info(
            "Изображение: %s; исходный размер: %s; форма: %s",
            path,
            original_size,
            array.shape,
        )

    return np.stack(images, axis=0)


def save_array(array: np.ndarray, output_file: Path) -> None:
    """Сохранить .npy через временный файл.

    Готовый файл заменяет целевой только после успешной записи.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_file.parent,
            prefix=f".{output_file.stem}_",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            np.save(
                temporary_file,
                array,
                allow_pickle=False,
            )

        os.replace(temporary_path, output_file)

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def prepare_dataset(
    input_dir: Path,
    extensions: tuple[str, ...],
    output_file: Path,
) -> dict:
    """Найти файлы, подготовить массив и сохранить результат.

    Возвращаемые ключи совместимы с ранее созданным cli.py.
    """
    started = perf_counter()

    input_dir = Path(input_dir)
    output_file = Path(output_file)

    if output_file.suffix.lower() != ".npy":
        raise PreparationError(
            "Выходной файл должен иметь расширение .npy."
        )

    files = find_files(input_dir, extensions)

    has_csv = any(path.suffix.lower() == ".csv" for path in files)
    has_images = any(
        path.suffix.lower() in IMAGE_EXTENSIONS
        for path in files
    )

    if has_csv and has_images:
        raise PreparationError(
            "Найдены одновременно CSV и изображения. "
            "Подготовьте их отдельными командами "
            "с разными выходными файлами."
        )

    input_bytes = sum(path.stat().st_size for path in files)

    if has_csv:
        array = prepare_csv_files(files)
    else:
        array = prepare_image_files(files)

    save_array(array, output_file)

    elapsed = perf_counter() - started

    LOGGER.info(
        "Обработка завершена: файлов=%d, форма=%s, "
        "тип=%s, время=%.3f с",
        len(files),
        array.shape,
        array.dtype,
        elapsed,
    )

    return {
        "file_count": len(files),
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
        "input_bytes": input_bytes,
        "elapsed_seconds": elapsed,
    }