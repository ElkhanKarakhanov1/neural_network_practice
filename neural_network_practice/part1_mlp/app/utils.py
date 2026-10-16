from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from exceptions import InvalidInputError, MismatchedDataError


def parse_layer_sizes(value: str) -> list[int]:
    """Parse a string such as ``4,8,1`` into positive layer sizes."""
    try:
        sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise InvalidInputError("Размеры слоёв должны быть целыми числами.") from exc
    if len(sizes) < 3 or any(size <= 0 for size in sizes):
        raise InvalidInputError("Укажите минимум три положительных размера слоёв.")
    return sizes


def parse_float_list(value: str) -> np.ndarray:
    """Parse comma- or whitespace-separated numeric values."""
    try:
        values = [float(part) for part in value.replace(",", " ").split()]
    except ValueError as exc:
        raise InvalidInputError("Ожидались числовые значения.") from exc
    if not values:
        raise InvalidInputError("Список значений не может быть пустым.")
    return np.asarray(values, dtype=float)


def validate_file(path: str | Path, suffixes: Iterable[str] | None = None) -> Path:
    """Return an existing regular file, optionally checking its extension."""
    result = Path(path).expanduser()
    if not result.is_file():
        raise InvalidInputError(f"Файл не найден: {result}")
    if suffixes and result.suffix.lower() not in {suffix.lower() for suffix in suffixes}:
        raise InvalidInputError(f"Недопустимое расширение файла: {result.suffix}")
    return result


def ensure_2d(values, expected_columns: int | None = None) -> np.ndarray:
    """Convert values to a two-dimensional float array."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MismatchedDataError("Данные должны быть числовыми.") from exc
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] == 0:
        raise MismatchedDataError("Ожидался непустой двумерный массив данных.")
    if expected_columns is not None and array.shape[1] != expected_columns:
        raise MismatchedDataError(
            f"Ожидалось столбцов: {expected_columns}, получено: {array.shape[1]}."
        )
    return array


def mean_squared_error(actual, predicted) -> float:
    """Calculate mean squared error after checking matching shapes."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise MismatchedDataError("Размерности фактических и предсказанных значений различаются.")
    return float(np.mean((actual - predicted) ** 2))


def load_numeric_matrix(path: str | Path, delimiter: str = ",") -> np.ndarray:
    """Load a numeric matrix, useful for prediction files without headers."""
    path = validate_file(path)
    try:
        matrix = np.loadtxt(path, delimiter=delimiter, ndmin=2)
    except (OSError, ValueError) as exc:
        raise InvalidInputError(f"Не удалось прочитать числовой файл: {path}") from exc
    return ensure_2d(matrix)


def format_predictions(predictions) -> str:
    """Format prediction values for readable console output."""
    array = np.asarray(predictions)
    return np.array2string(array, precision=6, suppress_small=False)