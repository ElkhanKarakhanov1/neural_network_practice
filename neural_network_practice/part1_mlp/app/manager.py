

from __future__ import annotations

from pathlib import Path
import csv
import numpy as np

from exceptions import DataFileError, MismatchedDataError


class DatasetManager:
    def __init__(self):
        self.feature_names = []
        self.target_name = None
        self.means = None
        self.stds = None
        self.X = self.y = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def load_csv(self, path):
        path = Path(path)
        if not path.is_file():
            raise DataFileError(f"Файл не найден: {path}")
        try:
            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.reader(file))
            if len(rows) < 2 or len(rows[0]) < 2:
                raise MismatchedDataError("CSV должен содержать заголовок и минимум один признак.")
            data = np.asarray(rows[1:], dtype=float)
        except (ValueError, UnicodeError) as exc:
            raise DataFileError("CSV содержит нечисловые или некорректные данные.") from exc
        if data.ndim != 2 or data.shape[1] != len(rows[0]):
            raise MismatchedDataError("Строки CSV имеют разное число столбцов.")
        self.feature_names, self.target_name = rows[0][:-1], rows[0][-1]
        self.X, self.y = data[:, :-1], data[:, -1].reshape(-1, 1)
        self.means = self.X.mean(axis=0)
        self.stds = self.X.std(axis=0)
        self.stds[self.stds == 0] = 1
        self.X = (self.X - self.means) / self.stds
        return self.X, self.y

    def split(self, test_ratio=0.2, seed=None):
        if self.X is None:
            raise DataFileError("Сначала загрузите CSV-файл.")
        if not 0 < test_ratio < 1:
            raise ValueError("test_ratio должен быть между 0 и 1.")
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.X))
        test_size = max(1, int(len(indices) * test_ratio))
        test_idx, train_idx = indices[:test_size], indices[test_size:]
        self.X_train, self.X_test = self.X[train_idx], self.X[test_idx]
        self.y_train, self.y_test = self.y[train_idx], self.y[test_idx]
        return self.X_train, self.X_test, self.y_train, self.y_test

    def transform(self, X):
        if self.means is None:
            raise DataFileError("Сначала загрузите CSV-файл.")
        X = np.asarray(X, dtype=float)
        return (X - self.means) / self.stds