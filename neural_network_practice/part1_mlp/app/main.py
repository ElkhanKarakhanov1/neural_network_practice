from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from .exceptions import MLPError, InvalidInputError, MismatchedDataError
    from .manager import DatasetManager
    from .models import NeuralNetwork
    from .utils import (
        format_predictions, load_numeric_matrix, mean_squared_error,
        parse_float_list, parse_layer_sizes, validate_file,
    )
except ImportError:  # direct execution: python app/main.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from part1_mlp.app.exceptions import MLPError, InvalidInputError, MismatchedDataError
    from part1_mlp.app.manager import DatasetManager
    from part1_mlp.app.models import NeuralNetwork
    from part1_mlp.app.utils import (
        format_predictions, load_numeric_matrix, mean_squared_error,
        parse_float_list, parse_layer_sizes, validate_file,
    )


DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "data" / "input" / "dataset.csv"


class MLPApplication:
    def __init__(self):
        self.network = None
        self.dataset = DatasetManager()

    def create_network(self):
        sizes = parse_layer_sizes(input("Размеры слоёв через запятую: "))
        activation = input("Активация (relu/sigmoid) [relu]: ").strip().lower() or "relu"
        self.network = NeuralNetwork(sizes, activation)
        print(f"Сеть создана: {sizes}, активация: {activation}")

    def load_data(self):
        entered = input(f"Путь к CSV [{DEFAULT_DATASET}]: ").strip()
        path = entered or str(DEFAULT_DATASET)
        self.dataset.load_csv(path)
        ratio = input("Доля test [0.2]: ").strip()
        test_ratio = float(ratio) if ratio else 0.2
        self.dataset.split(test_ratio, seed=42)
        print(f"Загружено примеров: {len(self.dataset.X)}; train: {len(self.dataset.X_train)}; test: {len(self.dataset.X_test)}")
        print(f"Признаки: {self.dataset.feature_names}; цель: {self.dataset.target_name}")

    def train(self):
        if self.network is None:
            raise InvalidInputError("Сначала создайте сеть.")
        if self.dataset.X_train is None:
            raise InvalidInputError("Сначала загрузите данные.")
        if self.network.layer_sizes[0] != self.dataset.X_train.shape[1]:
            raise MismatchedDataError("Входной слой не совпадает с количеством признаков.")
        if self.network.layer_sizes[-1] != self.dataset.y_train.shape[1]:
            raise MismatchedDataError("Выходной слой не совпадает с целью.")
        learning_rate = float(input("Скорость обучения [0.01]: ") or "0.01")
        epochs = int(input("Количество эпох [100]: ") or "100")
        batch_size = int(input("Размер батча [32]: ") or "32")
        history = self.network.train(self.dataset.X_train, self.dataset.y_train, epochs, learning_rate, batch_size)
        print(f"Обучение завершено. Последняя ошибка: {history[-1]:.6f}")
        if self.dataset.X_test is not None:
            print(f"Ошибка на test: {mean_squared_error(self.dataset.y_test, self.network.predict(self.dataset.X_test)):.6f}")

    def predict(self):
        if self.network is None:
            raise InvalidInputError("Сначала создайте или загрузите сеть.")
        if self.dataset.means is None:
            raise InvalidInputError("Сначала загрузите CSV для параметров нормализации.")
        choice = input("Ввод данных вручную (1) или из файла (2) [1]: ").strip() or "1"
        if choice == "1":
            rows = []
            print("Введите признаки через запятую. Пустая строка завершает ввод.")
            while True:
                line = input(f"Пример {len(rows) + 1}: ").strip()
                if not line:
                    break
                rows.append(parse_float_list(line))
            if not rows:
                raise InvalidInputError("Не введено ни одного примера.")
            raw = np.vstack(rows)
        elif choice == "2":
            raw = load_numeric_matrix(input("Путь к файлу: ").strip())
        else:
            raise InvalidInputError("Неизвестный вариант ввода.")
        if raw.shape[1] != len(self.dataset.feature_names):
            raise MismatchedDataError("Число признаков в примере не совпадает с CSV.")
        print("Предсказания:", format_predictions(self.network.predict(self.dataset.transform(raw))))

    def save_model(self):
        if self.network is None:
            raise InvalidInputError("Сеть ещё не создана.")
        path = input("Файл модели [model.json]: ").strip() or "model.json"
        self.network.save(path)
        print(f"Модель сохранена: {path}")

    def load_model(self):
        path = validate_file(input("Файл модели: ").strip(), {".json"})
        self.network = NeuralNetwork.load(path)
        print(f"Модель загружена: {path}; слои: {self.network.layer_sizes}")

    def plot_loss(self):
        if self.network is None or not self.network.loss_history:
            raise InvalidInputError("История обучения отсутствует.")
        if plt is None:
            raise InvalidInputError("Для графика установите matplotlib.")
        plt.plot(range(1, len(self.network.loss_history) + 1), self.network.loss_history)
        plt.xlabel("Эпоха")
        plt.ylabel("MSE")
        plt.title("История ошибки")
        plt.grid(True)
        plt.show()


def print_menu():
    print("""
=== Neural Network ===
1. Создать сеть
2. Загрузить данные из CSV
3. Обучить сеть
4. Предсказание
5. Сохранить веса
6. Загрузить веса
7. Показать график ошибки
0. Выход""")


def main():
    app = MLPApplication()
    actions = {"1": app.create_network, "2": app.load_data, "3": app.train,
               "4": app.predict, "5": app.save_model, "6": app.load_model, "7": app.plot_loss}
    while True:
        print_menu()
        choice = input("Выберите действие: ").strip()
        if choice == "0":
            print("Выход.")
            return
        action = actions.get(choice)
        if action is None:
            print("Ошибка: выберите пункт от 0 до 7.")
            continue
        try:
            action()
        except (MLPError, ValueError, OSError, TypeError) as exc:
            print(f"Ошибка: {exc}")


if __name__ == "__main__":
    main()