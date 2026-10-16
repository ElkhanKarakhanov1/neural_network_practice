"""Интерактивное меню подготовки данных.

Запуск из корня репозитория:
    python -m part2_prepare.app.main

Прямой запуск:
    python part2_prepare/app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path


# Поддержка запуска через -m и непосредственно как файла.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from part2_prepare.app.cli import doctor, run_prepare
    from part2_prepare.app.config import (
        ConfigurationError,
        Settings,
    )
    from part2_prepare.app.logging_config import configure_logging
else:
    from .cli import doctor, run_prepare
    from .config import ConfigurationError, Settings
    from .logging_config import configure_logging


class MenuInputError(ValueError):
    """Некорректный ввод в интерактивном меню."""


def read_value(prompt: str, default: str) -> str:
    """Прочитать значение или вернуть значение по умолчанию."""
    entered = input(f"{prompt} [{default}]: ").strip()

    if not entered:
        return default

    return entered


def read_path(prompt: str, default: Path) -> str:
    """Прочитать путь, разрешив вставку пути в кавычках."""
    entered = read_value(prompt, str(default))

    # Пользователь может скопировать Windows-путь вместе с кавычками.
    if (
        len(entered) >= 2
        and entered[0] == entered[-1]
        and entered[0] in {'"', "'"}
    ):
        entered = entered[1:-1].strip()

    if not entered:
        raise MenuInputError("Путь не может быть пустым.")

    return entered


def confirm_overwrite(path: Path) -> bool:
    """Запросить подтверждение замены существующего файла."""
    if not path.exists():
        return True

    if not path.is_file():
        raise ConfigurationError(
            f"Выходной путь существует, но не является файлом: {path}"
        )

    while True:
        answer = input(
            f"Файл уже существует: {path}\n"
            "Перезаписать? (да/нет) [нет]: "
        ).strip().lower()

        if answer in {"да", "д", "yes", "y"}:
            return True

        if answer in {"", "нет", "н", "no", "n"}:
            return False

        print("Введите «да» или «нет».")


def prepare_interactively() -> None:
    """Запросить настройки и запустить подготовку данных."""
    defaults = Settings.from_sources()

    print("\n=== Подготовка данных ===")
    print("Enter — использовать значение в квадратных скобках.")
    print("Расширения указываются через запятую: csv или png,jpg.")
    print("CSV и изображения нужно готовить отдельными запусками.\n")

    input_path = read_path(
        "Каталог с исходными данными",
        defaults.data_path,
    )

    extensions = read_value(
        "Расширения файлов",
        ",".join(defaults.extensions),
    )

    output_path = read_path(
        "Выходной файл .npy",
        defaults.output_file,
    )

    log_level = read_value(
        "Уровень логирования",
        defaults.log_level,
    )

    # Проверка введённых значений выполняется централизованно.
    settings = Settings.from_sources(
        path=input_path,
        output=output_path,
        extensions=extensions,
        log_level=log_level,
    )

    if not settings.data_path.is_dir():
        raise ConfigurationError(
            f"Каталог данных не найден: {settings.data_path}"
        )

    if not confirm_overwrite(settings.output_file):
        print("Подготовка отменена. Существующий файл не изменён.")
        return

    configure_logging(settings.log_level)

    # Используем тот же обработчик, что и команда CLI prepare.
    run_prepare(settings)


def show_settings() -> None:
    """Показать настройки по умолчанию из окружения и config.py."""
    settings = Settings.from_sources()

    print("\n=== Настройки по умолчанию ===")
    print(f"Входной каталог: {settings.data_path}")
    print(f"Выходной файл: {settings.output_file}")
    print(f"Расширения: {', '.join(settings.extensions)}")
    print(f"Уровень логирования: {settings.log_level}")

    print(
        "\nЭти значения используются при нажатии Enter. "
        "Значения, введённые для отдельной обработки, "
        "не записываются в .env."
    )


def print_menu() -> None:
    print(
        "\n"
        "=== Подготовка данных: задание 2 ===\n"
        "1. Подготовить данные и сохранить .npy\n"
        "2. Проверить окружение (doctor)\n"
        "3. Показать настройки по умолчанию\n"
        "0. Выход"
    )


def main() -> int:
    while True:
        try:
            print_menu()
            choice = input("Выберите действие: ").strip()

            if choice == "0":
                print("Работа завершена.")
                return 0

            if choice == "1":
                prepare_interactively()

            elif choice == "2":
                result = doctor()

                if result != 0:
                    print(
                        "\nДиагностика обнаружила отсутствие NumPy. "
                        "Установите зависимости перед подготовкой данных."
                    )

            elif choice == "3":
                show_settings()

            else:
                raise MenuInputError(
                    "Неизвестный пункт. Введите 0, 1, 2 или 3."
                )

        except KeyboardInterrupt:
            # Ctrl+C завершает работу, включая текущую обработку.
            print("\nРабота прервана пользователем.")
            return 130

        except EOFError:
            print("\nВвод завершён. Выход.")
            return 0

        except (
            ConfigurationError,
            OSError,
            ValueError,
            RuntimeError,
            ImportError,
            KeyError,
            TypeError,
        ) as exc:
            print(f"\nОшибка: {exc}")
            print("Исправьте настройки и повторите действие.")


if __name__ == "__main__":
    raise SystemExit(main())