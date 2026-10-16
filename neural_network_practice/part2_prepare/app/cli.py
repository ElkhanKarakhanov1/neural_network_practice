"""CLI подготовки данных и диагностики окружения.

Запуск из корня репозитория:
    python -m part2_prepare.app.cli doctor
    python -m part2_prepare.app.cli prepare
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from time import perf_counter


# Поддержка и запуска через -m, и прямого запуска файла.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from part2_prepare.app.config import (
        ConfigurationError,
        Settings,
    )
else:
    from .config import ConfigurationError, Settings


LOGGER = logging.getLogger(__name__)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="part2_prepare",
        description="Подготовка наборов данных и диагностика окружения.",
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    prepare = commands.add_parser(
        "prepare",
        help="Рекурсивно собрать данные и сохранить массив .npy.",
    )

    prepare.add_argument(
        "--path",
        default=None,
        help="Каталог данных. По умолчанию: PREPARE_DATA_PATH или data/input.",
    )
    prepare.add_argument(
        "--ext",
        default=None,
        help="Расширения через запятую, например csv или png,jpg.",
    )
    prepare.add_argument(
        "--output",
        default=None,
        help="Путь к выходному .npy-файлу.",
    )
    prepare.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        default=None,
        help="Уровень логирования.",
    )

    commands.add_parser(
        "doctor",
        help="Проверить Python, библиотеки и доступность NVIDIA GPU.",
    )

    return parser


def print_library_status(
    module_name: str,
    distribution_name: str,
    *,
    required: bool,
) -> bool:
    """Проверить установку пакета без импорта тяжёлых библиотек."""
    try:
        installed = importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        installed = False

    importance = "обязательная" if required else "необязательная"

    if not installed:
        print(f"  {distribution_name}: не найдена ({importance})")
        return False

    try:
        version = metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        version = "версия неизвестна"

    print(f"  {distribution_name}: {version} ({importance})")
    return True


def doctor() -> int:
    """Проверить окружение; отсутствие CUDA не считается ошибкой."""
    print("=== Диагностика окружения ===")
    print(f"Python: {platform.python_version()}")
    print(f"Интерпретатор: {sys.executable}")
    print(f"Операционная система: {platform.platform()}")
    print(f"Рабочая директория: {Path.cwd()}")

    print("\\nБиблиотеки:")

    numpy_available = print_library_status(
        "numpy", "numpy", required=True
    )
    print_library_status("PIL", "Pillow", required=False)
    print_library_status("dotenv", "python-dotenv", required=False)
    print_library_status("torch", "torch", required=False)
    print_library_status("tensorflow", "tensorflow", required=False)

    print(
        "\\nPillow нужен для изображений. "
        "Для подготовки CSV достаточно NumPy. "
        "TensorFlow и PyTorch для этого задания не обязательны."
    )

    print("\\nПроверка NVIDIA GPU через nvidia-smi:")

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except FileNotFoundError:
        print(
            "  nvidia-smi не найдена. "
            "Для подготовки CSV и изображений GPU не требуется."
        )
    except subprocess.TimeoutExpired:
        print("  Превышено время ожидания nvidia-smi.")
    except OSError as exc:
        print(f"  Не удалось запустить nvidia-smi: {exc}")
    else:
        if result.returncode == 0:
            print(result.stdout.strip() or "  Команда выполнена успешно.")
            print(
                "Драйвер NVIDIA доступен. Это не гарантирует, что "
                "установленный PyTorch или TensorFlow поддерживает CUDA."
            )
        else:
            details = result.stderr.strip() or result.stdout.strip()
            print(
                f"  nvidia-smi завершилась с кодом {result.returncode}."
            )
            if details:
                print(details)

    return 0 if numpy_available else 1


def run_prepare(settings: Settings) -> int:
    """Проверить настройки и вызвать обработчик из scanner.py."""
    if not settings.data_path.is_dir():
        raise ConfigurationError(
            f"Каталог данных не найден: {settings.data_path}"
        )

    if settings.output_file.exists() and not settings.output_file.is_file():
        raise ConfigurationError(
            f"Выходной путь не является файлом: {settings.output_file}"
        )

    # Отложенный импорт: doctor работает и без scanner.py/NumPy.
    try:
        from part2_prepare.app.scanner import prepare_dataset
    except ImportError as exc:
        raise ConfigurationError(
            "Не удалось импортировать scanner.prepare_dataset. "
            "Нужен scanner.py с функцией prepare_dataset "
            "и установленными зависимостями."
        ) from exc

    settings.output_file.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Входной каталог: %s", settings.data_path)
    LOGGER.info("Расширения: %s", ", ".join(settings.extensions))
    LOGGER.info("Выходной файл: %s", settings.output_file)

    started = perf_counter()

    stats = prepare_dataset(
        input_dir=settings.data_path,
        extensions=settings.extensions,
        output_file=settings.output_file,
    )

    elapsed = perf_counter() - started

    if not settings.output_file.is_file():
        raise RuntimeError(
            "Обработчик завершился, но выходной .npy-файл не создан."
        )

    print("\\nПодготовка завершена.")
    print(f"Обработано файлов: {stats['file_count']}")
    print(f"Форма массива: {stats['shape']}")
    print(f"Тип данных: {stats['dtype']}")
    print(f"Общий размер входных файлов: {stats['input_bytes']} байт")
    print(f"Размер .npy: {settings.output_file.stat().st_size} байт")
    print(f"Время обработки: {elapsed:.3f} с")
    print(f"Результат: {settings.output_file}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "doctor":
            return doctor()

        settings = Settings.from_sources(
            path=args.path,
            output=args.output,
            extensions=args.ext,
            log_level=args.log_level,
        )

        configure_logging(settings.log_level)
        return run_prepare(settings)

    except KeyboardInterrupt:
        print("\\nОперация прервана пользователем.", file=sys.stderr)
        return 130
    except (
        ConfigurationError,
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        KeyError,
        TypeError,
    ) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())