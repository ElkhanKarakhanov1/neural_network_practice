from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from part3_parallel.app.performance import benchmark, prepare_sequential
    from part3_parallel.app.producer_consumer import PipelineConfig, prepare_parallel
else:
    from .performance import benchmark, prepare_sequential
    from .producer_consumer import PipelineConfig, prepare_parallel

from part2_prepare.app.config import load_environment, parse_extensions
from part2_prepare.app.logging_config import configure_logging

ROOT = Path(__file__).resolve().parents[2]


def ask(prompt, default):
    return input(f"{prompt} [{default}]: ").strip() or str(default)


def ask_path(prompt, default):
    value = ask(prompt, default)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {chr(34), chr(39)}:
        value = value[1:-1]
    if not value.strip():
        raise ValueError("Путь не может быть пустым")
    return Path(os.path.expandvars(value)).expanduser().resolve()


def settings():
    load_environment()
    source = ask_path("Каталог данных", os.getenv("PARALLEL_DATA_PATH", str(ROOT / "data" / "input")))
    output = ask_path("Выходной каталог", os.getenv("PARALLEL_OUTPUT_DIR", str(ROOT / "data" / "output")))
    extensions = parse_extensions(ask("Расширения через запятую", os.getenv("PARALLEL_EXTENSIONS", "csv")))
    if ".csv" in extensions and len(extensions) > 1:
        raise ValueError("CSV и изображения обрабатываются отдельно")
    config = PipelineConfig(
        readers=int(ask("Число потоков чтения", os.getenv("PARALLEL_READERS", "4"))),
        processes=int(ask("Число процессов", os.getenv("PARALLEL_PROCESSES", "2"))),
        queue_size=int(ask("Ёмкость очередей", os.getenv("PARALLEL_QUEUE_SIZE", "8"))),
        max_pending=int(ask("Максимум задач в пуле одновременно", os.getenv("PARALLEL_MAX_PENDING", "4"))),
        copies=int(ask("Копий на изображение (для CSV не применяется)", os.getenv("PARALLEL_COPIES", "2"))),
        image_size=(int(ask("Ширина изображения", os.getenv("PARALLEL_IMAGE_WIDTH", "64"))),
                    int(ask("Высота изображения", os.getenv("PARALLEL_IMAGE_HEIGHT", "64")))),
        noise_std=float(ask("Стандартное отклонение шума", os.getenv("PARALLEL_NOISE_STD", "0.02"))),
        seed=int(ask("Seed", os.getenv("PARALLEL_SEED", "42"))),
    )
    configure_logging(os.getenv("PARALLEL_LOG_LEVEL", "INFO"))
    return source, output, extensions, config


def confirm(paths):
    existing = [p for p in paths if p.exists()]
    if not existing:
        return True
    print("Следующие файлы будут перезаписаны:")
    for path in existing:
        print(f"  {path}")
        if not path.is_file():
            raise ValueError(f"Путь не является файлом: {path}")
    return input("Продолжить? (да/нет) [нет]: ").strip().lower() in {"да", "д", "yes", "y"}


def main():
    while True:
        try:
            print("\n=== Задание 3: параллельная подготовка ===\n"
                  "1. Последовательная обработка\n"
                  "2. Потоки чтения + пул процессов\n"
                  "3. Сравнить версии и сохранить performance_report.json\n"
                  "0. Выход")
            choice = input("Выберите действие: ").strip()
            if choice == "0":
                return 0
            if choice not in {"1", "2", "3"}:
                print("Введите 0, 1, 2 или 3.")
                continue
            source, output, extensions, config = settings()
            if choice == "3":
                repeats = int(ask("Число повторов сравнения", os.getenv("PARALLEL_REPEATS", "3")))
                if repeats < 1:
                    raise ValueError("Число повторов должно быть положительным")
                targets = [output / name for name in ("sequential.npy", "parallel.npy", "performance_report.json")]
                if not confirm(targets):
                    print("Операция отменена.")
                    continue
                report = benchmark(source, output, extensions, config, repeats)
                print(f"Последовательно (медиана): {report['sequential_median_seconds']:.6f} с")
                print(f"Параллельно (медиана): {report['parallel_median_seconds']:.6f} с")
                speedup = report["speedup"]
                if speedup is not None:
                    print(f"Отношение T_seq / T_parallel: {speedup:.3f}")
                    if speedup < 1:
                        print("Параллельная версия медленнее на этом наборе.")
                print(f"Массивы совпадают: {report['outputs_equal']}")
                print(f"Отчёт: {output / 'performance_report.json'}")
            else:
                name = "sequential.npy" if choice == "1" else "parallel.npy"
                destination = output / name
                if not confirm([destination]):
                    print("Операция отменена.")
                    continue
                runner = prepare_sequential if choice == "1" else prepare_parallel
                stats = runner(source, destination, extensions, config)
                print(f"Обработано файлов: {stats['file_count']}")
                print(f"Форма массива: {stats['shape']}; тип: {stats['dtype']}")
                print(f"Время: {stats['elapsed_seconds']:.6f} с")
                print(f"Результат: {stats['output_file']}")
        except (KeyboardInterrupt, EOFError):
            print("\nЗавершение работы.")
            return 0
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            print(f"Ошибка: {exc}")


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())