"""Настройки подготовки данных.

Переменные окружения:
    PREPARE_DATA_PATH       — входной каталог;
    PREPARE_OUTPUT_DIR      — выходной каталог;
    PREPARE_OUTPUT_FILE     — имя выходного файла;
    PREPARE_EXTENSIONS      — расширения через запятую;
    PREPARE_LOG_LEVEL       — уровень логирования.

Значения CLI имеют приоритет над переменными окружения.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path




PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_EXTENSIONS = frozenset(
    {".csv", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)

LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigurationError(ValueError):
    """Некорректные настройки приложения."""


def load_environment() -> None:
    """Загрузить .env из корня проекта, если установлен python-dotenv.

    Уже заданные переменные окружения не перезаписываются.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(PROJECT_ROOT / ".env", override=False)


def parse_extensions(value: str) -> tuple[str, ...]:
    """Преобразовать 'csv,png,.jpg' в ('.csv', '.png', '.jpg')."""
    extensions = []

    for item in value.split(","):
        item = item.strip().lower()

        if not item:
            continue

        extension = item if item.startswith(".") else f".{item}"

        if extension not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ConfigurationError(
                f"Расширение {extension!r} не поддерживается. "
                f"Допустимые расширения: {supported}"
            )

        if extension not in extensions:
            extensions.append(extension)

    if not extensions:
        raise ConfigurationError("Нужно указать хотя бы одно расширение.")

    return tuple(extensions)


def resolve_path(value: str | Path) -> Path:
    """Раскрыть ~ и переменные окружения, получить абсолютный путь.

    Относительные пользовательские пути считаются от рабочей директории.
    """
    expanded = os.path.expandvars(str(value))
    return Path(expanded).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    data_path: Path
    output_file: Path
    extensions: tuple[str, ...]
    log_level: str

    @classmethod
    def from_sources(
        cls,
        *,
        path: str | None = None,
        output: str | None = None,
        extensions: str | None = None,
        log_level: str | None = None,
    ) -> "Settings":
        """Собрать настройки из CLI, окружения и значений по умолчанию."""
        load_environment()

        data_path = resolve_path(
            path
            if path is not None
            else os.getenv(
                "PREPARE_DATA_PATH",
                str(PROJECT_ROOT / "data" / "input"),
            )
        )

        output_dir = resolve_path(
            os.getenv(
                "PREPARE_OUTPUT_DIR",
                str(PROJECT_ROOT / "data" / "output"),
            )
        )

        if output is not None:
            # --output — полный путь или путь от рабочей директории.
            output_file = resolve_path(output)
        else:
            filename = Path(
                os.getenv("PREPARE_OUTPUT_FILE", "prepared.npy")
            )
            output_file = resolve_path(output_dir / filename)

        if output_file.suffix.lower() != ".npy":
            raise ConfigurationError(
                "Выходной файл должен иметь расширение .npy."
            )

        selected_extensions = parse_extensions(
            extensions
            if extensions is not None
            else os.getenv("PREPARE_EXTENSIONS", "csv")
        )

        selected_level = (
            log_level
            if log_level is not None
            else os.getenv("PREPARE_LOG_LEVEL", "INFO")
        ).strip().upper()

        if selected_level not in LOG_LEVELS:
            raise ConfigurationError(
                "Уровень логирования должен быть одним из: "
                + ", ".join(sorted(LOG_LEVELS))
            )

        return cls(
            data_path=data_path,
            output_file=output_file,
            extensions=selected_extensions,
            log_level=selected_level,
        )