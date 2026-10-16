"""Настройка логирования для модуля подготовки данных."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_LEVELS = {
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
}


def configure_logging(
    level: str | None = None,
    log_file: str | Path | None = None,
) -> None:
    """Настроить обработчики корневого логгера.

    Аргументы функции имеют приоритет над окружением.

    Переменные:
        PREPARE_LOG_LEVEL — уровень логирования;
        PREPARE_LOG_FILE  — путь к файлу журнала.

    Если путь к журналу не задан, используется только консоль.
    """
    selected_level = (
        level
        if level is not None
        else os.getenv("PREPARE_LOG_LEVEL", "INFO")
    ).strip().upper()

    if selected_level not in LOG_LEVELS:
        raise ValueError(
            "Некорректный уровень логирования: "
            f"{selected_level}"
        )

    selected_file = (
        log_file
        if log_file is not None
        else os.getenv("PREPARE_LOG_FILE")
    )

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    try:
        if selected_file:
            expanded_path = os.path.expandvars(str(selected_file))
            path = Path(expanded_path).expanduser().resolve()

            path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                filename=path,
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        logging.basicConfig(
            level=getattr(logging, selected_level),
            handlers=handlers,
            force=True,
        )

    except Exception:
        # Закрываем уже созданные обработчики при ошибке настройки.
        for handler in handlers:
            handler.close()

        raise