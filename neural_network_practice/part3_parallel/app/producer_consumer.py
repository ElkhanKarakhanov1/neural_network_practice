from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from part2_prepare.app.scanner import read_csv_file, save_array
from .augment import augment_array


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
STOP = object()


class PipelineError(RuntimeError):
    """Failure of the file loading or augmentation pipeline."""


@dataclass(frozen=True)
class PipelineConfig:
    readers: int = 4
    processes: int = 2
    queue_size: int = 8
    max_pending: int = 4
    copies: int = 2
    image_size: tuple[int, int] = (64, 64)  # width, height
    noise_std: float = 0.02
    seed: int = 42

    def __post_init__(self):
        for name in ("readers", "processes", "queue_size", "max_pending"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.copies, int) or isinstance(self.copies, bool) or self.copies < 0:
            raise ValueError("copies must be a nonnegative integer")
        if len(self.image_size) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in self.image_size):
            raise ValueError("image_size must contain two positive integers")
        if not np.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def _read_file(path: Path, image_size):
    if path.suffix.lower() == ".csv":
        header, array = read_csv_file(path)
        with np.errstate(over="ignore"):
            array = array.astype(np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: values exceed float32 range")
        return "csv", header, array
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = image.resize(image_size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return "image", None, array


def prepare_parallel(
    input_dir: str | Path,
    output_file: str | Path,
    extensions=("csv",),
    config: PipelineConfig | None = None,
) -> dict:
    """Read files with threads, augment with apply_async, save a single NPY.

    Call this function only below an `if __name__ == "__main__"` guard
    when launching a script. Errors abort the result; files are not skipped.
    """
    config = config or PipelineConfig()
    input_dir = Path(input_dir).expanduser().resolve()
    output_file = Path(output_file).expanduser().resolve()
    if not input_dir.is_dir():
        raise PipelineError(f"Input directory does not exist: {input_dir}")
    if output_file.suffix.lower() != ".npy":
        raise PipelineError("Output must have the .npy extension")
    if isinstance(extensions, str):
        extensions = extensions.split(",")
    selected = {"." + ext.strip().lower().lstrip(".") for ext in extensions if ext.strip()}
    if not selected or not selected <= (IMAGE_EXTENSIONS | {".csv"}):
        raise PipelineError("Unsupported or empty extensions")
    if ".csv" in selected and selected & IMAGE_EXTENSIONS:
        raise PipelineError("Prepare CSV and images in separate runs")

    paths = queue.Queue(maxsize=config.queue_size)
    loaded = queue.Queue(maxsize=config.queue_size)
    completed = queue.Queue()
    stop = threading.Event()
    lock = threading.Lock()
    counters = {"found": 0, "read": 0, "processed": 0, "input_bytes": 0}
    errors = []
    results = {}
    header = None
    started = perf_counter()

    def put_interruptibly(target, item):
        while not stop.is_set():
            try:
                target.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def record_error(message):
        with lock:
            errors.append(str(message))

    def producer():
        try:
            def walk_error(exc):
                raise exc
            index = 0
            for directory, subdirs, filenames in os.walk(input_dir, followlinks=False, onerror=walk_error):
                subdirs.sort()
                if stop.is_set():
                    break
                for filename in sorted(filenames):
                    if stop.is_set():
                        break
                    path = Path(directory) / filename
                    if path.suffix.lower() not in selected:
                        continue
                    size = path.stat().st_size
                    if not put_interruptibly(paths, (index, path)):
                        break
                    with lock:
                        counters["found"] += 1
                        counters["input_bytes"] += size
                    index += 1
        except Exception as exc:
            record_error(f"Directory scan failed: {exc}")
        finally:
            for _ in range(config.readers):
                put_interruptibly(paths, STOP)

    def consumer():
        try:
            while not stop.is_set():
                try:
                    item = paths.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is STOP:
                    break
                index, path = item
                try:
                    kind, names, array = _read_file(path, config.image_size)
                    with lock:
                        counters["read"] += 1
                    if not put_interruptibly(loaded, (index, kind, names, array)):
                        break
                except Exception as exc:
                    record_error(f"Cannot read {path}: {exc}")
        finally:
            put_interruptibly(loaded, STOP)

    # Pool before starting threads; spawn is also safe on Windows.
    pool = mp.get_context("spawn").Pool(processes=config.processes)
    threads = []
    pending = 0
    finished_readers = 0
    kind = "csv" if selected == {".csv"} else "image"
    try:
        for number in range(config.readers):
            thread = threading.Thread(target=consumer, name=f"reader-{number}")
            thread.start()
            threads.append(thread)
        thread = threading.Thread(target=producer, name="producer")
        thread.start()
        threads.append(thread)

        while finished_readers < config.readers or pending:
            if pending and (pending >= config.max_pending or finished_readers == config.readers):
                index, success, value = completed.get()
                pending -= 1
                if success:
                    results[index] = value
                    with lock:
                        counters["processed"] += 1
                else:
                    record_error(f"Augmentation task {index}: {value}")
                continue
            try:
                item = loaded.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is STOP:
                finished_readers += 1
                continue
            index, item_kind, names, array = item
            if item_kind == "csv":
                if header is None:
                    header = names
                elif header != names:
                    record_error(f"CSV header mismatch for file index {index}")
                    continue
            pool.apply_async(
                augment_array,
                args=(array, item_kind, config.copies, config.seed + index, config.noise_std),
                callback=lambda value, i=index: completed.put((i, True, value)),
                error_callback=lambda error, i=index: completed.put((i, False, str(error))),
            )
            pending += 1

        for thread in threads:
            thread.join()
        pool.close()
        pool.join()
    except BaseException:
        stop.set()
        pool.terminate()
        for thread in threads:
            thread.join()
        pool.join()
        raise

    if errors:
        raise PipelineError("\n".join(errors))
    if not results:
        raise PipelineError("No matching files were processed")
    array = np.concatenate([results[i] for i in sorted(results)], axis=0)
    if kind == "csv":
        # Global CSV feature scaling; target column remains unchanged.
        features = array[:, :-1].astype(np.float64)
        low = features.min(axis=0)
        span = features.max(axis=0) - low
        array[:, :-1] = ((features - low) / np.where(span == 0, 1, span)).astype(np.float32)
    save_array(array, output_file)
    return {
        "file_count": counters["processed"],
        "read_count": counters["read"],
        "found_count": counters["found"],
        "input_bytes": counters["input_bytes"],
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
        "elapsed_seconds": perf_counter() - started,
        "output_file": str(output_file),
    }