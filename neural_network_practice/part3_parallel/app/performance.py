from __future__ import annotations

import json
import os
import platform
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from part2_prepare.app.scanner import find_files, save_array
from .augment import augment_array
from .producer_consumer import PipelineConfig, PipelineError, _read_file, prepare_parallel


def prepare_sequential(input_dir, output_file, extensions=("csv",), config=None):
    """Same transformations and file ordering as prepare_parallel, no workers."""
    config = config or PipelineConfig()
    started = perf_counter()
    source = Path(input_dir).expanduser().resolve()
    destination = Path(output_file).expanduser().resolve()
    if destination.suffix.lower() != ".npy":
        raise PipelineError("Output must have the .npy extension")
    if isinstance(extensions, str):
        extensions = extensions.split(",")
    files = find_files(source, extensions)
    kinds = {"csv" if p.suffix.lower() == ".csv" else "image" for p in files}
    if len(kinds) != 1:
        raise PipelineError("Prepare CSV and images separately")
    input_bytes = sum(p.stat().st_size for p in files)
    header = None
    results = []
    for index, path in enumerate(files):
        kind, names, array = _read_file(path, config.image_size)
        if kind == "csv":
            if header is None:
                header = names
            elif header != names:
                raise PipelineError(f"CSV header mismatch: {path}")
        results.append(augment_array(array, kind, config.copies, config.seed + index, config.noise_std))
    array = np.concatenate(results, axis=0)
    if kind == "csv":
        features = array[:, :-1].astype(np.float64)
        low = features.min(axis=0)
        span = features.max(axis=0) - low
        array[:, :-1] = ((features - low) / np.where(span == 0, 1, span)).astype(np.float32)
    save_array(array, destination)
    return {
        "file_count": len(files), "read_count": len(files), "found_count": len(files),
        "input_bytes": input_bytes, "shape": tuple(array.shape), "dtype": str(array.dtype),
        "elapsed_seconds": perf_counter() - started, "output_file": str(destination),
    }


def _same_arrays(left_path, right_path, batch_size=32):
    left = np.load(left_path, mmap_mode="r", allow_pickle=False)
    right = np.load(right_path, mmap_mode="r", allow_pickle=False)
    try:
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        return all(np.array_equal(left[i:i + batch_size], right[i:i + batch_size])
                   for i in range(0, len(left), batch_size))
    finally:
        del left, right


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def benchmark(input_dir, output_dir, extensions=("csv",), config=None, repeats=3):
    """Compare full scan/read/transform/save cycles; alternate run order.

    Input files must not change during the benchmark. Output files are
    overwritten. Call only from a guarded entry point when using spawn.
    """
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    config = config or PipelineConfig()
    if isinstance(extensions, str):
        extensions = extensions.split(",")
    extensions = tuple(extensions)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"sequential": output_dir / "sequential.npy", "parallel": output_dir / "parallel.npy"}
    runners = {"sequential": prepare_sequential, "parallel": prepare_parallel}
    runs = {"sequential": [], "parallel": []}
    matches = []
    for repeat in range(repeats):
        order = ("sequential", "parallel") if repeat % 2 == 0 else ("parallel", "sequential")
        for name in order:
            stats = runners[name](input_dir, paths[name], extensions, config)
            runs[name].append(stats)
        matches.append(_same_arrays(paths["sequential"], paths["parallel"]))
    sequential_time = median(x["elapsed_seconds"] for x in runs["sequential"])
    parallel_time = median(x["elapsed_seconds"] for x in runs["parallel"])
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(Path(input_dir).expanduser().resolve()),
        "extensions": extensions, "config": asdict(config), "repeats": repeats,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform(), "cpu_count": os.cpu_count()},
        "measurement": "End-to-end: scan, read, augment, normalize and save; includes pool startup and shutdown; excludes result comparison",
        "runs": runs, "sequential_median_seconds": sequential_time,
        "parallel_median_seconds": parallel_time,
        "speedup": sequential_time / parallel_time if parallel_time > 0 else None,
        "outputs_equal": all(matches), "comparison_by_repeat": matches,
        "notes": ["Input must remain unchanged during all runs.",
                  "OS file caching affects timing; order alternates but caches are not cleared.",
                  "CSV is not augmented; image copies and seeds are identical in both modes.",
                  "Final arrays are accumulated in RAM in both implementations."],
    }
    report_path = output_dir / "performance_report.json"
    _write_json(report_path, report)
    if not report["outputs_equal"]:
        raise PipelineError(f"Sequential and parallel outputs differ. See {report_path}")
    return report