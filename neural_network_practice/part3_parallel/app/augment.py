from __future__ import annotations

import numpy as np


def augment_array(
    array: np.ndarray,
    kind: str,
    copies: int = 2,
    seed: int = 42,
    noise_std: float = 0.02,
) -> np.ndarray:
    """Return original plus augmented images, or an unchanged CSV table.

    Images: input (H, W, 3), output (1 + copies, H, W, 3).
    CSV: input/output (rows, columns); the target column is never modified.
    """
    if not isinstance(copies, int) or isinstance(copies, bool) or copies < 0:
        raise ValueError("copies must be a nonnegative integer")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    data = np.asarray(array, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("Input contains NaN or infinity")
    if kind == "csv":
        if data.ndim != 2 or min(data.shape) == 0:
            raise ValueError("Expected a nonempty CSV matrix")
        return data.copy()
    if kind != "image":
        raise ValueError(f"Unsupported data kind: {kind}")
    if data.ndim != 3 or data.shape[2] != 3 or min(data.shape[:2]) == 0:
        raise ValueError("Expected an RGB image with shape (H, W, 3)")
    if data.min() < 0 or data.max() > 1:
        raise ValueError("Image values must be normalized to [0, 1]")

    rng = np.random.default_rng(seed)
    result = [data.copy()]
    for _ in range(copies):
        # For rectangular images only 0/180-degree rotations preserve shape.
        turns = int(rng.integers(0, 4)) if data.shape[0] == data.shape[1] else int(rng.integers(0, 2)) * 2
        transformed = np.rot90(data, k=turns).copy()
        if rng.random() < 0.5:
            transformed = np.flip(transformed, axis=1)
        if rng.random() < 0.5:
            transformed = np.flip(transformed, axis=0)
        noise = rng.normal(0.0, noise_std, size=transformed.shape).astype(np.float32)
        result.append(np.clip(transformed + noise, 0.0, 1.0))
    return np.stack(result).astype(np.float32, copy=False)