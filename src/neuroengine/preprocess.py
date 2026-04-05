import numpy as np


def compute_mean_image(stack: np.ndarray) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError("Expected stack with shape (time, height, width).")
    return np.mean(stack, axis=0)


def compute_baseline(stack: np.ndarray, percentile: float = 20.0) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError("Expected stack with shape (time, height, width).")
    return np.percentile(stack, percentile, axis=0)


def compute_dff(stack: np.ndarray, baseline: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError("Expected stack with shape (time, height, width).")
    if baseline.ndim != 2:
        raise ValueError("Expected baseline with shape (height, width).")
    if stack.shape[1:] != baseline.shape:
        raise ValueError("Spatial dimensions of stack and baseline do not match.")

    stack = stack.astype(np.float32)
    baseline = baseline.astype(np.float32)

    return (stack - baseline[None, :, :]) / (baseline[None, :, :] + eps)


def extract_pixel_trace(stack: np.ndarray, y: int, x: int) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError("Expected stack with shape (time, height, width).")
    return stack[:, y, x]
