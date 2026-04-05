import numpy as np


def basic_stats(stack: np.ndarray) -> dict:
    """
    Compute basic statistics for an image stack.
    """
    return {
        "shape": stack.shape,
        "dtype": str(stack.dtype),
        "min": float(np.min(stack)),
        "max": float(np.max(stack)),
        "mean": float(np.mean(stack)),
        "std": float(np.std(stack)),
    }
