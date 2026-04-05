from pathlib import Path
import tifffile
import numpy as np


def load_tiff(path: str | Path) -> np.ndarray:
    """
    Load a TIFF stack from disk.

    Parameters
    ----------
    path : str | Path
        Path to the TIFF file.

    Returns
    -------
    np.ndarray
        Loaded image stack.
    """
    return tifffile.imread(path)
