from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.neuroengine.io import load_tiff
from src.neuroengine.preprocess import (
    compute_mean_image,
    compute_baseline,
    compute_dff,
    extract_pixel_trace,
)

stack = load_tiff("data/raw/camera.tif")
print("original shape:", stack.shape)
print("dtype:", stack.dtype)

if stack.ndim == 2:
    stack = stack[None, :, :]
    print("expanded shape:", stack.shape)

mean_image = compute_mean_image(stack)
baseline = compute_baseline(stack)
dff = compute_dff(stack, baseline)

y = stack.shape[1] // 2
x = stack.shape[2] // 2

raw_trace = extract_pixel_trace(stack, y, x)
dff_trace = extract_pixel_trace(dff, y, x)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

np.save(results_dir / "mean_image.npy", mean_image)
np.save(results_dir / "baseline.npy", baseline)
np.save(results_dir / "dff.npy", dff)
np.save(results_dir / "raw_trace.npy", raw_trace)
np.save(results_dir / "dff_trace.npy", dff_trace)

plt.figure(figsize=(6, 6))
plt.imshow(mean_image, cmap="gray")
plt.title("Mean image")
plt.colorbar()
plt.tight_layout()
plt.savefig(results_dir / "mean_image.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 6))
plt.imshow(dff[0], cmap="gray")
plt.title("First frame dF/F")
plt.colorbar()
plt.tight_layout()
plt.savefig(results_dir / "dff_first_frame.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(raw_trace)
plt.title(f"Raw trace at pixel ({y}, {x})")
plt.xlabel("Time")
plt.ylabel("Intensity")
plt.tight_layout()
plt.savefig(results_dir / "raw_trace.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(dff_trace)
plt.title(f"dF/F trace at pixel ({y}, {x})")
plt.xlabel("Time")
plt.ylabel("dF/F")
plt.tight_layout()
plt.savefig(results_dir / "dff_trace.png", dpi=150)
plt.close()

print("Saved results to:", results_dir.resolve())
print("done")