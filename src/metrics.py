"""Image-similarity metrics for scoring DLG reconstructions.

All functions accept 2-D arrays or single-channel tensors in the ``[0, 1]``
range and return a scalar. Higher PSNR/SSIM and lower MSE mean a better (more
faithful) reconstruction of the victim's image.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_2d(image) -> np.ndarray:
    """Coerce a tensor/array of shape (..., H, W) into a 2-D float64 array."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    return np.asarray(image, dtype=np.float64).squeeze()


def compute_mse(original, recovered) -> float:
    a, b = _to_2d(original), _to_2d(recovered)
    return float(np.mean((a - b) ** 2))


def compute_psnr(original, recovered) -> float:
    a, b = _to_2d(original), _to_2d(recovered)
    # An exact match has zero MSE; skimage would warn on the 1/0 and return inf.
    # Short-circuit so a perfect reconstruction reads as a clean +inf dB.
    if np.array_equal(a, b):
        return float("inf")
    return float(peak_signal_noise_ratio(a, b, data_range=1.0))


def compute_ssim(original, recovered) -> float:
    a, b = _to_2d(original), _to_2d(recovered)
    return float(structural_similarity(a, b, data_range=1.0))
