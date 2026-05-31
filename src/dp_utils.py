"""Differential-privacy defence for federated learning (Step 3).

This is the **Gaussian mechanism of DP-FedAvg** (McMahan et al., 2018), the
federated form of DP-SGD (Abadi et al., 2016). Instead of hiding the gradient
from the server (homomorphic encryption, Step 3-1), each client *perturbs* the
update it uploads so that the server -- which still sees a plaintext update --
sees one that provably leaks little about any single training example:

  1. **Clip** the client update to a fixed L2 norm bound ``C``. This bounds the
     *sensitivity* of the upload to one record, which is what makes a finite
     privacy budget possible -- without it there is no valid ``epsilon``.
  2. **Add Gaussian noise** with standard deviation ``z * C`` per coordinate,
     where ``z`` is the *noise multiplier*. The released vector is then
     (eps, delta)-DP, and the budget ``epsilon`` depends only on ``z`` and the
     number of rounds (not on ``C``, since both signal-clip and noise scale
     with ``C``).

Sweeping ``z`` traces the privacy-utility trade-off the assignment asks for:
small ``z`` -> large ``epsilon`` (weak privacy) but little accuracy loss; large
``z`` -> small ``epsilon`` (strong privacy) but the noise swamps the update.

``epsilon`` is accounted with Renyi Differential Privacy (Mironov, 2017): the
Gaussian mechanism is ``(alpha, alpha / (2 z**2))``-RDP, RDP composes additively
over the ``T`` rounds, and the result converts to ``(eps, delta)``-DP.
"""

from __future__ import annotations

import math

import torch

# RDP orders to search when converting to (eps, delta)-DP. A spread of fractional
# and integer orders, as used by standard accountants (e.g. Opacus).
DEFAULT_RDP_ORDERS: tuple[float, ...] = (
    1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0,
    10.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0,
)


# --- L2 clipping: bound the sensitivity ---------------------------------------

def flat_l2_norm(update: dict[str, torch.Tensor]) -> float:
    """L2 norm of an update taken jointly over all its tensors."""
    flat = torch.cat([v.flatten().cpu() for v in update.values()])
    return float(flat.norm())


def clip_update(
    update: dict[str, torch.Tensor], clip_norm: float
) -> tuple[dict[str, torch.Tensor], float]:
    """Scale ``update`` so its joint L2 norm is at most ``clip_norm``.

    Returns ``(clipped_update, original_norm)``. A zero update is returned
    unchanged (no division by zero).
    """
    norm = flat_l2_norm(update)
    scale = 1.0 if norm == 0.0 else min(1.0, clip_norm / norm)
    return {k: v * scale for k, v in update.items()}, norm


def clip_grad_list(
    grads: list[torch.Tensor], clip_norm: float
) -> tuple[list[torch.Tensor], float]:
    """List variant of :func:`clip_update` for raw gradient tensors."""
    flat = torch.cat([g.flatten().cpu() for g in grads])
    norm = float(flat.norm())
    scale = 1.0 if norm == 0.0 else min(1.0, clip_norm / norm)
    return [g * scale for g in grads], norm


# --- Gaussian mechanism -------------------------------------------------------

def dp_fedavg_update(
    update: dict[str, torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Clip ``update`` to ``clip_norm`` then add ``N(0, (z*clip_norm)^2)`` noise.

    ``noise_multiplier`` (``z``) <= 0 skips the noise (clipping only). Noise is
    drawn on CPU (the Generator is CPU-bound) and moved to each tensor's device.
    """
    clipped, _ = clip_update(update, clip_norm)
    if noise_multiplier <= 0:
        return clipped
    std = noise_multiplier * clip_norm
    return {
        k: v + torch.randn(v.shape, generator=generator, dtype=v.dtype).to(v.device) * std
        for k, v in clipped.items()
    }


def dp_fedavg_grad_list(
    grads: list[torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """List variant of :func:`dp_fedavg_update` for the single-gradient attack demo."""
    clipped, _ = clip_grad_list(grads, clip_norm)
    if noise_multiplier <= 0:
        return clipped
    std = noise_multiplier * clip_norm
    return [
        g + torch.randn(g.shape, generator=generator, dtype=g.dtype).to(g.device) * std
        for g in clipped
    ]


# --- RDP privacy accountant ---------------------------------------------------

def gaussian_rdp(
    noise_multiplier: float,
    steps: int,
    orders: tuple[float, ...] = DEFAULT_RDP_ORDERS,
) -> dict[float, float]:
    """RDP of ``steps`` compositions of the Gaussian mechanism at each order.

    A single Gaussian mechanism with noise multiplier ``z`` is
    ``(alpha, alpha / (2 z**2))``-RDP; ``steps`` of them compose additively to
    ``steps * alpha / (2 z**2)``. ``z <= 0`` yields infinite RDP (no privacy).
    """
    if noise_multiplier <= 0:
        return {alpha: float("inf") for alpha in orders}
    return {alpha: steps * alpha / (2.0 * noise_multiplier**2) for alpha in orders}


def rdp_to_epsilon(rdp_by_order: dict[float, float], delta: float) -> tuple[float, float]:
    """Convert RDP to ``(eps, delta)``-DP, returning the tightest ``(eps, order)``.

    Mironov (2017), Prop. 3: an ``(alpha, rho)``-RDP mechanism is
    ``(rho + ln(1/delta) / (alpha - 1), delta)``-DP. We minimise over orders.
    """
    best_eps, best_order = float("inf"), float("nan")
    for alpha, rho in rdp_by_order.items():
        if alpha <= 1.0:
            continue
        eps = rho + math.log(1.0 / delta) / (alpha - 1.0)
        if eps < best_eps:
            best_eps, best_order = eps, alpha
    return best_eps, best_order


def compute_epsilon(
    noise_multiplier: float,
    steps: int,
    delta: float = 1e-5,
    orders: tuple[float, ...] = DEFAULT_RDP_ORDERS,
) -> float:
    """End-to-end ``epsilon`` for ``steps`` rounds at noise multiplier ``z``."""
    if noise_multiplier <= 0:
        return float("inf")
    eps, _ = rdp_to_epsilon(gaussian_rdp(noise_multiplier, steps, orders), delta)
    return eps
