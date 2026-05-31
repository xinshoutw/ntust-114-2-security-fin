"""Differential-privacy noise for the gradient/update defence (Step 3).

A lighter-weight alternative to homomorphic encryption: instead of hiding the
gradient from the server, the *client* perturbs it with calibrated Gaussian
noise before upload (the Gaussian mechanism behind DP-SGD / DP-FedAvg). The
server still sees a plaintext gradient, but a noisy one -- enough to blunt the
DLG reconstruction, at the cost of some model accuracy. Sweeping the noise level
traces a privacy-utility trade-off.

We parameterise the noise *relative to the signal*: the standard deviation is
``sigma * rms(update)`` where ``rms`` is the root-mean-square of the update's
elements. This makes a single ``sigma`` knob meaningful across two very
differently-scaled objects -- the weight delta the server aggregates and the
single-sample loss gradient the DLG attacker inverts -- so the accuracy curve
and the leakage curve share one comparable x-axis. ``sigma`` reads as "noise std
as a fraction of the update's RMS magnitude"; ``sigma=0`` is a no-op.
"""

from __future__ import annotations

import torch


def _rms(flat: torch.Tensor) -> float:
    """Root-mean-square magnitude of a flattened tensor."""
    return float(flat.pow(2).mean().sqrt())


def add_relative_gaussian_noise(
    update: dict[str, torch.Tensor],
    sigma: float,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Return a copy of ``update`` with std=``sigma * rms(update)`` Gaussian noise.

    The RMS is taken over *all* elements jointly so every tensor in the update is
    perturbed on the same absolute scale. ``sigma <= 0`` returns the update
    unchanged (a clone).
    """
    if sigma <= 0:
        return {k: v.clone() for k, v in update.items()}
    flat = torch.cat([v.flatten().cpu() for v in update.values()])
    std = sigma * _rms(flat)
    # Draw noise on CPU (Generator is CPU) and move to the update's device (e.g. MPS).
    return {
        k: v + torch.randn(v.shape, generator=generator, dtype=v.dtype).to(v.device) * std
        for k, v in update.items()
    }


def add_relative_gaussian_noise_list(
    grads: list[torch.Tensor],
    sigma: float,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """List variant of :func:`add_relative_gaussian_noise` for raw gradients.

    Used to perturb the single-sample loss gradient the DLG attacker observes,
    so the leakage curve is measured under the same relative-noise model as the
    accuracy curve.
    """
    if sigma <= 0:
        return [g.clone() for g in grads]
    flat = torch.cat([g.flatten().cpu() for g in grads])
    std = sigma * _rms(flat)
    return [
        g + torch.randn(g.shape, generator=generator, dtype=g.dtype).to(g.device) * std
        for g in grads
    ]
