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

import numpy as np
import torch
from scipy import special
from torch import nn
from torch.func import functional_call, grad as func_grad, vmap

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


# --- Subsampled-Gaussian (DP-SGD) RDP accountant ------------------------------
#
# The plain Gaussian RDP above has no privacy amplification: with full
# participation it composes T copies of a ``z``-Gaussian and ``epsilon`` blows up
# (see ``run_dp.py``). The amplification that makes DP-SGD's ``epsilon`` small comes
# from *Poisson subsampling*: each step touches a random lot where every record is
# included independently with probability ``q``. The Renyi DP of that Sampled
# Gaussian Mechanism (SGM) is the Mironov-Talwar-Zhang (2019) bound, implemented
# below exactly as in the TF-Privacy / Opacus accountants (numerically stable in
# log space). This is what lets ``compute_epsilon_subsampled`` reach a meaningful
# ``epsilon`` (single digits) where ``compute_epsilon`` cannot.


def _log_add(logx: float, logy: float) -> float:
    """Numerically stable ``log(exp(logx) + exp(logy))``."""
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:
        return b
    return math.log1p(math.exp(a - b)) + b


def _log_sub(logx: float, logy: float) -> float:
    """Numerically stable ``log(exp(logx) - exp(logy))`` (requires logx >= logy)."""
    if logx < logy:
        raise ValueError("log of a negative number")
    if logy == -np.inf:
        return logx
    if logx == logy:
        return -np.inf
    return math.log(math.expm1(logx - logy)) + logy


def _log_erfc(x: float) -> float:
    """Stable ``log(erfc(x))`` via the standard-normal log-CDF."""
    return math.log(2.0) + special.log_ndtr(-x * math.sqrt(2.0))


def _compute_log_a_int(q: float, sigma: float, alpha: int) -> float:
    """log(A_alpha) for the SGM at integer order ``alpha``."""
    log_a = -np.inf
    for i in range(alpha + 1):
        log_coef_i = (
            math.log(special.comb(alpha, i))
            + i * math.log(q)
            + (alpha - i) * math.log(1.0 - q)
        )
        s = log_coef_i + (i * i - i) / (2.0 * sigma**2)
        log_a = _log_add(log_a, s)
    return float(log_a)


def _compute_log_a_frac(q: float, sigma: float, alpha: float) -> float:
    """log(A_alpha) for the SGM at fractional order ``alpha``."""
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0
    z0 = sigma**2 * math.log(1.0 / q - 1.0) + 0.5
    while True:
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i
        log_t0 = log_coef + i * math.log(q) + j * math.log(1.0 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1.0 - q)
        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))
        log_s0 = log_t0 + (i * i - i) / (2.0 * sigma**2) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2.0 * sigma**2) + log_e1
        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)
        i += 1
        if max(log_s0, log_s1) < -30:
            break
    return _log_add(log_a0, log_a1)


def _compute_log_a(q: float, sigma: float, alpha: float) -> float:
    if float(alpha).is_integer():
        return _compute_log_a_int(q, sigma, int(alpha))
    return _compute_log_a_frac(q, sigma, alpha)


def compute_rdp_subsampled_gaussian(
    q: float,
    noise_multiplier: float,
    steps: int,
    orders: tuple[float, ...] = DEFAULT_RDP_ORDERS,
) -> dict[float, float]:
    """RDP of ``steps`` Poisson-subsampled Gaussian steps (rate ``q``, multiplier ``z``).

    ``q=1`` reduces to the plain Gaussian (no amplification); ``z<=0`` is no privacy.
    """
    if noise_multiplier <= 0:
        return {alpha: float("inf") for alpha in orders}
    out: dict[float, float] = {}
    for alpha in orders:
        if q == 0.0:
            rdp = 0.0
        elif q >= 1.0:
            rdp = alpha / (2.0 * noise_multiplier**2)
        else:
            rdp = _compute_log_a(q, noise_multiplier, alpha) / (alpha - 1.0)
        out[alpha] = rdp * steps
    return out


def compute_epsilon_subsampled(
    q: float,
    noise_multiplier: float,
    steps: int,
    delta: float = 1e-5,
    orders: tuple[float, ...] = DEFAULT_RDP_ORDERS,
) -> float:
    """End-to-end ``epsilon`` for ``steps`` Poisson-subsampled Gaussian steps."""
    if noise_multiplier <= 0:
        return float("inf")
    rdp = compute_rdp_subsampled_gaussian(q, noise_multiplier, steps, orders)
    eps, _ = rdp_to_epsilon(rdp, delta)
    return eps


# --- DP-SGD local training (per-example clipping) -----------------------------
#
# The Gaussian mechanism above (``dp_fedavg_update``) clips the *aggregate* weight
# delta -- that bounds one client's influence but accounts no amplification. True
# DP-SGD (Abadi et al., 2016) instead clips the *per-example* gradient inside each
# local step and samples lots by Poisson subsampling, which is what unlocks the
# amplified ``epsilon`` from ``compute_epsilon_subsampled``. The per-example
# gradients are computed with ``torch.func`` (vmap over a single-sample grad).


def per_sample_gradients(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """Per-example loss gradients, ``{param_name: (B, *param_shape)}``."""
    criterion = criterion or nn.CrossEntropyLoss()
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def compute_loss(prm, buf, x, y):
        out = functional_call(model, (prm, buf), (x.unsqueeze(0),))
        return criterion(out, y.unsqueeze(0))

    return vmap(func_grad(compute_loss), in_dims=(None, None, 0, 0))(
        params, buffers, images, labels
    )


def dp_sgd_local_update(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    clip_norm: float,
    noise_multiplier: float,
    sample_rate: float,
    local_steps: int,
    lr: float,
    generator: torch.Generator | None = None,
    criterion: nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """Run ``local_steps`` DP-SGD steps on a client and return the weight delta.

    Each step Poisson-samples a lot (every example kept with prob. ``sample_rate``),
    computes per-example gradients, clips each to L2 ``clip_norm``, sums them, adds
    ``N(0, (noise_multiplier*clip_norm)^2)`` per coordinate, divides by the expected
    lot size, and takes a plain-SGD step. The returned ``delta = w_after - w_before``
    is what the client uploads to FedAvg; the privacy budget is accounted with
    :func:`compute_epsilon_subsampled` over ``rounds * local_steps`` such steps.
    """
    start = {k: v.detach().clone() for k, v in model.state_dict().items()}
    n = images.shape[0]
    expected_lot = max(sample_rate * n, 1e-12)
    std = noise_multiplier * clip_norm
    model.train()
    for _ in range(local_steps):
        mask = torch.rand(n, generator=generator) < sample_rate
        if not bool(mask.any()):
            continue
        xb, yb = images[mask], labels[mask]
        ps = per_sample_gradients(model, xb, yb, criterion)
        flat = torch.cat([g.reshape(g.shape[0], -1) for g in ps.values()], dim=1)
        per_norm = flat.norm(dim=1)
        factor = (clip_norm / (per_norm + 1e-12)).clamp(max=1.0)
        with torch.no_grad():
            for name, p in model.named_parameters():
                g = ps[name]
                shape = (g.shape[0],) + (1,) * (g.dim() - 1)
                summed = (g * factor.reshape(shape)).sum(dim=0)
                noise = torch.randn(summed.shape, generator=generator) * std
                p -= lr * (summed + noise) / expected_lot
    end = model.state_dict()
    return {k: (end[k] - start[k]).detach().clone() for k in start}
