"""Differential-privacy noise is calibrated to the update's RMS and is opt-in."""

import torch

from src.dp_utils import (
    add_relative_gaussian_noise,
    add_relative_gaussian_noise_list,
)


def _rms(flat: torch.Tensor) -> float:
    return float(flat.pow(2).mean().sqrt())


def test_sigma_zero_is_a_noop():
    update = {"a": torch.randn(5, 7), "b": torch.randn(3)}
    out = add_relative_gaussian_noise(update, 0.0)
    for k in update:
        assert torch.equal(out[k], update[k])
        assert out[k] is not update[k]  # a clone, not the same object


def test_noise_std_matches_sigma_times_rms():
    g = torch.Generator().manual_seed(0)
    update = {"w": torch.randn(20000, generator=g) * 3.0}  # large for a tight estimate
    rms = _rms(update["w"])
    sigma = 0.1
    out = add_relative_gaussian_noise(update, sigma, generator=torch.Generator().manual_seed(1))
    measured = float((out["w"] - update["w"]).std())
    assert abs(measured - sigma * rms) / (sigma * rms) < 0.05  # within 5%


def test_list_variant_perturbs_every_tensor():
    grads = [torch.randn(12, 1, 5, 5), torch.randn(12), torch.randn(40, 768)]
    out = add_relative_gaussian_noise_list(grads, 0.2, generator=torch.Generator().manual_seed(2))
    assert len(out) == len(grads)
    for o, g in zip(out, grads):
        assert o.shape == g.shape
        assert not torch.equal(o, g)


def test_more_noise_means_larger_perturbation():
    update = {"w": torch.randn(5000, generator=torch.Generator().manual_seed(3))}
    low = add_relative_gaussian_noise(update, 0.05, generator=torch.Generator().manual_seed(4))
    high = add_relative_gaussian_noise(update, 0.5, generator=torch.Generator().manual_seed(4))
    dev_low = float((low["w"] - update["w"]).abs().mean())
    dev_high = float((high["w"] - update["w"]).abs().mean())
    assert dev_high > dev_low
