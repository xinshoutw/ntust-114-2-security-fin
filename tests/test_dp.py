"""DP-FedAvg defence: L2-clipping bounds sensitivity, Gaussian noise gives (eps, delta)-DP.

The mechanism is the client-level Gaussian mechanism of DP-FedAvg (McMahan et al.,
2018) / DP-SGD (Abadi et al., 2016): each client clips its update to an L2 norm
bound ``C`` (so one client's contribution has bounded sensitivity), then adds
Gaussian noise with std ``z * C`` where ``z`` is the noise multiplier. The privacy
budget ``epsilon`` is accounted via RDP composition over the communication rounds.
"""

import math

import torch

from src.dp_utils import (
    clip_grad_list,
    clip_update,
    compute_epsilon,
    dp_fedavg_grad_list,
    dp_fedavg_update,
    flat_l2_norm,
    gaussian_rdp,
    rdp_to_epsilon,
)


# --- clipping: bound the L2 sensitivity ---------------------------------------

def test_flat_l2_norm_is_joint_over_all_tensors():
    update = {"a": torch.tensor([3.0, 0.0]), "b": torch.tensor([4.0])}
    assert math.isclose(flat_l2_norm(update), 5.0, rel_tol=1e-6)


def test_clip_scales_down_an_oversized_update_to_the_bound():
    update = {"w": torch.ones(100)}  # L2 norm = 10
    clipped, orig_norm = clip_update(update, clip_norm=5.0)
    assert math.isclose(orig_norm, 10.0, rel_tol=1e-6)
    assert math.isclose(flat_l2_norm(clipped), 5.0, rel_tol=1e-5)


def test_clip_leaves_a_small_update_untouched():
    update = {"w": torch.ones(100)}  # L2 norm = 10
    clipped, orig_norm = clip_update(update, clip_norm=20.0)
    assert math.isclose(orig_norm, 10.0, rel_tol=1e-6)
    for k in update:
        assert torch.allclose(clipped[k], update[k])


def test_clip_handles_a_zero_update_without_dividing_by_zero():
    update = {"w": torch.zeros(10)}
    clipped, orig_norm = clip_update(update, clip_norm=1.0)
    assert orig_norm == 0.0
    assert torch.allclose(clipped["w"], torch.zeros(10))


# --- Gaussian mechanism: noise std is z * C -----------------------------------

def test_noise_multiplier_zero_only_clips_no_noise():
    update = {"w": torch.ones(100)}  # norm 10, clip to 5 -> exact, deterministic
    out = dp_fedavg_update(update, clip_norm=5.0, noise_multiplier=0.0)
    assert math.isclose(flat_l2_norm(out), 5.0, rel_tol=1e-5)


def test_added_noise_std_equals_z_times_clip_bound():
    # Small-norm update so clipping is inactive; isolate the noise term.
    update = {"w": torch.full((50000,), 0.01)}  # norm ~2.24 << clip
    clip_norm, z = 10.0, 0.5
    out = dp_fedavg_update(
        update, clip_norm=clip_norm, noise_multiplier=z,
        generator=torch.Generator().manual_seed(0),
    )
    measured = float((out["w"] - update["w"]).std())
    expected = z * clip_norm  # 5.0
    assert abs(measured - expected) / expected < 0.05


def test_grad_list_variant_clips_jointly_and_noises_every_tensor():
    grads = [torch.ones(3, 4), torch.ones(8)]  # joint norm = sqrt(12+8) = sqrt(20)
    clipped, norm = clip_grad_list(grads, clip_norm=1.0)
    assert math.isclose(norm, math.sqrt(20.0), rel_tol=1e-6)
    flat = torch.cat([t.flatten() for t in clipped])
    assert math.isclose(float(flat.norm()), 1.0, rel_tol=1e-5)

    noised = dp_fedavg_grad_list(
        grads, clip_norm=1.0, noise_multiplier=0.3,
        generator=torch.Generator().manual_seed(1),
    )
    assert len(noised) == len(grads)
    for o, g in zip(noised, grads):
        assert o.shape == g.shape
        assert not torch.equal(o, g)


# --- RDP accountant -----------------------------------------------------------

def test_gaussian_rdp_matches_closed_form_steps_times_alpha_over_2z2():
    rdp = gaussian_rdp(noise_multiplier=1.0, steps=1, orders=(2.0,))
    assert math.isclose(rdp[2.0], 1.0, rel_tol=1e-9)  # 1 * 2 / (2 * 1^2)
    rdp = gaussian_rdp(noise_multiplier=2.0, steps=10, orders=(4.0,))
    assert math.isclose(rdp[4.0], 5.0, rel_tol=1e-9)  # 10 * 4 / (2 * 4)


def test_rdp_to_epsilon_uses_the_mironov_conversion():
    # eps(alpha) = rdp(alpha) + ln(1/delta) / (alpha - 1); single order -> exact.
    eps, order = rdp_to_epsilon({2.0: 1.0}, delta=1e-5)
    expected = 1.0 + math.log(1e5) / (2.0 - 1.0)
    assert order == 2.0
    assert math.isclose(eps, expected, rel_tol=1e-9)


def test_epsilon_decreases_as_noise_increases():
    eps_low = compute_epsilon(noise_multiplier=0.5, steps=50, delta=1e-5)
    eps_high = compute_epsilon(noise_multiplier=4.0, steps=50, delta=1e-5)
    assert eps_high < eps_low
    assert eps_high > 0.0


def test_epsilon_grows_with_more_rounds():
    eps_few = compute_epsilon(noise_multiplier=1.0, steps=10, delta=1e-5)
    eps_many = compute_epsilon(noise_multiplier=1.0, steps=100, delta=1e-5)
    assert eps_many > eps_few


def test_zero_noise_gives_infinite_epsilon():
    assert compute_epsilon(noise_multiplier=0.0, steps=50, delta=1e-5) == float("inf")


# --- client integration: the uploaded update is actually clipped --------------

def test_client_clips_the_update_it_uploads_when_dp_is_on():
    from torch.utils.data import TensorDataset

    from src.fl_client import FLClient
    from src.models import LeNet

    torch.manual_seed(0)
    ds = TensorDataset(torch.randn(16, 1, 32, 32), torch.randint(0, 40, (16,)))
    client = FLClient(0, ds, LeNet, "cpu", batch_size=8)
    delta, n = client.train_one_round(
        local_epochs=1, lr=0.01, dp_clip=0.5, dp_noise_multiplier=0.0
    )
    assert n == 16
    assert flat_l2_norm(delta) <= 0.5 + 1e-4  # clip-only (z=0) bounds the norm exactly
