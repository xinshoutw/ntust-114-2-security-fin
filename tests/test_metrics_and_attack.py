"""Metrics behave sensibly and iDLG recovers the label from a real gradient."""

import torch

from src.dlg_attack import compute_real_gradients, dlg_attack, idlg_label_inference
from src.metrics import compute_mse, compute_psnr, compute_ssim
from src.models import LeNet


def test_metrics_on_identical_and_noisy_images():
    img = torch.rand(1, 1, 32, 32)
    assert compute_mse(img, img) == 0.0
    assert compute_ssim(img, img) > 0.999
    assert compute_psnr(img, img) == float("inf")  # exact match short-circuits to +inf

    noisy = (img + 0.5 * torch.rand_like(img)).clamp(0, 1)
    assert compute_mse(img, noisy) > 0.0
    assert compute_ssim(img, noisy) < 0.999
    assert compute_psnr(img, noisy) < compute_psnr(img, img)


def test_idlg_recovers_the_true_label():
    torch.manual_seed(0)
    model = LeNet(num_classes=40)
    for label in (0, 7, 23, 39):
        image = torch.randn(1, 1, 32, 32)
        target = torch.tensor([label])
        grads = compute_real_gradients(model, image, target)
        assert idlg_label_inference(grads, num_classes=40) == label


def test_dlg_attack_handles_single_and_batched_labels():
    torch.manual_seed(0)
    model = LeNet(num_classes=40)

    # Single sample, joint-label DLG -> scalar int label.
    img1 = torch.randn(1, 1, 32, 32)
    g1 = compute_real_gradients(model, img1, torch.tensor([3]))
    _, label1, _ = dlg_attack(model, g1, (1, 1, 32, 32), (1, 40), num_iterations=2)
    assert isinstance(label1, int)

    # Batch of 3, joint-label DLG -> one label per row (regression: no .item() crash).
    imgs = torch.randn(3, 1, 32, 32)
    g3 = compute_real_gradients(model, imgs, torch.tensor([1, 2, 3]))
    rec, label3, _ = dlg_attack(model, g3, (3, 1, 32, 32), (3, 40), num_iterations=2)
    assert isinstance(label3, list) and len(label3) == 3
    assert rec.shape == (3, 1, 32, 32)
