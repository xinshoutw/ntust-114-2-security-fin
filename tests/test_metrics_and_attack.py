"""Metrics behave sensibly and iDLG recovers the label from a real gradient."""

import torch

from src.dlg_attack import compute_real_gradients, idlg_label_inference
from src.metrics import compute_mse, compute_psnr, compute_ssim
from src.models import LeNet


def test_metrics_on_identical_and_noisy_images():
    img = torch.rand(1, 1, 32, 32)
    assert compute_mse(img, img) == 0.0
    assert compute_ssim(img, img) > 0.999
    assert compute_psnr(img, img) > 100  # skimage caps an exact match very high

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
