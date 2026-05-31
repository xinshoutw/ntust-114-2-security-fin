"""Deep Leakage from Gradients (DLG / iDLG) attack.

Threat model: an honest-but-curious server receives a client's per-sample loss
gradient and tries to reconstruct the private training image from it alone.

The attack optimises a dummy image (and, in plain DLG, a dummy label) so that
the gradient it produces matches the observed gradient. iDLG additionally infers
the ground-truth label analytically from the last layer's gradient, which makes
the image optimisation far more stable.

References: Zhu et al., "Deep Leakage from Gradients" (2019); Zhao et al.,
"iDLG: Improved Deep Leakage from Gradients" (2020).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def compute_real_gradients(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    criterion: nn.Module | None = None,
) -> list[torch.Tensor]:
    """Compute the loss gradient of ``model`` on a single (image, label) pair.

    This is what the server observes: the gradient a client would upload after
    one backward pass on the private sample. Returns a detached list of tensors
    in ``model.parameters()`` order.
    """
    criterion = criterion or nn.CrossEntropyLoss()
    model.zero_grad()
    loss = criterion(model(image), label)
    grads = torch.autograd.grad(loss, model.parameters())
    return [g.detach().clone() for g in grads]


def idlg_label_inference(gradients: list[torch.Tensor], num_classes: int) -> int:
    """Recover the ground-truth label from the final FC layer's gradient.

    For single-sample cross-entropy, dL/dlogit_i = softmax_i - onehot_i, which is
    negative only for the true class. Since the last layer's weight gradient is
    that term times the (non-negative, Sigmoid) features, the row whose summed
    gradient is most negative identifies the label.
    """
    last_weight_grad = gradients[-2]  # fc.weight gradient, shape (num_classes, in_features)
    return int(torch.argmin(last_weight_grad.sum(dim=1)).item())


def _soft_cross_entropy(pred: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a soft (probability) target, as in the DLG paper."""
    return torch.mean(torch.sum(-target_prob * F.log_softmax(pred, dim=-1), dim=-1))


def dlg_attack(
    model: nn.Module,
    real_gradients: list[torch.Tensor],
    image_shape: tuple[int, ...],
    label_shape: tuple[int, ...],
    num_iterations: int = 300,
    device: torch.device | str = "cpu",
    lr: float = 1.0,
    known_label: int | None = None,
    seed: int = 0,
    image_log: list[tuple[int, torch.Tensor]] | None = None,
    log_iters: tuple[int, ...] = (),
) -> tuple[torch.Tensor, int | list[int], list[float]]:
    """Reconstruct an input image by matching gradients.

    If ``known_label`` is given (iDLG), only the image is optimised against a
    fixed label; otherwise (plain DLG) a dummy label is optimised jointly.

    Returns ``(recovered_image, recovered_label, loss_history)`` with the image
    detached on CPU and the loss history one entry per LBFGS evaluation.

    Pass an ``image_log`` list together with ``log_iters`` to capture
    ``(iteration, dummy_image)`` snapshots (CPU copies) at those iterations --
    iteration 0 is the random initialisation -- for the noise->face progression
    figure.
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    real_gradients = [g.to(device) for g in real_gradients]

    generator = torch.Generator(device="cpu").manual_seed(seed)
    dummy_image = torch.randn(image_shape, generator=generator).to(device).requires_grad_(True)

    params = [dummy_image]
    dummy_label = None
    if known_label is None:
        dummy_label = torch.randn(label_shape, generator=generator).to(device).requires_grad_(True)
        params.append(dummy_label)
    else:
        fixed_label = torch.tensor([known_label], device=device)

    optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=20)
    criterion = nn.CrossEntropyLoss()
    history: list[float] = []

    if image_log is not None and 0 in log_iters:
        image_log.append((0, dummy_image.detach().cpu().clone()))

    for it in range(num_iterations):

        def closure():
            optimizer.zero_grad()
            pred = model(dummy_image)
            if known_label is None:
                loss = _soft_cross_entropy(pred, F.softmax(dummy_label, dim=-1))
            else:
                loss = criterion(pred, fixed_label)
            dummy_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            grad_diff = sum(((dg - rg) ** 2).sum() for dg, rg in zip(dummy_grads, real_gradients))
            grad_diff.backward()
            return grad_diff

        loss_value = optimizer.step(closure)
        history.append(float(loss_value.item()))
        if image_log is not None and (it + 1) in log_iters:
            image_log.append((it + 1, dummy_image.detach().cpu().clone()))

    if known_label is not None:
        recovered_label = known_label
    else:
        pred = torch.argmax(dummy_label, dim=-1)
        # Single-sample DLG returns a scalar label; a batch returns one per row.
        recovered_label = int(pred.item()) if pred.numel() == 1 else pred.tolist()
    return dummy_image.detach().cpu(), recovered_label, history
