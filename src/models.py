"""Model definitions for the federated-learning / DLG experiments.

We use the LeNet variant from the original *Deep Leakage from Gradients* paper
(Zhu et al., 2019). Two design choices matter for the gradient-inversion attack:

* **Sigmoid activations** rather than ReLU. LBFGS-based DLG minimises a loss
  that depends on second-order gradient information; ReLU's second derivative is
  zero almost everywhere, which cripples reconstruction. Sigmoid is smooth.
* **Strided convolutions** instead of pooling, so the body downsamples while
  keeping the architecture small (~40K parameters on 32x32 input).

For a (1, 32, 32) grayscale input the body produces a 12x8x8 = 768-d feature
vector, fed to a single fully-connected classifier.
"""

from __future__ import annotations

import torch
from torch import nn


class LeNet(nn.Module):
    """3-conv LeNet (~40K params) matching the DLG paper, emitting raw logits."""

    def __init__(self, num_classes: int = 40, in_channels: int = 1) -> None:
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=5, stride=2, padding=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, stride=2, padding=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, stride=1, padding=2),
            act(),
        )
        self.fc = nn.Linear(12 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        out = out.reshape(out.size(0), -1)
        return self.fc(out)


def dlg_weights_init(module: nn.Module) -> None:
    """Uniform init used by the DLG reference code.

    Applying this to an untrained model makes its gradients informative enough
    for the textbook single-image reconstruction demo.
    """
    if hasattr(module, "weight") and module.weight is not None:
        module.weight.data.uniform_(-0.5, 0.5)
    if hasattr(module, "bias") and module.bias is not None:
        module.bias.data.uniform_(-0.5, 0.5)


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LeNet(num_classes=40)
    dummy = torch.randn(2, 1, 32, 32)
    logits = model(dummy)
    print(f"output shape: {tuple(logits.shape)}")
    print(f"parameters: {count_parameters(model):,}")
