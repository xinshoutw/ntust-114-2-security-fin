"""Federated-learning client.

Each client owns a private data shard and a local copy of the model. A training
round loads the latest global weights, runs a few local SGD steps, and reports
the resulting **weight update** ``delta = w_local - w_global`` together with its
sample count for FedAvg weighting.

Note: this "update" is a weight delta, *not* an autograd loss gradient. The DLG
attack needs the true per-sample loss gradient and computes it separately (see
:mod:`src.dlg_attack`).
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader


class FLClient:
    def __init__(
        self,
        client_id: int,
        dataset,
        model_cls,
        device: torch.device | str,
        batch_size: int = 8,
        num_classes: int = 40,
        optimizer: str = "adam",
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.device = torch.device(device)
        self.model = model_cls(num_classes=num_classes).to(self.device)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_name = optimizer.lower()
        self._last_delta: dict[str, torch.Tensor] | None = None

    def _make_optimizer(self, lr: float) -> torch.optim.Optimizer:
        if self.optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        raise ValueError(f"unknown optimizer: {self.optimizer_name}")

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    def update_model(self, global_state_dict: dict[str, torch.Tensor]) -> None:
        """Load the latest global weights into the local model."""
        self.model.load_state_dict(copy.deepcopy(global_state_dict))

    def train_one_round(
        self, local_epochs: int = 1, lr: float = 0.01
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Run local training and return ``(weight_delta, num_samples)``."""
        start_weights = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        optimizer = self._make_optimizer(lr)
        self.model.train()
        for _ in range(local_epochs):
            for images, labels in self.loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()
        end_weights = self.model.state_dict()
        delta = {k: (end_weights[k] - start_weights[k]).detach().clone() for k in start_weights}
        self._last_delta = delta
        return delta, self.num_samples

    def get_gradients(self) -> dict[str, torch.Tensor]:
        """Return the most recent weight update (delta), not autograd gradients."""
        if self._last_delta is None:
            raise RuntimeError("call train_one_round() before get_gradients()")
        return self._last_delta
