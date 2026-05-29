"""Federated-learning server implementing FedAvg.

The server keeps the authoritative global model. Each round it receives weight
updates ``(delta, num_samples)`` from the clients, forms the sample-weighted
average of those deltas, and applies it to the global weights:

    w_global <- w_global + sum_i (n_i / N) * delta_i

which is exactly FedAvg over per-client weight updates.
"""

from __future__ import annotations

import copy

import torch


class FLServer:
    def __init__(self, model_cls, device: torch.device | str, num_classes: int = 40) -> None:
        self.device = torch.device(device)
        self.global_model = model_cls(num_classes=num_classes).to(self.device)

    def aggregate(self, client_updates: list[tuple[dict[str, torch.Tensor], int]]) -> None:
        """Apply the FedAvg of client weight updates to the global model."""
        if not client_updates:
            raise ValueError("no client updates to aggregate")
        total_samples = sum(n for _, n in client_updates)
        global_state = self.global_model.state_dict()
        aggregated = {
            k: torch.zeros_like(v, device=self.device) for k, v in global_state.items()
        }
        for delta, num_samples in client_updates:
            weight = num_samples / total_samples
            for k in aggregated:
                aggregated[k] += weight * delta[k].to(self.device)
        new_state = {k: global_state[k] + aggregated[k] for k in global_state}
        self.global_model.load_state_dict(new_state)

    def set_global_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Overwrite the global weights (used by the HE pipeline)."""
        self.global_model.load_state_dict(copy.deepcopy(state_dict))

    def get_global_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a detached copy of the current global weights."""
        return {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}
