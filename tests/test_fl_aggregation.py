"""FedAvg correctness: the global update must be the sample-weighted mean of deltas."""

import torch
from torch import nn

from src.fl_server import FLServer


class TinyModel(nn.Module):
    def __init__(self, num_classes: int = 40) -> None:
        super().__init__()
        self.fc = nn.Linear(3, num_classes, bias=True)


def test_fedavg_is_sample_weighted_mean_of_deltas():
    server = FLServer(TinyModel, device="cpu", num_classes=2)
    server.set_global_state_dict({
        "fc.weight": torch.zeros(2, 3),
        "fc.bias": torch.zeros(2),
    })

    delta_a = {"fc.weight": torch.ones(2, 3), "fc.bias": torch.ones(2)}
    delta_b = {"fc.weight": torch.full((2, 3), 5.0), "fc.bias": torch.full((2,), 5.0)}
    # weights: 10/40 = 0.25 and 30/40 = 0.75  ->  expected delta = 0.25*1 + 0.75*5 = 4.0
    server.aggregate([(delta_a, 10), (delta_b, 30)])

    new_state = server.get_global_state_dict()
    assert torch.allclose(new_state["fc.weight"], torch.full((2, 3), 4.0))
    assert torch.allclose(new_state["fc.bias"], torch.full((2,), 4.0))


def test_equal_weights_recover_plain_average():
    server = FLServer(TinyModel, device="cpu", num_classes=2)
    base = {"fc.weight": torch.full((2, 3), 2.0), "fc.bias": torch.full((2,), 2.0)}
    server.set_global_state_dict(base)

    deltas = [
        {"fc.weight": torch.full((2, 3), float(d)), "fc.bias": torch.full((2,), float(d))}
        for d in (1.0, 3.0)  # equal sample counts -> mean delta = 2.0
    ]
    server.aggregate([(deltas[0], 5), (deltas[1], 5)])

    new_state = server.get_global_state_dict()
    assert torch.allclose(new_state["fc.weight"], torch.full((2, 3), 4.0))  # 2.0 + 2.0
