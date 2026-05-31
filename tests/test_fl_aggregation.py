"""FedAvg correctness: the global update must be the sample-weighted mean of deltas."""

import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.data_utils import split_dirichlet, split_iid, split_noniid
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


def _labelled_dataset(num_subjects: int = 40, per: int = 8) -> TensorDataset:
    labels = torch.arange(num_subjects).repeat_interleave(per)
    images = torch.randn(len(labels), 1, 32, 32)
    return TensorDataset(images, labels)


def test_split_iid_partitions_all_samples_disjointly():
    ds = _labelled_dataset()
    shards = split_iid(ds, num_clients=4)
    counts = [len(s) for s in shards]
    assert sum(counts) == len(ds)
    # IID shards each see most subjects (unlike a non-IID client's disjoint 10),
    # which is the whole point of the IID split -- random, not label-partitioned.
    for shard in shards:
        subjects = {int(ds.tensors[1][i]) for i in shard.indices}
        assert len(subjects) >= 25


def test_split_noniid_gives_each_client_a_disjoint_block_of_subjects():
    ds = _labelled_dataset(num_subjects=40, per=8)
    shards = split_noniid(ds, num_clients=4)
    assert sum(len(s) for s in shards) == len(ds)
    seen: set[int] = set()
    for shard in shards:
        subjects = {int(ds.tensors[1][i]) for i in shard.indices}
        assert len(subjects) == 10  # 40 subjects / 4 clients
        assert subjects.isdisjoint(seen)  # no subject shared across clients
        seen |= subjects
    assert len(seen) == 40


def test_split_dirichlet_partitions_all_samples_with_no_empty_client():
    ds = _labelled_dataset(num_subjects=40, per=8)
    shards = split_dirichlet(ds, num_clients=10, alpha=0.5, seed=0, min_per_client=2)
    # Every sample assigned exactly once, no client left empty.
    all_idx = sorted(i for s in shards for i in s.indices)
    assert all_idx == list(range(len(ds)))
    assert all(len(s) >= 2 for s in shards)


def test_split_dirichlet_small_alpha_is_more_skewed_than_large_alpha():
    # Smaller alpha => each client is dominated by fewer classes (stronger non-IID).
    ds = _labelled_dataset(num_subjects=40, per=8)

    def mean_classes_per_client(alpha):
        shards = split_dirichlet(ds, num_clients=10, alpha=alpha, seed=0)
        counts = [len({int(ds.tensors[1][i]) for i in s.indices}) for s in shards]
        return sum(counts) / len(counts)

    assert mean_classes_per_client(0.1) < mean_classes_per_client(100.0)
