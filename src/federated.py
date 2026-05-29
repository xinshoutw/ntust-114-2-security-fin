"""Federated training loop (FedAvg) plus a centralized baseline.

``run_federated_learning`` wires the clients and server together for a number
of communication rounds and tracks test accuracy/loss. ``train_centralized``
trains a single model on the pooled data as an upper-bound reference.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data_utils import get_test_loader, load_orl_dataset, split_iid, train_test_split
from src.fl_client import FLClient
from src.fl_server import FLServer
from src.models import LeNet


def get_device() -> torch.device:
    """Auto-detect the compute device: MPS if available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Return ``(accuracy, mean_loss)`` of ``model`` over ``loader``."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total, total_loss / total


def run_federated_learning(
    num_rounds: int = 50,
    num_clients: int = 4,
    local_epochs: int = 1,
    lr: float = 0.01,
    device: torch.device | str | None = None,
    batch_size: int = 8,
    num_classes: int = 40,
    img_size: int = 32,
    seed: int = 0,
    optimizer: str = "adam",
    snapshot_rounds: tuple[int, ...] = (),
    verbose: bool = True,
) -> dict:
    """Train with FedAvg and return history plus the final (and snapshot) weights.

    Returns a dict with:
      - ``history``: list of ``{"round", "accuracy", "loss"}`` (round 0 = init)
      - ``global_state``: final global weights (on CPU)
      - ``snapshots``: ``{round: cpu_state_dict}`` for each round in ``snapshot_rounds``
    """
    device = torch.device(device) if device is not None else get_device()
    torch.manual_seed(seed)

    full = load_orl_dataset(img_size=img_size)
    train_set, test_set = train_test_split(full, seed=seed)
    shards = split_iid(train_set, num_clients=num_clients, seed=seed)
    test_loader = get_test_loader(test_set, batch_size=32)

    server = FLServer(LeNet, device=device, num_classes=num_classes)
    clients = [
        FLClient(
            i, shard, LeNet, device,
            batch_size=batch_size, num_classes=num_classes, optimizer=optimizer,
        )
        for i, shard in enumerate(shards)
    ]

    def cpu_state() -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in server.get_global_state_dict().items()}

    history: list[dict] = []
    snapshots: dict[int, dict] = {}

    acc, loss = evaluate(server.global_model, test_loader, device)
    history.append({"round": 0, "accuracy": acc, "loss": loss})
    if verbose:
        print(f"[fl] round  0 | acc {acc:.4f} | loss {loss:.4f}  (init)")

    for rnd in range(1, num_rounds + 1):
        global_state = server.get_global_state_dict()
        updates = []
        for client in clients:
            client.update_model(global_state)
            delta, n = client.train_one_round(local_epochs=local_epochs, lr=lr)
            updates.append((delta, n))
        server.aggregate(updates)

        acc, loss = evaluate(server.global_model, test_loader, device)
        history.append({"round": rnd, "accuracy": acc, "loss": loss})
        if rnd in snapshot_rounds:
            snapshots[rnd] = cpu_state()
        if verbose and (rnd % 5 == 0 or rnd == 1):
            print(f"[fl] round {rnd:2d} | acc {acc:.4f} | loss {loss:.4f}")

    return {"history": history, "global_state": cpu_state(), "snapshots": snapshots}


def train_centralized(
    num_epochs: int = 50,
    lr: float = 0.01,
    device: torch.device | str | None = None,
    batch_size: int = 8,
    num_classes: int = 40,
    img_size: int = 32,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Train one model on the pooled training data; mirrors the FL evaluation."""
    device = torch.device(device) if device is not None else get_device()
    torch.manual_seed(seed)

    full = load_orl_dataset(img_size=img_size)
    train_set, test_set = train_test_split(full, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = get_test_loader(test_set, batch_size=32)

    model = LeNet(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: list[dict] = []
    acc, loss = evaluate(model, test_loader, device)
    history.append({"round": 0, "accuracy": acc, "loss": loss})

    for epoch in range(1, num_epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()
        acc, loss = evaluate(model, test_loader, device)
        history.append({"round": epoch, "accuracy": acc, "loss": loss})
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"[central] epoch {epoch:2d} | acc {acc:.4f} | loss {loss:.4f}")

    return {"history": history, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
