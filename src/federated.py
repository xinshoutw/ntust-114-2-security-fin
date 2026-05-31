"""Federated training loop (FedAvg) plus a centralized baseline.

``run_federated_learning`` wires the clients and server together for a number
of communication rounds and tracks test accuracy/loss. ``train_centralized``
trains a single model on the pooled data as an upper-bound reference.
"""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from src import he_utils
from src.data_utils import (
    get_test_loader,
    load_orl_dataset,
    split_dirichlet,
    split_iid,
    split_noniid,
    train_test_split,
)
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
    use_he: bool = False,
    he_rounds: int = 5,
    dp_clip: float | None = None,
    dp_noise_multiplier: float = 0.0,
    split: str = "iid",
    dirichlet_alpha: float = 0.5,
    client_sample_rate: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Train with FedAvg and return history plus the final (and snapshot) weights.

    Returns a dict with:
      - ``history``: list of ``{"round", "accuracy", "loss"}`` (round 0 = init)
      - ``global_state``: final global weights (on CPU)
      - ``snapshots``: ``{round: cpu_state_dict}`` for each round in ``snapshot_rounds``

    ``split`` selects the client data partition: ``"iid"`` (equal random shards)
    or ``"noniid"`` (each client owns a disjoint block of subjects), the latter
    used to demonstrate client drift.

    With ``use_he=True`` the rounds run under CKKS encryption (see
    :func:`run_federated_learning_he`); only ``he_rounds`` rounds are run and the
    result additionally carries per-round ``timing`` and ``comm`` accounting.
    """
    if use_he:
        return run_federated_learning_he(
            num_rounds=he_rounds, num_clients=num_clients, local_epochs=local_epochs,
            lr=lr, device=device, batch_size=batch_size, num_classes=num_classes,
            img_size=img_size, seed=seed, optimizer=optimizer, verbose=verbose,
        )

    device = torch.device(device) if device is not None else get_device()
    torch.manual_seed(seed)

    full = load_orl_dataset(img_size=img_size)
    train_set, test_set = train_test_split(full, seed=seed)
    if split == "dirichlet":
        shards = split_dirichlet(train_set, num_clients=num_clients, alpha=dirichlet_alpha, seed=seed)
    elif split == "noniid":
        shards = split_noniid(train_set, num_clients=num_clients, seed=seed)
    else:
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

    # Partial participation: with client_sample_rate < 1 only a random subset of
    # clients trains each round (standard FedAvg). Combined with many clients and a
    # skewed split, this is what makes client drift bite -- the global model lurches
    # toward whichever clients were sampled. Sampling is seeded for reproducibility.
    part_gen = torch.Generator().manual_seed(seed + 12345)
    n_part = max(1, round(client_sample_rate * num_clients))

    for rnd in range(1, num_rounds + 1):
        global_state = server.get_global_state_dict()
        if n_part >= num_clients:
            participating = clients
        else:
            pick = torch.randperm(num_clients, generator=part_gen)[:n_part].tolist()
            participating = [clients[i] for i in pick]
        updates = []
        for client in participating:
            client.update_model(global_state)
            delta, n = client.train_one_round(
                local_epochs=local_epochs, lr=lr,
                dp_clip=dp_clip, dp_noise_multiplier=dp_noise_multiplier,
            )
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


def train_local_only(
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
    verbose: bool = True,
) -> dict:
    """Each client trains ONLY on its own shard and never aggregates.

    This is the lower-bound baseline that quantifies the *value* of federation: a
    participant who keeps its data local but also refuses to collaborate. Every
    client starts from one shared init (the same kind the FedAvg server uses) and
    runs the same per-round local training as in FedAvg, but the weights are never
    averaged. We report the mean over clients of the global-test accuracy each
    round, so the convergence figure can show ``centralized >= FedAvg >> local-only``
    -- i.e. FedAvg recovers most of the centralized accuracy a lone client (with
    only its 1/num_clients slice of the data) cannot reach on its own.
    """
    device = torch.device(device) if device is not None else get_device()
    torch.manual_seed(seed)

    full = load_orl_dataset(img_size=img_size)
    train_set, test_set = train_test_split(full, seed=seed)
    shards = split_iid(train_set, num_clients=num_clients, seed=seed)
    test_loader = get_test_loader(test_set, batch_size=32)

    clients = [
        FLClient(
            i, shard, LeNet, device,
            batch_size=batch_size, num_classes=num_classes, optimizer=optimizer,
        )
        for i, shard in enumerate(shards)
    ]
    # One shared starting point for every client, so the only difference from
    # FedAvg is the absence of aggregation.
    init_state = {k: v.detach().cpu().clone() for k, v in LeNet(num_classes=num_classes).state_dict().items()}
    for client in clients:
        client.update_model(init_state)

    def mean_metrics() -> tuple[float, float]:
        accs, losses = [], []
        for client in clients:
            acc, loss = evaluate(client.model, test_loader, device)
            accs.append(acc)
            losses.append(loss)
        return sum(accs) / len(accs), sum(losses) / len(losses)

    history: list[dict] = []
    acc, loss = mean_metrics()
    history.append({"round": 0, "accuracy": acc, "loss": loss})
    for rnd in range(1, num_rounds + 1):
        for client in clients:
            # No update_model(): each client keeps and extends its own weights.
            client.train_one_round(local_epochs=local_epochs, lr=lr)
        acc, loss = mean_metrics()
        history.append({"round": rnd, "accuracy": acc, "loss": loss})
        if verbose and (rnd % 5 == 0 or rnd == 1):
            print(f"[local] round {rnd:2d} | mean client acc {acc:.4f} | loss {loss:.4f}")

    return {"history": history}


def run_federated_learning_he(
    num_rounds: int = 5,
    num_clients: int = 4,
    local_epochs: int = 1,
    lr: float = 0.01,
    device: torch.device | str | None = None,
    batch_size: int = 8,
    num_classes: int = 40,
    img_size: int = 32,
    seed: int = 0,
    optimizer: str = "adam",
    encrypt: bool = True,
    verbose: bool = True,
) -> dict:
    """FedAvg where the server only ever sees CKKS-encrypted weight updates.

    Each round the clients train locally, encrypt their weight deltas with the
    secret-key context, and upload ciphertext. The server (public context only)
    homomorphically computes the sample-weighted FedAvg of the deltas and returns
    the encrypted result, which the clients decrypt and apply. The server never
    holds a plaintext gradient, so the DLG attack has nothing to invert.

    The aggregation is intentionally not routed through :class:`FLServer`: that
    aggregator operates on plaintext tensors, whereas here it must run on
    ciphertext (:func:`he_utils.aggregate_encrypted`). Both implement the same
    sample-weighted FedAvg, so they agree under the IID equal split.

    Returns ``history`` plus ``timing`` (per-round encrypt/aggregate/decrypt
    seconds) and ``comm`` (plaintext vs ciphertext bytes for one client update).
    """
    device = torch.device(device) if device is not None else get_device()
    torch.manual_seed(seed)

    full = load_orl_dataset(img_size=img_size)
    train_set, test_set = train_test_split(full, seed=seed)
    shards = split_iid(train_set, num_clients=num_clients, seed=seed)
    test_loader = get_test_loader(test_set, batch_size=32)

    clients = [
        FLClient(
            i, shard, LeNet, device,
            batch_size=batch_size, num_classes=num_classes, optimizer=optimizer,
        )
        for i, shard in enumerate(shards)
    ]

    full_ctx = public_ctx = None
    if encrypt:
        full_ctx = he_utils.create_he_context()
        public_ctx = he_utils.create_public_context(full_ctx)
        if verbose:
            print("[he] CKKS context ready (poly_modulus_degree=8192); server is public-key only")

    # All clients start from one shared initial model (random init leaks nothing).
    eval_model = LeNet(num_classes=num_classes).to(device)
    global_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
    shapes = he_utils.get_shapes(global_state)

    history, timing = [], []
    comm = {}
    acc, loss = evaluate(eval_model, test_loader, device)
    history.append({"round": 0, "accuracy": acc, "loss": loss})
    if verbose:
        print(f"[he] round 0 | acc {acc:.4f} | loss {loss:.4f}  (init)")

    for rnd in range(1, num_rounds + 1):
        # --- clients: local training, producing weight deltas + sample counts ---
        deltas, counts = [], []
        for client in clients:
            client.update_model(global_state)
            delta, n = client.train_one_round(local_epochs=local_epochs, lr=lr)
            deltas.append({k: v.detach().cpu() for k, v in delta.items()})
            counts.append(n)
        total = sum(counts)
        weights = [n / total for n in counts]  # FedAvg sample weighting

        encrypt_time = aggregate_time = decrypt_time = 0.0
        if encrypt:
            # --- clients: encrypt + serialise ciphertext ---
            wire = []
            for delta in deltas:
                t0 = time.perf_counter()
                enc = he_utils.encrypt_gradients(delta, full_ctx)
                serialized = he_utils.serialize_encrypted(enc)
                encrypt_time += time.perf_counter() - t0
                wire.append(serialized)
                if not comm:  # measure once
                    comm = {
                        "plaintext_bytes": sum(v.numel() * 4 for v in delta.values()),
                        "ciphertext_bytes": sum(len(b) for b in serialized.values()),
                    }
            # --- server: homomorphic averaging on the public context only ---
            t0 = time.perf_counter()
            on_server = [he_utils.deserialize_encrypted(w, public_ctx) for w in wire]
            aggregated = he_utils.aggregate_encrypted(on_server, weights=weights)
            aggregated_wire = he_utils.serialize_encrypted(aggregated)
            aggregate_time = time.perf_counter() - t0
            # --- client: decrypt and apply the averaged delta ---
            t0 = time.perf_counter()
            back = he_utils.deserialize_encrypted(aggregated_wire, full_ctx)
            avg_delta = he_utils.decrypt_gradients(back, shapes)
            decrypt_time = time.perf_counter() - t0
        else:
            # Plaintext control: same sample-weighted FedAvg, no CKKS rounding.
            avg_delta = {
                k: sum(w * d[k] for w, d in zip(weights, deltas)) for k in global_state
            }

        global_state = {k: global_state[k] + avg_delta[k] for k in global_state}

        eval_model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
        acc, loss = evaluate(eval_model, test_loader, device)
        history.append({"round": rnd, "accuracy": acc, "loss": loss})
        timing.append({
            "round": rnd, "encrypt": encrypt_time,
            "aggregate": aggregate_time, "decrypt": decrypt_time,
        })
        if verbose:
            print(
                f"[he] round {rnd:2d} | acc {acc:.4f} | loss {loss:.4f} | "
                f"enc {encrypt_time:.2f}s agg {aggregate_time:.2f}s dec {decrypt_time:.2f}s"
            )

    return {
        "history": history,
        "timing": timing,
        "comm": comm,
        "global_state": global_state,
    }
