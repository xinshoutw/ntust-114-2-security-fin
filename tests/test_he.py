"""Encrypted FedAvg must match the plaintext average, and the server must be blind."""

import pytest
import torch

from src import he_utils


def _fake_update(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "conv.weight": torch.randn(12, 1, 5, 5, generator=g),
        "conv.bias": torch.randn(12, generator=g),
        "fc.weight": torch.randn(40, 768, generator=g),  # larger than 4096 slots
        "fc.bias": torch.randn(40, generator=g),
    }


def test_encrypted_fedavg_matches_plaintext_average():
    full_ctx = he_utils.create_he_context()
    public_ctx = he_utils.create_public_context(full_ctx)

    updates = [_fake_update(s) for s in range(4)]
    shapes = he_utils.get_shapes(updates[0])

    # Clients encrypt and serialise; server only ever sees the public context.
    wire = [
        he_utils.serialize_encrypted(he_utils.encrypt_gradients(u, full_ctx)) for u in updates
    ]
    on_server = [he_utils.deserialize_encrypted(w, public_ctx) for w in wire]
    aggregated = he_utils.aggregate_encrypted(on_server, num_clients=len(updates))

    # Back to a client, which holds the secret key.
    back = he_utils.deserialize_encrypted(he_utils.serialize_encrypted(aggregated), full_ctx)
    decrypted = he_utils.decrypt_gradients(back, shapes)

    expected = {k: torch.stack([u[k] for u in updates]).mean(0) for k in shapes}
    for k in shapes:
        assert decrypted[k].shape == expected[k].shape
        assert torch.allclose(decrypted[k], expected[k], atol=1e-2), k


def test_encrypted_fedavg_supports_sample_weighting():
    """Weighted ciphertext aggregation must match the sample-weighted plaintext mean."""
    full_ctx = he_utils.create_he_context()
    public_ctx = he_utils.create_public_context(full_ctx)

    updates = [_fake_update(0), _fake_update(1), _fake_update(2)]
    shapes = he_utils.get_shapes(updates[0])
    counts = [20, 60, 20]  # -> weights 0.2, 0.6, 0.2
    total = sum(counts)
    weights = [c / total for c in counts]

    wire = [
        he_utils.serialize_encrypted(he_utils.encrypt_gradients(u, full_ctx)) for u in updates
    ]
    on_server = [he_utils.deserialize_encrypted(w, public_ctx) for w in wire]
    aggregated = he_utils.aggregate_encrypted(on_server, weights=weights)

    back = he_utils.deserialize_encrypted(he_utils.serialize_encrypted(aggregated), full_ctx)
    decrypted = he_utils.decrypt_gradients(back, shapes)

    expected = {k: sum(w * u[k] for w, u in zip(weights, updates)) for k in shapes}
    for k in shapes:
        assert torch.allclose(decrypted[k], expected[k], atol=1e-2), k


def test_server_cannot_decrypt():
    full_ctx = he_utils.create_he_context()
    public_ctx = he_utils.create_public_context(full_ctx)
    enc = he_utils.encrypt_gradients({"w": torch.randn(16)}, full_ctx)
    wire = he_utils.serialize_encrypted(enc)
    on_server = he_utils.deserialize_encrypted(wire, public_ctx)
    with pytest.raises(Exception):
        on_server["w"].decrypt()
