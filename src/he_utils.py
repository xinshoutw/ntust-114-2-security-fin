"""Homomorphic-encryption helpers (TenSEAL CKKS) for private FedAvg.

Defence against the DLG attack: clients encrypt their weight updates before
upload, so the honest-but-curious server only ever sees ciphertext. It can still
sum the updates and scale by 1/N homomorphically (CKKS supports ciphertext
addition and plaintext multiplication), but it never holds a plaintext gradient,
so there is nothing for DLG to invert.

Key handling:
  * the *full* context (with secret key) stays on the clients;
  * the server gets a *public* copy with the secret key stripped out.

CKKS vectors may be longer than the ring's slot count -- TenSEAL packs them
across several ciphertexts internally -- so each parameter is encrypted as one
vector and no manual chunking is needed.
"""

from __future__ import annotations

import tenseal as ts
import torch

DEFAULT_POLY_MODULUS_DEGREE = 8192
DEFAULT_COEFF_MOD_BIT_SIZES = (60, 40, 40, 60)
DEFAULT_GLOBAL_SCALE = 2**40


def create_he_context(
    poly_modulus_degree: int = DEFAULT_POLY_MODULUS_DEGREE,
    coeff_mod_bit_sizes=DEFAULT_COEFF_MOD_BIT_SIZES,
    global_scale: float = DEFAULT_GLOBAL_SCALE,
) -> ts.Context:
    """Create a CKKS context holding the secret key (client side)."""
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=list(coeff_mod_bit_sizes),
    )
    context.global_scale = global_scale
    context.generate_galois_keys()
    return context


def create_public_context(full_context: ts.Context) -> ts.Context:
    """Return a copy of the context with the secret key removed (server side)."""
    public = full_context.copy()
    public.make_context_public()
    return public


def get_shapes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Size]:
    """Record each tensor's shape so decryption can restore it."""
    return {k: v.shape for k, v in state_dict.items()}


def encrypt_gradients(
    gradients_dict: dict[str, torch.Tensor], context: ts.Context
) -> dict[str, ts.CKKSVector]:
    """Encrypt each parameter tensor as a flattened CKKS vector."""
    return {
        name: ts.ckks_vector(context, tensor.detach().cpu().flatten().tolist())
        for name, tensor in gradients_dict.items()
    }


def decrypt_gradients(
    encrypted_dict: dict[str, ts.CKKSVector], shapes_dict: dict[str, torch.Size]
) -> dict[str, torch.Tensor]:
    """Decrypt each vector and reshape it back to its original tensor shape.

    The vectors must be linked to a context that holds the secret key (i.e.
    deserialised with the full context).
    """
    return {
        name: torch.tensor(vec.decrypt(), dtype=torch.float32).reshape(shapes_dict[name])
        for name, vec in encrypted_dict.items()
    }


def serialize_encrypted(encrypted_dict: dict[str, ts.CKKSVector]) -> dict[str, bytes]:
    """Serialise each encrypted vector to bytes for transmission."""
    return {name: vec.serialize() for name, vec in encrypted_dict.items()}


def deserialize_encrypted(
    data_dict: dict[str, bytes], context: ts.Context
) -> dict[str, ts.CKKSVector]:
    """Reconstruct encrypted vectors, linking them to ``context``."""
    return {name: ts.ckks_vector_from(context, data) for name, data in data_dict.items()}


def aggregate_encrypted(
    encrypted_list: list[dict[str, ts.CKKSVector]], num_clients: int
) -> dict[str, ts.CKKSVector]:
    """FedAvg on ciphertext: sum the client vectors and scale by 1/num_clients.

    Runs entirely on encrypted data, so it is safe for the public-context server.
    """
    if not encrypted_list:
        raise ValueError("no encrypted updates to aggregate")
    names = encrypted_list[0].keys()
    aggregated: dict[str, ts.CKKSVector] = {}
    for name in names:
        acc = encrypted_list[0][name]
        for client in encrypted_list[1:]:
            acc = acc + client[name]
        aggregated[name] = acc * (1.0 / num_clients)
    return aggregated


def encrypted_size_bytes(encrypted_dict: dict[str, ts.CKKSVector]) -> int:
    """Total serialised size of an encrypted update, for communication accounting."""
    return sum(len(data) for data in serialize_encrypted(encrypted_dict).values())
