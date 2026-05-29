"""Data loading utilities for the ORL / AT&T / Olivetti face dataset.

The canonical ORL (AT&T) face mirrors are frequently offline, so we source the
*identical* data from the ``lloydmeta/Olivetti-PNG`` GitHub mirror -- an export
of scikit-learn's Olivetti faces (40 subjects x 10 images, 64x64 grayscale).
Olivetti, AT&T and ORL are three names for the same dataset.

In that mirror the images are ordered by subject, so ``image-N.png`` belongs to
subject ``N // 10`` (its ``N % 10``-th photo). We reorganise them into the usual
ORL layout ``data/orl_faces/s1 .. s40``, each folder holding 10 ``.png`` files.
"""

from __future__ import annotations

import io
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset, TensorDataset

ORL_DIR_NAME = "orl_faces"
NUM_SUBJECTS = 40
IMAGES_PER_SUBJECT = 10

# GitHub mirrors of scikit-learn's Olivetti faces, exported as individual PNGs.
_MIRRORS = (
    "https://codeload.github.com/lloydmeta/Olivetti-PNG/tar.gz/refs/heads/master",
    "https://github.com/lloydmeta/Olivetti-PNG/archive/refs/heads/master.tar.gz",
)
_USER_AGENT = "Mozilla/5.0 (orl-faces-fetcher)"


def _is_populated(root: Path) -> bool:
    """True when ``root`` already holds all 40 subjects with 10 images each."""
    if not root.is_dir():
        return False
    for subject in range(1, NUM_SUBJECTS + 1):
        folder = root / f"s{subject}"
        if not folder.is_dir() or len(list(folder.glob("*.png"))) < IMAGES_PER_SUBJECT:
            return False
    return True


def _download_tarball() -> bytes:
    """Fetch the Olivetti-PNG tarball, trying each mirror in turn."""
    last_err: Exception | None = None
    for url in _MIRRORS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as err:  # noqa: BLE001 - try the next mirror
            last_err = err
    raise RuntimeError(f"Could not download ORL/Olivetti dataset from any mirror: {last_err}")


def _extract_and_reorganize(tarball: bytes, root: Path) -> None:
    """Extract ``image-N.png`` files and lay them out as ``s{subject}/{idx}.png``."""
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(
        fileobj=io.BytesIO(tarball), mode="r:gz"
    ) as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".png") and "image-" in m.name]
        tar.extractall(tmp, members=members, filter="data")  # trusted source, filtered members
        for member in members:
            stem = Path(member.name).stem  # "image-37"
            index = int(stem.split("-")[1])
            subject = index // IMAGES_PER_SUBJECT + 1  # 1..40
            photo = index % IMAGES_PER_SUBJECT + 1  # 1..10
            dest_dir = root / f"s{subject}"
            dest_dir.mkdir(exist_ok=True)
            Path(tmp, member.name).replace(dest_dir / f"{photo}.png")


def ensure_orl_dataset(data_dir: str | Path = "data") -> Path:
    """Return the ORL faces directory, downloading and unpacking it if missing."""
    root = Path(data_dir) / ORL_DIR_NAME
    if _is_populated(root):
        return root
    print(f"[data] ORL faces not found in {root}, downloading from GitHub mirror...")
    _extract_and_reorganize(_download_tarball(), root)
    if not _is_populated(root):
        raise RuntimeError(f"ORL dataset incomplete after download in {root}")
    print(f"[data] ORL faces ready: {NUM_SUBJECTS} subjects x {IMAGES_PER_SUBJECT} images")
    return root


def load_orl_dataset(data_dir: str | Path = "data", img_size: int = 32) -> TensorDataset:
    """Load every ORL image as a normalised grayscale tensor.

    Returns a :class:`TensorDataset` whose ``.tensors`` are
    ``(images, labels)`` with shapes ``(N, 1, img_size, img_size)`` (float32 in
    ``[0, 1]``) and ``(N,)`` (int64 labels in ``0..39``).
    """
    root = ensure_orl_dataset(data_dir)
    images: list[np.ndarray] = []
    labels: list[int] = []
    for subject in range(1, NUM_SUBJECTS + 1):
        folder = root / f"s{subject}"
        for photo_path in sorted(folder.glob("*.png"), key=lambda p: int(p.stem)):
            img = Image.open(photo_path).convert("L").resize((img_size, img_size), Image.BILINEAR)
            images.append(np.asarray(img, dtype=np.float32) / 255.0)
            labels.append(subject - 1)  # 0-indexed labels
    images_tensor = torch.from_numpy(np.stack(images)).unsqueeze(1)  # (N, 1, H, W)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(images_tensor, labels_tensor)


def train_test_split(
    dataset: TensorDataset, test_per_subject: int = 2, seed: int = 0
) -> tuple[Subset, Subset]:
    """Hold out ``test_per_subject`` images per subject for a global test set.

    With the default of 2, the test set has ``40 * 2 = 80`` images and the
    training set has the remaining ``40 * 8 = 320``.
    """
    labels = dataset.tensors[1]
    generator = torch.Generator().manual_seed(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for subject in labels.unique().tolist():
        subject_idx = (labels == subject).nonzero(as_tuple=True)[0]
        perm = subject_idx[torch.randperm(len(subject_idx), generator=generator)]
        test_idx.extend(perm[:test_per_subject].tolist())
        train_idx.extend(perm[test_per_subject:].tolist())
    return Subset(dataset, sorted(train_idx)), Subset(dataset, sorted(test_idx))


def split_iid(dataset, num_clients: int = 4, seed: int = 0) -> list[Subset]:
    """IID-partition a dataset into ``num_clients`` roughly equal shards."""
    indices = list(range(len(dataset)))
    generator = torch.Generator().manual_seed(seed)
    shuffled = [indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()]
    shards = [shuffled[i::num_clients] for i in range(num_clients)]
    return [Subset(dataset, sorted(shard)) for shard in shards]


def get_test_loader(dataset, batch_size: int = 32) -> DataLoader:
    """Build a non-shuffling DataLoader over the global test set."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    ds = load_orl_dataset()
    imgs, lbls = ds.tensors
    print(f"images: {tuple(imgs.shape)}  range=[{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"labels: {tuple(lbls.shape)}  classes={lbls.unique().numel()}")
    train, test = train_test_split(ds)
    clients = split_iid(train, num_clients=4)
    print(f"train={len(train)}  test={len(test)}  clients={[len(c) for c in clients]}")
