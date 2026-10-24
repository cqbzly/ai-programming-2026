"""Read MNIST IDX/gzip files with the standard library and NumPy.

The data then enters our C++ Tensor through the pybind11::array_t binding.
No torchvision or torch is needed for downloading or reading MNIST.
"""
import gzip
import hashlib
from pathlib import Path
import struct
import tempfile
from urllib.request import Request, urlopen

import numpy as np

from . import Tensor


# Checksums from torchvision.datasets.MNIST.resources.
RESOURCES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data/MNIST/raw"


def _checksum(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_mnist(root=None):
    """Download the four original archives, verify MD5, and reuse valid files."""
    root = Path(root) if root is not None else DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in RESOURCES.items():
        destination = root / name
        if destination.is_file() and _checksum(destination) == expected:
            continue
        errors = []
        for mirror in MIRRORS:
            temporary = None
            try:
                request = Request(mirror + name, headers={"User-Agent": "Lab4-MNIST/0.2"})
                with urlopen(request, timeout=30) as source, tempfile.NamedTemporaryFile(
                    dir=root, suffix=".part", delete=False
                ) as target:
                    temporary = Path(target.name)
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
                if _checksum(temporary) != expected:
                    raise ValueError(f"Checksum mismatch for {name}")
                temporary.replace(destination)
                break
            except (OSError, ValueError) as error:
                errors.append(f"{mirror}: {error}")
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Could not download {name}: " + "; ".join(errors))
    return root


def _read_idx(path, images):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        raw = stream.read()
    header_size = 16 if images else 8
    if len(raw) < header_size:
        raise ValueError(f"Truncated IDX header: {path}")
    if images:
        magic, count, rows, columns = struct.unpack(">IIII", raw[:16])
        if magic != 2051 or rows != 28 or columns != 28 or count == 0:
            raise ValueError(f"Invalid MNIST image header: {path}")
        shape = (count, 1, rows, columns)
    else:
        magic, count = struct.unpack(">II", raw[:8])
        if magic != 2049 or count == 0:
            raise ValueError(f"Invalid MNIST label header: {path}")
        shape = (count,)
    expected_size = header_size + int(np.prod(shape, dtype=np.int64))
    if len(raw) != expected_size:
        raise ValueError(f"IDX payload size does not match header: {path}")
    return np.frombuffer(raw, dtype=np.uint8, offset=header_size).reshape(shape).copy()


def load_mnist_numpy(root=None, train=True, download=False):
    """Return normalized float32 images (N,1,28,28) and int64 labels (N,)."""
    root = Path(root) if root is not None else DEFAULT_ROOT
    if download:
        download_mnist(root)
    prefix = "train" if train else "t10k"
    paths = []
    for suffix in ("images-idx3-ubyte", "labels-idx1-ubyte"):
        raw_path = root / f"{prefix}-{suffix}"
        compressed = raw_path.with_name(raw_path.name + ".gz")
        if raw_path.is_file():
            paths.append(raw_path)
        elif compressed.is_file():
            if _checksum(compressed) != RESOURCES[compressed.name]:
                raise ValueError(f"MNIST checksum mismatch: {compressed}")
            paths.append(compressed)
        else:
            raise FileNotFoundError(f"MNIST data missing at {root}; pass download=True")
    images = _read_idx(paths[0], images=True)
    labels = _read_idx(paths[1], images=False)
    if images.shape[0] != labels.shape[0]:
        raise ValueError("MNIST image and label counts differ")
    if np.any(labels > 9):
        raise ValueError("MNIST labels must be in [0,9]")
    return images.astype(np.float32) / np.float32(255), labels.astype(np.int64)


def load_mnist_tensors(root=None, train=True, device="cpu", limit=None, download=False):
    """Read MNIST and copy a batch through C++ pybind11::array_t into Tensor.

    Tensor uses float32 only. Label values 0..9 are represented exactly and are
    validated as integer class IDs by the cross entropy operator.
    """
    images, labels = load_mnist_numpy(root=root, train=train, download=download)
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        images, labels = images[:limit], labels[:limit]
    return Tensor.from_numpy(images, device), Tensor.from_numpy(labels, device)
