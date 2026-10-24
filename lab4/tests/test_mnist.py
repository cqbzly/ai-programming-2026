import gzip
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import mytensor as mt
from mytensor.mnist import _read_idx, load_mnist_numpy, load_mnist_tensors


class TestMNIST(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.pixels = np.arange(2 * 28 * 28, dtype=np.uint16).astype(np.uint8).reshape(2, 1, 28, 28)
        self.labels = np.array([3, 7], dtype=np.uint8)
        for prefix in ("train", "t10k"):
            (self.root / f"{prefix}-images-idx3-ubyte").write_bytes(
                struct.pack(">IIII", 2051, 2, 28, 28) + self.pixels.tobytes())
            (self.root / f"{prefix}-labels-idx1-ubyte").write_bytes(
                struct.pack(">II", 2049, 2) + self.labels.tobytes())

    def test_read_normalize_and_labels(self):
        for train in (True, False):
            x, y = load_mnist_numpy(self.root, train=train)
            self.assertEqual(x.shape, (2, 1, 28, 28))
            self.assertEqual(x.dtype, np.float32)
            self.assertEqual(y.dtype, np.int64)
            np.testing.assert_array_equal(x, self.pixels.astype(np.float32) / np.float32(255))
            np.testing.assert_array_equal(y, self.labels)

    def test_numpy_to_tensor_cpu(self):
        x, y = load_mnist_tensors(self.root, limit=1)
        self.assertIsInstance(x, mt.Tensor)
        self.assertEqual(x.shape, (1, 1, 28, 28))
        np.testing.assert_array_equal(x.numpy(), self.pixels[:1].astype(np.float32) / np.float32(255))
        np.testing.assert_array_equal(y.numpy(), [3])

    @unittest.skipUnless(mt.cuda_is_available(), "CUDA device unavailable")
    def test_numpy_to_tensor_gpu(self):
        x, y = load_mnist_tensors(self.root, train=False, device="gpu")
        self.assertEqual(x.device, "gpu")
        np.testing.assert_array_equal(x.cpu().numpy(), self.pixels.astype(np.float32) / np.float32(255))
        np.testing.assert_array_equal(y.cpu().numpy(), self.labels.astype(np.float32))

    def test_gzip_idx_reader(self):
        source = self.root / "train-images-idx3-ubyte"
        compressed = self.root / "example.gz"
        with gzip.open(compressed, "wb") as stream:
            stream.write(source.read_bytes())
        np.testing.assert_array_equal(_read_idx(compressed, images=True), self.pixels)

    def test_missing_file(self):
        (self.root / "train-images-idx3-ubyte").unlink()
        with self.assertRaises(FileNotFoundError):
            load_mnist_numpy(self.root)

    def test_corrupt_header_and_payload(self):
        path = self.root / "train-images-idx3-ubyte"
        for content in (b"", struct.pack(">IIII", 0, 2, 28, 28),
                        struct.pack(">IIII", 2051, 2, 27, 28),
                        struct.pack(">IIII", 2051, 2, 28, 28) + b"\x00"):
            path.write_bytes(content)
            with self.subTest(length=len(content)), self.assertRaises(ValueError):
                load_mnist_numpy(self.root)

    def test_mismatched_count_and_bad_labels(self):
        path = self.root / "train-labels-idx1-ubyte"
        for content in (struct.pack(">II", 2049, 1) + b"\x00",
                        struct.pack(">II", 2049, 2) + b"\x00\x0a"):
            path.write_bytes(content)
            with self.assertRaises(ValueError):
                load_mnist_numpy(self.root)

    def test_bad_limit(self):
        for limit in (0, -1, 0.5, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                load_mnist_tensors(self.root, limit=limit)

    def test_corrupt_archive_checksum(self):
        (self.root / "train-images-idx3-ubyte").unlink()
        (self.root / "train-images-idx3-ubyte.gz").write_bytes(b"bad archive")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_mnist_numpy(self.root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
