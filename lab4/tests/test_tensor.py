"""Run with: python -m unittest discover -s tests -v"""
import copy
import gc
import unittest

import numpy as np
import mytensor as mt


class TensorTests(unittest.TestCase):
    def test_constructor_and_metadata(self):
        x = mt.Tensor([2, 3])
        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(x.device, "cpu")
        self.assertEqual(x.dtype, "float32")
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.size(), 6)
        self.assertEqual(x.numel(), 6)
        self.assertIn("Tensor(shape=(2, 3)", repr(x))
        np.testing.assert_array_equal(x.numpy(), np.zeros((2, 3), dtype=np.float32))

    def test_device_enum(self):
        self.assertEqual(mt.Tensor([2], mt.Device.CPU).device, "cpu")

    def test_numpy_roundtrip_and_cast(self):
        for dtype in (np.float32, np.float64, np.int64, np.uint8):
            with self.subTest(dtype=dtype):
                array = np.arange(24, dtype=dtype).reshape(2, 3, 4)
                result = mt.from_numpy(array).numpy()
                self.assertEqual(result.dtype, np.float32)
                np.testing.assert_array_equal(result, array.astype(np.float32))

    def test_noncontiguous_numpy(self):
        array = np.arange(24, dtype=np.float32).reshape(4, 6)
        for view in (array.T, array[:, ::2], array[::-1], np.asfortranarray(array)):
            with self.subTest(strides=view.strides):
                np.testing.assert_array_equal(mt.from_numpy(view).numpy(), view)

    def test_input_is_owned(self):
        array = np.array([1, 2, 3], dtype=np.float32)
        x = mt.from_numpy(array)
        array[:] = -1
        del array
        gc.collect()
        np.testing.assert_array_equal(x.numpy(), [1, 2, 3])

    def test_numpy_output_is_owned(self):
        x = mt.from_numpy(np.array([1, 2, 3], dtype=np.float32))
        result = x.numpy()
        result[0] = 99
        np.testing.assert_array_equal(x.numpy(), [1, 2, 3])
        del x
        gc.collect()
        np.testing.assert_array_equal(result, [99, 2, 3])

    def test_set_data(self):
        x = mt.Tensor([2, 3])
        array = np.arange(6).reshape(2, 3)
        x.set_data(array)
        np.testing.assert_array_equal(x.numpy(), array)
        with self.assertRaises(ValueError):
            x.set_data(array.reshape(3, 2))

    def test_scalar_and_empty(self):
        for array in (np.array(3.5, dtype=np.float32), np.empty((2, 0, 3), dtype=np.float32)):
            with self.subTest(shape=array.shape):
                x = mt.from_numpy(array)
                self.assertEqual(x.shape, array.shape)
                self.assertEqual(x.size(), array.size)
                np.testing.assert_array_equal(x.numpy(), array)
                np.testing.assert_array_equal(x.cpu().numpy(), array)

    def test_invalid_shape_and_device(self):
        with self.assertRaises(ValueError):
            mt.Tensor([-1, 2])
        with self.assertRaises(OverflowError):
            mt.Tensor([2**62, 8])
        with self.assertRaises(ValueError):
            mt.Tensor([2], "invalid")
        with self.assertRaises(ValueError):
            mt.Tensor([2]).to("invalid")

    def test_cpu_deep_copy(self):
        source = mt.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))
        for clone in (source.clone(), source.cpu(), source.to("cpu"), copy.copy(source), copy.deepcopy(source)):
            clone.set_data(np.zeros((2, 3), dtype=np.float32))
            np.testing.assert_array_equal(source.numpy(), np.arange(6).reshape(2, 3))

    @unittest.skipUnless(mt.cuda_is_available(), "CUDA device unavailable")
    def test_gpu_roundtrip(self):
        array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        x = mt.from_numpy(array)
        for gpu in (x.gpu(), x.to("cuda"), mt.from_numpy(array, "gpu"), mt.Tensor([2, 3, 4], mt.Device.GPU)):
            gpu.set_data(array)
            self.assertEqual(gpu.device, "gpu")
            np.testing.assert_array_equal(gpu.numpy(), array)
            np.testing.assert_array_equal(gpu.cpu().numpy(), array)
        self.assertEqual(x.device, "cpu")

    @unittest.skipUnless(mt.cuda_is_available(), "CUDA device unavailable")
    def test_gpu_deep_copy_and_lifetime(self):
        array = np.arange(6, dtype=np.float32)
        source = mt.from_numpy(array, "gpu")
        clones = (source.clone(), source.gpu(), copy.copy(source), copy.deepcopy(source))
        for clone in clones:
            clone.set_data(np.zeros(6, dtype=np.float32))
            np.testing.assert_array_equal(source.numpy(), array)
        clone = source.clone()
        del source
        gc.collect()
        np.testing.assert_array_equal(clone.numpy(), array)

    @unittest.skipUnless(mt.cuda_is_available(), "CUDA device unavailable")
    def test_gpu_scalar_empty_and_zero_init(self):
        for shape in ([], [2, 3], [2, 0, 3]):
            with self.subTest(shape=shape):
                tensor = mt.Tensor(shape, "gpu")
                np.testing.assert_array_equal(tensor.numpy(), np.zeros(shape, dtype=np.float32))
                np.testing.assert_array_equal(tensor.gpu().cpu().numpy(), tensor.numpy())

    @unittest.skipUnless(mt.cuda_is_available(), "CUDA device unavailable")
    def test_repeated_gpu_allocation_and_release(self):
        for _ in range(100):
            x = mt.from_numpy(np.arange(1024, dtype=np.float32), "gpu")
            np.testing.assert_array_equal(x.clone().cpu().numpy(), np.arange(1024))
        gc.collect()


class ActivationTests(unittest.TestCase):
    def devices(self):
        return ["cpu", "gpu"] if mt.cuda_is_available() else ["cpu"]

    def test_relu_forward_backward(self):
        array = np.array([[-3, 0, 2], [4, -5, 0.5]], dtype=np.float32)
        gradient = np.arange(6, dtype=np.float32).reshape(2, 3) + 1
        for device in self.devices():
            with self.subTest(device=device):
                x, dout = mt.from_numpy(array, device), mt.from_numpy(gradient, device)
                np.testing.assert_array_equal(mt.relu(x).numpy(), np.maximum(array, 0))
                np.testing.assert_array_equal(mt.relu_backward(x, dout).numpy(), gradient * (array > 0))

    def test_sigmoid_forward_backward(self):
        array = np.linspace(-10, 10, 257, dtype=np.float32)
        gradient = np.linspace(0, 1, 257, dtype=np.float32)
        expected = 1 / (1 + np.exp(-array))
        for device in self.devices():
            with self.subTest(device=device):
                y = mt.sigmoid(mt.from_numpy(array, device))
                np.testing.assert_allclose(y.numpy(), expected, rtol=2e-6, atol=1e-7)
                dx = mt.sigmoid_backward(y, mt.from_numpy(gradient, device))
                np.testing.assert_allclose(dx.numpy(), gradient * expected * (1 - expected), rtol=2e-5, atol=1e-7)

    def test_empty_activation(self):
        for device in self.devices():
            with self.subTest(device=device):
                x = mt.Tensor([2, 0, 3], device)
                for result in (mt.relu(x), mt.sigmoid(x), mt.relu_backward(x, x), mt.sigmoid_backward(x, x)):
                    self.assertEqual(result.shape, (2, 0, 3))
                    self.assertEqual(result.size(), 0)

    def test_shape_and_device_validation(self):
        x = mt.Tensor([2, 3])
        with self.assertRaises(ValueError):
            mt.relu_backward(x, mt.Tensor([6]))
        with self.assertRaises(ValueError):
            mt.sigmoid_backward(x, mt.Tensor([3, 2]))
        if mt.cuda_is_available():
            with self.assertRaises(ValueError):
                mt.relu_backward(x, x.gpu())

    def test_against_pytorch(self):
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError:
            self.skipTest("PyTorch not installed; NumPy comparisons are covered above")
        rng = np.random.default_rng(42)
        array = rng.normal(size=(7, 13)).astype(np.float32)
        array[0, 0] = 0
        gradient = rng.normal(size=array.shape).astype(np.float32)
        for name, operation, backward in (
            ("relu", functional.relu, mt.relu_backward),
            ("sigmoid", torch.sigmoid, mt.sigmoid_backward),
        ):
            reference_x = torch.tensor(array, requires_grad=True)
            reference_y = operation(reference_x)
            reference_y.backward(torch.tensor(gradient))
            for device in self.devices():
                with self.subTest(operation=name, device=device):
                    x = mt.from_numpy(array, device)
                    y = getattr(mt, name)(x)
                    dx = backward(y if name == "sigmoid" else x, mt.from_numpy(gradient, device))
                    np.testing.assert_allclose(y.numpy(), reference_y.detach().numpy(), rtol=2e-6, atol=1e-7)
                    np.testing.assert_allclose(dx.numpy(), reference_x.grad.numpy(), rtol=2e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
