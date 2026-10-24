"""Seven operators compared to torch.nn.functional on deterministic inputs."""
import unittest

import numpy as np
import torch
import torch.nn.functional as F

import mytensor as mt


COMPARISONS = []


class OperatorCases:
    device = "cpu"

    def setUp(self):
        self.rng = np.random.default_rng(20261024)

    def array(self, shape):
        return self.rng.normal(size=shape).astype(np.float32)

    def tensor(self, array):
        return mt.from_numpy(np.asarray(array), self.device)

    def compare(self, actual, reference, name, rtol=2e-4, atol=2e-5):
        self.assertIsInstance(actual, mt.Tensor)
        self.assertEqual(actual.device, self.device)
        expected = reference.detach().cpu().numpy()
        got = actual.numpy()
        self.assertEqual(got.shape, expected.shape)
        np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol, equal_nan=True)
        error = np.abs(got.astype(np.float64) - expected.astype(np.float64))
        finite = error[np.isfinite(error)]
        COMPARISONS.append({"name": name, "device": self.device,
                            "shape": list(got.shape), "rtol": rtol, "atol": atol,
                            "max_abs_error": float(finite.max()) if finite.size else 0.0})

    def test_relu_forward_and_backward(self):
        array = self.array((3, 17))
        array[0, :4] = [-1, -0.0, 0.0, 2]
        gradient = self.array(array.shape)
        ref_x = torch.tensor(array, requires_grad=True)
        ref_y = F.relu(ref_x)
        ref_y.backward(torch.tensor(gradient))
        x = self.tensor(array)
        self.compare(mt.relu(x), ref_y, "relu.forward", rtol=0, atol=0)
        self.compare(mt.relu_backward(x, self.tensor(gradient)), ref_x.grad, "relu.dx", rtol=0, atol=0)

    def test_sigmoid_forward_and_backward(self):
        array = self.array((5, 19)) * 5
        array[0, :5] = [-1000, -100, 0, 100, 1000]
        gradient = self.array(array.shape)
        ref_x = torch.tensor(array, requires_grad=True)
        ref_y = F.sigmoid(ref_x)
        ref_y.backward(torch.tensor(gradient))
        y = mt.sigmoid(self.tensor(array))
        self.compare(y, ref_y, "sigmoid.forward", rtol=2e-6, atol=1e-7)
        self.compare(mt.sigmoid_backward(y, self.tensor(gradient)), ref_x.grad, "sigmoid.dx", rtol=2e-5, atol=1e-7)

    def test_linear_forward_and_backward(self):
        for n, cin, cout in ((1, 1, 1), (3, 5, 4), (2, 13, 7)):
            with self.subTest(shape=(n, cin, cout)):
                x, w, b, g = self.array((n, cin)), self.array((cout, cin)), self.array((cout,)), self.array((n, cout))
                ref = [torch.tensor(a, requires_grad=True) for a in (x, w, b)]
                y = F.linear(*ref)
                y.backward(torch.tensor(g))
                tx, tw, tb = (self.tensor(a) for a in (x, w, b))
                self.compare(mt.linear(tx, tw, tb), y, "linear.forward")
                gradients = mt.linear_backward(tx, tw, self.tensor(g))
                for actual, expected, name in zip(gradients, ref, ("dx", "dw", "db")):
                    self.compare(actual, expected.grad, "linear." + name)
                self.compare(mt.fc_forward(tx, tw, tb), y, "linear.alias")

    def test_convolution_forward_and_backward(self):
        for n, cin, h, width, cout in ((2, 2, 5, 7, 3), (1, 1, 1, 1, 1), (1, 3, 4, 6, 2)):
            with self.subTest(shape=(n, cin, h, width, cout)):
                x, w, b = self.array((n, cin, h, width)), self.array((cout, cin, 3, 3)), self.array((cout,))
                gradient = self.array((n, cout, h, width))
                ref = [torch.tensor(a, requires_grad=True) for a in (x, w, b)]
                y = F.conv2d(*ref, stride=1, padding=1)
                y.backward(torch.tensor(gradient))
                tx, tw, tb = (self.tensor(a) for a in (x, w, b))
                self.compare(mt.conv2d(tx, tw, tb), y, "conv2d.forward")
                gradients = mt.conv2d_backward(tx, tw, self.tensor(gradient))
                for actual, expected, name in zip(gradients, ref, ("dx", "dw", "db")):
                    self.compare(actual, expected.grad, "conv2d." + name)

    def test_pooling_even_and_odd_dimensions(self):
        for shape in ((2, 3, 6, 8), (2, 2, 5, 7), (1, 1, 2, 2), (1, 1, 3, 2)):
            with self.subTest(shape=shape):
                x = self.array(shape)
                ref_x = torch.tensor(x, requires_grad=True)
                y = F.max_pool2d(ref_x, 2, 2)
                gradient = self.array(tuple(y.shape))
                y.backward(torch.tensor(gradient))
                tx = self.tensor(x)
                self.compare(mt.max_pool2d(tx), y, "max_pool2d.forward", rtol=0, atol=0)
                self.compare(mt.max_pool2d_backward(tx, self.tensor(gradient)), ref_x.grad, "max_pool2d.dx", rtol=0, atol=0)

    def test_pooling_ties_and_nan(self):
        for values in ([2, 2, 2, 2], [-1, -1, -3, -2], [np.nan, 1, np.nan, 2]):
            with self.subTest(values=values):
                x = np.array(values, dtype=np.float32).reshape(1, 1, 2, 2)
                ref_x = torch.tensor(x, requires_grad=True)
                y = F.max_pool2d(ref_x, 2, 2)
                y.backward(torch.ones_like(y))
                tx = self.tensor(x)
                self.compare(mt.max_pool2d(tx), y, "max_pool2d.ties_forward", rtol=0, atol=0)
                self.compare(mt.max_pool2d_backward(tx, self.tensor(np.ones((1, 1, 1, 1), np.float32))), ref_x.grad,
                             "max_pool2d.ties_dx", rtol=0, atol=0)

    def test_softmax_and_shift_invariance(self):
        for shape in ((1, 1), (3, 10), (257, 13)):
            with self.subTest(shape=shape):
                x = self.array(shape)
                result = mt.softmax(self.tensor(x))
                self.compare(result, F.softmax(torch.tensor(x), dim=1), "softmax.forward", rtol=2e-6, atol=1e-7)
                np.testing.assert_allclose(result.numpy().sum(axis=1), 1, atol=3e-7)
                shifted = mt.softmax(self.tensor(x + np.float32(1000)))
                self.compare(shifted, F.softmax(torch.tensor(x + np.float32(1000)), dim=1), "softmax.shift")
                np.testing.assert_allclose(shifted.numpy(), result.numpy(), rtol=1e-4, atol=1e-5)

    def test_softmax_extreme_logits(self):
        x = np.array([[10000, -10000, 0], [-10000, -10001, -9999], [10000, 10000, 10000]], np.float32)
        result = mt.softmax(self.tensor(x))
        self.compare(result, F.softmax(torch.tensor(x), dim=1), "softmax.extreme", rtol=2e-6, atol=1e-7)
        self.assertTrue(np.isfinite(result.numpy()).all())

    def test_cross_entropy_logits_forward_backward(self):
        for shape in ((1, 1), (4, 10), (257, 13)):
            with self.subTest(shape=shape):
                x = self.array(shape)
                labels = self.rng.integers(0, shape[1], size=shape[0], dtype=np.int64)
                ref_x = torch.tensor(x, requires_grad=True)
                loss = F.cross_entropy(ref_x, torch.tensor(labels), reduction="mean")
                loss.backward()
                tx, ty = self.tensor(x), self.tensor(labels)
                self.compare(mt.cross_entropy(tx, ty), loss, "cross_entropy.forward", rtol=2e-6, atol=1e-6)
                self.compare(mt.cross_entropy_backward(tx, ty), ref_x.grad, "cross_entropy.dx", rtol=2e-5, atol=1e-7)

    def test_cross_entropy_extreme_logits(self):
        x = np.array([[10000, -10000, 0], [-10000, -10001, -9999], [10000, 10000, 10000]], np.float32)
        labels = np.array([1, 0, 2], np.int64)
        ref_x = torch.tensor(x, requires_grad=True)
        loss = F.cross_entropy(ref_x, torch.tensor(labels))
        loss.backward()
        tx, ty = self.tensor(x), self.tensor(labels)
        self.compare(mt.cross_entropy(tx, ty), loss, "cross_entropy.extreme_loss", rtol=2e-6, atol=1e-6)
        self.compare(mt.cross_entropy_backward(tx, ty), ref_x.grad, "cross_entropy.extreme_dx", rtol=2e-5, atol=1e-7)

    def test_lab3_probability_loss_and_fused_gradient(self):
        x = self.array((7, 10))
        labels = self.rng.integers(0, 10, size=7, dtype=np.int64)
        ref_x = torch.tensor(x, requires_grad=True)
        probabilities = F.softmax(ref_x, dim=1)
        loss = F.nll_loss(probabilities.log(), torch.tensor(labels))
        loss.backward()
        p, ty = mt.softmax(self.tensor(x)), self.tensor(labels)
        self.compare(mt.cross_entropy_from_probs(p, ty), loss, "cross_entropy.probability_loss", rtol=2e-6, atol=1e-6)
        self.compare(mt.softmax_cross_entropy_backward(p, ty), ref_x.grad, "cross_entropy.fused_dx", rtol=2e-5, atol=1e-7)

    def test_probability_loss_clamp(self):
        p = np.array([[1, 0], [0, 1]], np.float32)
        labels = np.array([1, 0], np.int64)
        expected = F.nll_loss(torch.tensor(p).clamp_min(1e-12).log(), torch.tensor(labels))
        self.compare(mt.cross_entropy_from_probs(self.tensor(p), self.tensor(labels)), expected,
                     "cross_entropy.clamped", rtol=2e-6, atol=1e-6)

    def test_invalid_labels(self):
        x = self.tensor(np.zeros((2, 3), np.float32))
        for labels in ([-1, 0], [0, 3], [0.5, 1], [np.nan, 1], [np.inf, 0]):
            for op in (mt.cross_entropy, mt.cross_entropy_backward):
                with self.subTest(labels=labels, op=op.__name__), self.assertRaises(ValueError):
                    op(x, self.tensor(np.array(labels, np.float32)))
        with self.assertRaises(ValueError):
            mt.cross_entropy(x, self.tensor(np.array([[0, 1]], np.float32)))
        with self.assertRaises(ValueError):
            mt.cross_entropy(x, self.tensor(np.array([0], np.float32)))

    def test_invalid_probabilities(self):
        for p in ([[0, 0]], [[0.2, 0.2]], [[-1, 2]], [[np.nan, 0]], [[1.1, -0.1]]):
            with self.subTest(p=p), self.assertRaises(ValueError):
                mt.cross_entropy_from_probs(self.tensor(np.array(p, np.float32)), self.tensor(np.array([0])))

    def test_invalid_shapes(self):
        def zeros(shape):
            return mt.Tensor(shape, self.device)
        invalid_calls = (
            lambda: mt.linear(zeros([2, 3]), zeros([4, 2]), zeros([4])),
            lambda: mt.linear(zeros([2, 3]), zeros([4, 3]), zeros([3])),
            lambda: mt.linear_backward(zeros([2, 3]), zeros([4, 3]), zeros([2, 3])),
            lambda: mt.conv2d(zeros([1, 2, 3, 4]), zeros([3, 2, 1, 1]), zeros([3])),
            lambda: mt.conv2d(zeros([1, 2, 3, 4]), zeros([3, 1, 3, 3]), zeros([3])),
            lambda: mt.conv2d_backward(zeros([1, 2, 3, 4]), zeros([3, 2, 3, 3]), zeros([1, 3, 4, 3])),
            lambda: mt.max_pool2d(zeros([1, 1, 1, 2])),
            lambda: mt.max_pool2d(zeros([2, 3])),
            lambda: mt.max_pool2d_backward(zeros([1, 1, 5, 7]), zeros([1, 1, 3, 3])),
            lambda: mt.softmax(zeros([2, 0])),
            lambda: mt.softmax(zeros([2, 3, 4])),
            lambda: mt.cross_entropy(zeros([0, 3]), zeros([0])),
        )
        for index, call in enumerate(invalid_calls):
            with self.subTest(index=index), self.assertRaises(ValueError):
                call()

    def test_device_mismatch(self):
        if not mt.cuda_is_available():
            self.skipTest("CUDA unavailable")
        opposite = "gpu" if self.device == "cpu" else "cpu"
        x, w = mt.Tensor([2, 3], self.device), mt.Tensor([4, 3], opposite)
        with self.assertRaises(ValueError):
            mt.linear(x, w, mt.Tensor([4], self.device))
        with self.assertRaises(ValueError):
            mt.cross_entropy(x, mt.Tensor([2], opposite))
        with self.assertRaises(ValueError):
            mt.conv2d(mt.Tensor([1, 1, 3, 3], self.device), mt.Tensor([2, 1, 3, 3], opposite), mt.Tensor([2], self.device))


class TestOperatorsCPU(OperatorCases, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(mt.cuda_is_available(), "CUDA device unavailable")
class TestOperatorsGPU(OperatorCases, unittest.TestCase):
    device = "gpu"


if __name__ == "__main__":
    unittest.main(verbosity=2)
