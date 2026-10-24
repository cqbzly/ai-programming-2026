#include "activation.h"
#include "ops.h"
#include "tensor.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {
Device parse_device(const std::string& device) {
    if (device == "cpu") return Device::CPU;
    if (device == "gpu" || device == "cuda") return Device::GPU;
    throw std::invalid_argument("device must be 'cpu', 'gpu', or 'cuda'");
}

std::vector<size_t> checked_shape(const std::vector<py::ssize_t>& shape) {
    std::vector<size_t> result;
    for (auto dimension : shape) {
        if (dimension < 0) throw std::invalid_argument("Tensor dimensions must be non-negative");
        result.push_back(static_cast<size_t>(dimension));
    }
    return result;
}

std::vector<py::ssize_t> numpy_shape(const Tensor& tensor) {
    return {tensor.shape.begin(), tensor.shape.end()};
}

py::tuple shape_tuple(const Tensor& tensor) {
    py::tuple result(tensor.shape.size());
    for (size_t i = 0; i < tensor.shape.size(); ++i) result[i] = tensor.shape[i];
    return result;
}

Tensor from_numpy(const FloatArray& array, const std::string& device) {
    auto info = array.request();
    Tensor tensor(checked_shape(info.shape), parse_device(device));
    tensor.set_data(static_cast<const float*>(info.ptr), static_cast<size_t>(info.size));
    return tensor;
}

py::array_t<float> to_numpy(const Tensor& tensor) {
    // NumPy owns a new allocation. No dangling view after Tensor destruction.
    py::array_t<float> array(numpy_shape(tensor));
    tensor.copy_to_host(array.mutable_data());
    return array;
}
}  // namespace

PYBIND11_MODULE(_mytensor, m) {
    m.doc() = "CPU/CUDA Tensor and seven neural network operators";
    m.attr("__version__") = "0.2.0";
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU);

    py::class_<Tensor>(m, "Tensor")
        .def(py::init([](const std::vector<py::ssize_t>& shape, const std::string& device) {
            return Tensor(checked_shape(shape), parse_device(device));
        }), py::arg("shape"), py::arg("device") = "cpu")
        .def(py::init([](const std::vector<py::ssize_t>& shape, Device device) {
            return Tensor(checked_shape(shape), device);
        }), py::arg("shape"), py::arg("device"))
        .def_static("from_numpy", &from_numpy, py::arg("array"), py::arg("device") = "cpu",
            "Copy a NumPy-compatible array into an owned float32 Tensor.")
        .def("numpy", &to_numpy, "Return an owned float32 NumPy copy (GPU data is copied to CPU).")
        .def("set_data", [](Tensor& tensor, const FloatArray& array) {
            auto info = array.request();
            if (checked_shape(info.shape) != tensor.shape) {
                throw std::invalid_argument("Array shape must match Tensor shape");
            }
            tensor.set_data(static_cast<const float*>(info.ptr), static_cast<size_t>(info.size));
        }, py::arg("array"))
        .def_property_readonly("shape", &shape_tuple)
        .def_property_readonly("ndim", [](const Tensor& tensor) { return tensor.shape.size(); })
        .def_property_readonly("device", [](const Tensor& tensor) {
            return tensor.device == Device::CPU ? "cpu" : "gpu";
        })
        .def_property_readonly("dtype", [](const Tensor&) { return "float32"; })
        .def("size", [](const Tensor& tensor) { return tensor.sz; })
        .def("numel", [](const Tensor& tensor) { return tensor.sz; })
        .def("cpu", &Tensor::cpu)
        .def("gpu", &Tensor::gpu)
        .def("to", [](const Tensor& tensor, const std::string& device) {
            return tensor.to(parse_device(device));
        }, py::arg("device"))
        .def("clone", [](const Tensor& tensor) { return Tensor(tensor); })
        .def("__copy__", [](const Tensor& tensor) { return Tensor(tensor); })
        .def("__deepcopy__", [](const Tensor& tensor, py::dict) { return Tensor(tensor); }, py::arg("memo"))
        .def("print_data", [](const Tensor& tensor) { py::print(to_numpy(tensor)); })
        .def("__repr__", [](const Tensor& tensor) {
            std::ostringstream result;
            result << "Tensor(shape=" << py::str(shape_tuple(tensor)).cast<std::string>()
                   << ", device='" << (tensor.device == Device::CPU ? "cpu" : "gpu")
                   << "', dtype='float32')";
            return result.str();
        });

    m.def("from_numpy", &from_numpy, py::arg("array"), py::arg("device") = "cpu");
    m.def("cuda_is_available", &cuda_is_available);
    m.def("relu", &relu_forward, py::arg("x"));
    m.def("relu_forward", &relu_forward, py::arg("x"));
    m.def("relu_backward", &relu_backward, py::arg("x"), py::arg("dout"));
    m.def("sigmoid", &sigmoid_forward, py::arg("x"));
    m.def("sigmoid_forward", &sigmoid_forward, py::arg("x"));
    m.def("sigmoid_backward", &sigmoid_backward, py::arg("y"), py::arg("dout"),
        "y must be the sigmoid forward OUTPUT, as in Lab2.");
    m.def("linear", &linear, py::arg("x"), py::arg("weight"), py::arg("bias"));
    m.def("fc_forward", &linear, py::arg("x"), py::arg("weight"), py::arg("bias"));
    m.def("linear_backward", &linear_backward, py::arg("x"), py::arg("weight"), py::arg("dout"), "Return (dx, dw, db).");
    m.def("fc_backward", &linear_backward, py::arg("x"), py::arg("weight"), py::arg("dout"));
    m.def("conv2d", &conv2d, py::arg("x"), py::arg("weight"), py::arg("bias"), "3x3 convolution; stride=1, padding=1.");
    m.def("conv2d_backward", &conv2d_backward, py::arg("x"), py::arg("weight"), py::arg("dout"), "Return (dx, dw, db).");
    m.def("max_pool2d", &max_pool2d, py::arg("x"), "2x2 max pooling; stride=2, floor output size.");
    m.def("max_pool2d_backward", &max_pool2d_backward, py::arg("x"), py::arg("dout"), "Recompute the argmax from x and return dx.");
    m.def("softmax", &softmax, py::arg("x"), "Stable softmax over classes, input shape (N,C).");
    m.def("cross_entropy", &cross_entropy, py::arg("logits"), py::arg("labels"), "Mean cross entropy from raw logits. Return a scalar Tensor.");
    m.def("cross_entropy_backward", &cross_entropy_backward, py::arg("logits"), py::arg("labels"), "Return gradient wrt logits for mean loss.");
    m.def("cross_entropy_from_probs", &cross_entropy_from_probs, py::arg("probabilities"), py::arg("labels"), "Lab3 loss from probabilities; clamp minimum to 1e-12.");
    m.def("softmax_cross_entropy_backward", &softmax_cross_entropy_backward, py::arg("probabilities"), py::arg("labels"), "Lab3 fused derivative wrt logits: (p-one_hot)/N.");
}
