#include "tensor.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void CudaDeleter::operator()(float* pointer) const noexcept {
    if (pointer != nullptr) cudaFree(pointer);
}

Tensor::Tensor(std::vector<size_t> shape_t, Device device_t)
    : shape(std::move(shape_t)), device(device_t) {
    // An empty shape denotes a scalar; any zero dimension denotes an empty tensor.
    sz = 1;
    if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
        sz = 0;
    } else {
        const auto limit = static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
        for (size_t dimension : shape) {
            if (sz > limit / dimension) throw std::overflow_error("Tensor shape is too large");
            sz *= dimension;
        }
    }
    if (device == Device::CPU) {
        cpu_data.resize(sz, 0.0f);
    } else {
        if (!cuda_is_available()) throw std::runtime_error("No CUDA device is available");
        if (sz != 0) {
            float* raw_pointer = nullptr;
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&raw_pointer), bytes()));
            gpu_data.reset(raw_pointer);
            check_cuda(cudaMemset(gpu_data.get(), 0, bytes()));
        }
    }
}

Tensor::Tensor(const Tensor& other) : Tensor(other.shape, other.device) {
    if (device == Device::CPU) {
        cpu_data = other.cpu_data;
    } else if (sz != 0) {
        check_cuda(cudaMemcpy(gpu_data.get(), other.gpu_data.get(), bytes(), cudaMemcpyDeviceToDevice));
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        Tensor copy(other);
        *this = std::move(copy);
    }
    return *this;
}

size_t Tensor::bytes() const { return sz * sizeof(float); }

Tensor Tensor::to(Device target) const {
    if (target == device) return Tensor(*this);
    Tensor result(shape, target);
    if (sz != 0) {
        if (target == Device::GPU) {
            check_cuda(cudaMemcpy(result.gpu_data.get(), cpu_data.data(), bytes(), cudaMemcpyHostToDevice));
        } else {
            copy_to_host(result.cpu_data.data());
        }
    }
    return result;
}

Tensor Tensor::cpu() const { return to(Device::CPU); }
Tensor Tensor::gpu() const { return to(Device::GPU); }

void Tensor::set_data(const float* data, size_t count) {
    if (count != sz) throw std::invalid_argument("Data size must match Tensor size");
    if (sz == 0) return;
    if (device == Device::CPU) std::copy_n(data, sz, cpu_data.data());
    else check_cuda(cudaMemcpy(gpu_data.get(), data, bytes(), cudaMemcpyHostToDevice));
}

void Tensor::copy_to_host(float* destination) const {
    if (sz == 0) return;
    if (device == Device::CPU) std::copy_n(cpu_data.data(), sz, destination);
    else check_cuda(cudaMemcpy(destination, gpu_data.get(), bytes(), cudaMemcpyDeviceToHost));
}

void require_device(const Tensor& tensor, Device expected) {
    if (tensor.device != expected) throw std::invalid_argument("Tensor is on the wrong device");
}

void require_same_shape_and_device(const Tensor& first, const Tensor& second) {
    if (first.device != second.device || first.shape != second.shape) {
        throw std::invalid_argument("Tensor shapes and devices must match");
    }
}

void check_kernel() {
    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());
}

bool cuda_is_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
