#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <vector>

// Adapted from lab2/code.cu. GPU memory still has a single RAII owner.
enum class Device { CPU, GPU };

void check_cuda(cudaError_t status);

struct CudaDeleter {
    void operator()(float* pointer) const noexcept;
};
using CudaPtr = std::unique_ptr<float, CudaDeleter>;

struct Tensor {
    std::vector<size_t> shape;
    std::vector<float> cpu_data;
    CudaPtr gpu_data = nullptr;
    Device device = Device::CPU;
    size_t sz = 0;

    Tensor(std::vector<size_t> shape, Device device = Device::CPU);
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    ~Tensor() = default;

    size_t bytes() const;
    Tensor cpu() const;
    Tensor gpu() const;
    Tensor to(Device target) const;
    void set_data(const float* data, size_t count);
    void copy_to_host(float* destination) const;
};

void require_device(const Tensor& tensor, Device expected);
void require_same_shape_and_device(const Tensor& first, const Tensor& second);
void check_kernel();
bool cuda_is_available();
