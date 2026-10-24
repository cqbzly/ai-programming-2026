// Activation implementations migrated from lab2/code.cu.
#include "activation.h"
#include <cmath>
using std::exp;

Tensor relu_forward_cpu(const Tensor &x) {
    require_device(x, Device::CPU);
    Tensor tmp(x.shape, Device::CPU);
    for(size_t i = 0; i < x.sz; i ++) tmp.cpu_data[i] = x.cpu_data[i] > 0 ? x.cpu_data[i] : 0;
    return tmp;
}

__global__ void relu_forward_kernel(const float *input, float *output, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx < n) {
        output[idx] = input[idx] > 0 ? input[idx] : 0; 
    }
}

constexpr int threads_per_block = 256;

Tensor relu_forward_gpu(const Tensor &x) {
    require_device(x, Device::GPU);
    size_t n = x.sz;
    unsigned int cnt = static_cast<unsigned int>((n + threads_per_block - 1) / threads_per_block);
    Tensor out(x.shape, Device::GPU);
    if (n == 0) return out;
    relu_forward_kernel<<<cnt, threads_per_block>>>(x.gpu_data.get(), out.gpu_data.get(), n);
    check_kernel();
    return out;
}

Tensor relu_forward(const Tensor &x) {
    if(x.device == Device::CPU)return relu_forward_cpu(x);
    return relu_forward_gpu(x);
}

Tensor relu_backward_cpu(const Tensor &x, const Tensor &dout) {
    require_device(x, Device::CPU);
    require_same_shape_and_device(x, dout);
    Tensor dx(x.shape, Device::CPU);
    for(size_t i = 0; i < x.sz; i++) {
        dx.cpu_data[i] = x.cpu_data[i] > 0 ? dout.cpu_data[i] : 0;
    }
    return dx;
}

__global__ void relu_backward_kernel(const float *input, const float *dout, float *dx, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx < n) {
        dx[idx] = input[idx] > 0 ? dout[idx] : 0; 
    }
}

Tensor relu_backward_gpu(const Tensor &x, const Tensor &dout) {
    require_device(x, Device::GPU);
    require_same_shape_and_device(x, dout);
    size_t n = x.sz;
    unsigned int cnt = static_cast<unsigned int>((n + threads_per_block - 1) / threads_per_block);
    Tensor dx(x.shape, Device::GPU);
    if (n == 0) return dx;
    relu_backward_kernel<<<cnt, threads_per_block>>>(x.gpu_data.get(), dout.gpu_data.get(), dx.gpu_data.get(), n);
    check_kernel();
    return dx;
}

Tensor relu_backward(const Tensor &x, const Tensor &dout) {
    if(x.device == Device::CPU)return relu_backward_cpu(x, dout);
    return relu_backward_gpu(x, dout);
}

Tensor sigmoid_forward_cpu(const Tensor &x) {
    require_device(x, Device::CPU);
    Tensor tmp(x.shape, Device::CPU);
    for(size_t i = 0; i < x.sz; i ++) tmp.cpu_data[i] = 1.0f / (1.0f + exp(-x.cpu_data[i]));
    return tmp;
}

__global__ void sigmoid_forward_kernel(const float *input, float *output, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

Tensor sigmoid_forward_gpu(const Tensor &x) {
    require_device(x, Device::GPU);
    size_t n = x.sz;
    unsigned int cnt = static_cast<unsigned int>((n + threads_per_block - 1) / threads_per_block);
    Tensor out(x.shape, Device::GPU);
    if (n == 0) return out;
    sigmoid_forward_kernel<<<cnt, threads_per_block>>>(x.gpu_data.get(), out.gpu_data.get(), n);
    check_kernel();
    return out;
}

Tensor sigmoid_forward(const Tensor &x) {
    if(x.device == Device::CPU)return sigmoid_forward_cpu(x);
    return sigmoid_forward_gpu(x);
}

Tensor sigmoid_backward_cpu(const Tensor &y, const Tensor &dout) {
    require_device(y, Device::CPU);
    require_same_shape_and_device(y, dout);
    Tensor dx(y.shape, Device::CPU);
    for(size_t i = 0; i < y.sz; i++) {
        dx.cpu_data[i] = dout.cpu_data[i] * y.cpu_data[i] * (1.0f - y.cpu_data[i]); 
    }
    return dx;
}

__global__ void sigmoid_backward_kernel(const float *input, const float *dout, float *dx, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx < n) {
        dx[idx] = dout[idx] * input[idx] * (1.0f - input[idx]);
    }
}

Tensor sigmoid_backward_gpu(const Tensor &y, const Tensor &dout) {
    require_device(y, Device::GPU);
    require_same_shape_and_device(y, dout);
    size_t n = y.sz;
    unsigned int cnt = static_cast<unsigned int>((n + threads_per_block - 1) / threads_per_block);
    Tensor dx(y.shape, Device::GPU);
    if (n == 0) return dx;
    sigmoid_backward_kernel<<<cnt, threads_per_block>>>(y.gpu_data.get(), dout.gpu_data.get(), dx.gpu_data.get(), n);
    check_kernel();
    return dx;
}

Tensor sigmoid_backward(const Tensor &x, const Tensor &dout) {
    if(x.device == Device::CPU)return sigmoid_backward_cpu(x, dout);
    return sigmoid_backward_gpu(x, dout);
}


