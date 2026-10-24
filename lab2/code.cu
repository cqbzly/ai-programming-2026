#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <cmath>
#include <memory>
#include <stdexcept>

using namespace std;

enum class Device {
    CPU,
    GPU
};

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw runtime_error(cudaGetErrorString(status));
    }
}

struct CudaDeleter {
    void operator()(float *pointer) const noexcept {
        if(pointer != nullptr) {
            cudaFree(pointer);
        }
    }
};

using CudaPtr = unique_ptr<float, CudaDeleter>;

struct Tensor {
    vector<size_t> shape;
    vector<float> cpu_data;
    CudaPtr gpu_data = nullptr;
    Device device = Device::CPU;
    size_t sz = 0;
    size_t bytes() {
        return sz * sizeof(float);
    }
    Tensor(vector<size_t> shape_t, Device device_t) {
        sz = 1;
        shape = shape_t;
        device = device_t;
        for(int i = 0; i < shape.size(); i++) sz *= shape[i];
        if(device == Device::GPU) {
            float* raw_pointer = nullptr;
            check_cuda(cudaMalloc(
                reinterpret_cast<void**>(&raw_pointer),
                bytes()
            ));
            gpu_data.reset(raw_pointer);
            check_cuda(cudaMemset(gpu_data.get(), 0, bytes()));
        }
        else {
            cpu_data.resize(sz);
        }
    }

    Tensor gpu() {
        if(device != Device::CPU) {
            throw invalid_argument("gpu() expects a CPU Tensor");
        }
        Tensor tmp (shape, Device::GPU);
        check_cuda(cudaMemcpy(
            tmp.gpu_data.get(),
            cpu_data.data(),
            bytes(),
            cudaMemcpyHostToDevice
        ));
        return tmp;
    }
    Tensor cpu() {
        if(device != Device::GPU) {
            throw invalid_argument("cpu() expects a GPU Tensor");
        }
        Tensor tmp (shape, Device::CPU);
        check_cuda(cudaMemcpy(
            tmp.cpu_data.data(),
            gpu_data.get(),
            bytes(),
            cudaMemcpyDeviceToHost
        ));
        return tmp;
    }
};

void require_device(const Tensor& tensor, Device expected) {
    if(tensor.device != expected) {
        throw invalid_argument("Tensor is on the wrong device");
    }
}

void require_same_shape_and_device(
    const Tensor& first,
    const Tensor& second
) {
    if(first.device != second.device || first.shape != second.shape) {
        throw invalid_argument("Tensor shapes and devices must match");
    }
}

void check_kernel() {
    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());
}

Tensor relu_forward_cpu(const Tensor &x) {
    require_device(x, Device::CPU);
    Tensor tmp(x.shape, Device::CPU);
    for(size_t i = 0; i < x.sz; i ++) tmp.cpu_data[i] = x.cpu_data[i] > 0 ? x.cpu_data[i] : 0;
    return tmp;
}

__global__ void relu_forward_kernel(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        output[idx] = input[idx] > 0 ? input[idx] : 0; 
    }
}

int threads_per_block = 256;

Tensor relu_forward_gpu(const Tensor &x) {
    require_device(x, Device::GPU);
    int n = x.sz;
    int cnt = (n + threads_per_block - 1) / threads_per_block;
    Tensor out(x.shape, Device::GPU);
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

__global__ void relu_backward_kernel(const float *input, const float *dout, float *dx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        dx[idx] = input[idx] > 0 ? dout[idx] : 0; 
    }
}

Tensor relu_backward_gpu(const Tensor &x, const Tensor &dout) {
    require_device(x, Device::GPU);
    require_same_shape_and_device(x, dout);
    int n = x.sz;
    int cnt = (n + threads_per_block - 1) / threads_per_block;
    Tensor dx(x.shape, Device::GPU);
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
    for(int i = 0; i < x.sz; i ++) tmp.cpu_data[i] = 1.0f / (1.0f + exp(-x.cpu_data[i]));
    return tmp;
}

__global__ void sigmoid_forward_kernel(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

Tensor sigmoid_forward_gpu(const Tensor &x) {
    require_device(x, Device::GPU);
    int n = x.sz;
    int cnt = (n + threads_per_block - 1) / threads_per_block;
    Tensor out(x.shape, Device::GPU);
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

__global__ void sigmoid_backward_kernel(const float *input, const float *dout, float *dx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        dx[idx] = dout[idx] * input[idx] * (1.0f - input[idx]);
    }
}

Tensor sigmoid_backward_gpu(const Tensor &y, const Tensor &dout) {
    require_device(y, Device::GPU);
    require_same_shape_and_device(y, dout);
    int n = y.sz;
    int cnt = (n + threads_per_block - 1) / threads_per_block;
    Tensor dx(y.shape, Device::GPU);
    sigmoid_backward_kernel<<<cnt, threads_per_block>>>(y.gpu_data.get(), dout.gpu_data.get(), dx.gpu_data.get(), n);
    check_kernel();
    return dx;
}

Tensor sigmoid_backward(const Tensor &x, const Tensor &dout) {
    if(x.device == Device::CPU)return sigmoid_backward_cpu(x, dout);
    return sigmoid_backward_gpu(x, dout);
}

int main() {
    return 0;
}
