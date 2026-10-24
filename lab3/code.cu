#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


struct DeviceArray {
    float* data = nullptr;

    explicit DeviceArray(size_t count) {
        cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(float));
    }

    ~DeviceArray() {
        cudaFree(data);
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

__global__ void add_bias(float* output, const float* bias, int total, int output_channels) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total) {
        output[index] += bias[index % output_channels];
    }
}

__global__ void sum_bias_gradient(const float* output_gradient, float* bias_gradient, int batch_size, int output_channels) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel < output_channels) {
        float sum = 0.0f;
        for (int sample = 0; sample < batch_size; ++sample) {
            sum += output_gradient[sample * output_channels + channel];
        }
        bias_gradient[channel] = sum;
    }
}

void gemm_row_major(
    cublasHandle_t handle, const float* left, const float* right, float* result,
    int rows, int cols, int shared, bool transpose_left, bool transpose_right,
    float beta = 0.0f
) {
    const float one = 1.0f;
    cublasOperation_t right_op = transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t left_op = transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N;
    int right_leading = transpose_right ? shared : cols;
    int left_leading = transpose_left ? rows : shared;
    cublasSgemm(
        handle, right_op, left_op, cols, rows, shared,
        &one, right, right_leading,
        left, left_leading,
        &beta, result, cols
    );
}

void fc_forward(
    cublasHandle_t handle, const float* input, const float* weight,
    const float* bias, float* output,
    int batch_size, int input_channels, int output_channels
) {
    gemm_row_major(
        handle, input, weight, output,
        batch_size, output_channels, input_channels, false, true
    );

    int total = batch_size * output_channels;
    add_bias<<<(total + 255) / 256, 256>>>(output, bias, total, output_channels);
}

void fc_backward(
    cublasHandle_t handle, const float* input, const float* weight,
    const float* output_gradient, float* input_gradient,
    float* weight_gradient, float* bias_gradient,
    int batch_size, int input_channels, int output_channels
) {
    gemm_row_major(
        handle, output_gradient, input, weight_gradient,
        output_channels, input_channels, batch_size, true, false
    );

    gemm_row_major(
        handle, output_gradient, weight, input_gradient,
        batch_size, input_channels, output_channels, false, false
    );

    sum_bias_gradient<<<(output_channels + 255) / 256, 256>>>(output_gradient, bias_gradient, batch_size, output_channels);
}

__global__ void im2col(const float* input, float* col, int Cin, int H, int W) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= Cin * 9 * H * W) return;
    int position = index % (H * W);
    int kernel_index = index / (H * W);
    int c = kernel_index / 9;
    int kh = (kernel_index % 9) / 3;
    int kw = kernel_index % 3;
    int h = position / W;
    int w = position % W;
    int new_h = h + kh - 1;
    int new_w = w + kw - 1;
    if(new_h < 0 || new_h >= H || new_w < 0 || new_w >= W) {
        col[index] = 0;
    }
    else {
        col[index] = input[c * H * W + new_h * W + new_w];
    }
}

__global__ void fill_ones(float* data, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) data[index] = 1.0f;
}

__global__ void col2im(const float* col, float* grad_input, int Cin, int H, int W) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= Cin * H * W) return;
    int c = index / (H * W);
    int h = (index % (H * W)) / W;
    int w = index % W;
    float sum = 0.0f;

    for(int kh = 0; kh < 3; kh ++) {
        for(int kw = 0; kw < 3; kw ++) {
            int oh = h + 1 - kh, ow = w + 1 - kw;
            if(oh < 0 || oh >= H || ow < 0 || ow >= W) continue;
            sum += col[(c * 9 + kh * 3 + kw) * H * W + oh * W + ow];
        }
    }
    grad_input[index] = sum;
}

void Convolution_forward(cublasHandle_t handle, const float* input, 
    const float* kernel, const float* bias, float* output,
    int batch_size, int Cin, int Cout, int H, int W
) {
    int spatial = H * W;
    int total = Cin * 9 * spatial;
    DeviceArray col(total);
    DeviceArray ones(spatial);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fill_ones<<<(spatial + threads - 1) / threads, threads>>>(ones.data, spatial);
    for (int i = 0; i < batch_size; ++i) {
        float* sample_output = output + i * Cout * spatial;
        im2col<<<blocks, threads>>>(input + i * Cin * spatial, col.data, Cin, H, W);
        gemm_row_major(
            handle, kernel, col.data, sample_output,
            Cout, spatial, Cin * 9, false, false
        );
        // Add bias (Cout, 1) * ones (1, H*W) to the output.
        gemm_row_major(handle, bias, ones.data, sample_output,
                       Cout, spatial, 1, false, false, 1.0f);
    }
}


void Convolution_backward(cublasHandle_t handle, const float* input,
    const float* kernel, const float* grad_output,
    float* grad_w, float* grad_b, float* grad_input,
    int batch_size, int Cin, int Cout, int H, int W
) {
    int spatial = H * W;
    int total = Cin * 9 * spatial;
    DeviceArray col(total);
    DeviceArray grad_col(total);
    DeviceArray ones(spatial);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fill_ones<<<(spatial + threads - 1) / threads, threads>>>(ones.data, spatial);

    for (int i = 0; i < batch_size; ++i) {
        const float* sample_grad_output = grad_output + i * Cout * spatial;
        float* sample_grad_input = grad_input + i * Cin * spatial;
        im2col<<<blocks, threads>>>(input + i * Cin * spatial, col.data, Cin, H, W);

        // dW += dY * col^T; overwrite for the first sample, then accumulate.
        gemm_row_major(
            handle, sample_grad_output, col.data, grad_w,
            Cout, Cin * 9, spatial, false, true, i == 0 ? 0.0f : 1.0f
        );
        // db += dY * ones.
        gemm_row_major(handle, sample_grad_output, ones.data, grad_b,
                       Cout, 1, spatial, false, false, i == 0 ? 0.0f : 1.0f);

        // dcol = W^T * dY; col2im accumulates overlapping windows into dX.
        gemm_row_major(handle, kernel, sample_grad_output, grad_col.data,
                       Cin * 9, spatial, Cout, true, false);
        col2im<<<(Cin * spatial + threads - 1) / threads, threads>>>(
            grad_col.data, sample_grad_input, Cin, H, W
        );
    }
}


__global__ void max_pool(const float* input,float* output, int* out_mask, int Cin, int H,int W) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= Cin * H * W) return;
    int c = index / (H * W);
    int h = index % (H * W) / W;
    int w = index % W;
    int position = c * (H << 1) * (W << 1) + (h << 1) * (W << 1) + (w << 1);
    output[index] = input[position];
    out_mask[index] = 0;
    for(int i = 0;i < 2;i ++) {
        for(int j = 0; j < 2;j ++) {
            if(input[position + i * (W << 1) + j] > output[index]) {
                output[index] = input[position + i * (W << 1) + j];
                out_mask[index] = (i << 1) + j;
            }
        }
    }
}

void max_pool_forward(const float* input, float* output, 
    int batch_size, int Cin, int H, int W, int* out_mask
) {
    int H_out = H >> 1;
    int W_out = W >> 1;
    int total = Cin * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    for(int i = 0; i < batch_size; i ++) {
        const float* sample_input = input + i * Cin * H * W;
        float* sample_output = output + i * Cin * H_out * W_out;
        int* sample_out_mask = out_mask + i * Cin * H_out * W_out;
        max_pool<<<blocks, threads>>>(sample_input, sample_output, sample_out_mask, Cin, H_out, W_out);
    }
}

__global__ void max_pool_back(const float* output_grad,float* input_grad, const int* out_mask, int Cin, int H,int W) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= Cin * H * W) return;
    int c = index / (H * W);
    int h = index % (H * W) / W;
    int w = index % W;
    int position = c * (H << 1) * (W << 1) + (h << 1) * (W << 1) + (w << 1);
    for(int i = 0;i < 2;i ++) {
        for(int j = 0; j < 2;j ++) {
            if((i << 1) + j == out_mask[index]) {
                input_grad[position + i * (W << 1) + j] = output_grad[index];
            }
            else {
                input_grad[position + i * (W << 1) + j] = 0;
            }
        }
    }
}

void max_pool_backward(const float* output_grad, float* input_grad, 
    int batch_size, int Cin, int H, int W, const int* out_mask 
) {
    int total = Cin * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    for(int i = 0; i < batch_size; i ++) {
        const float* sample_output_grad = output_grad + i * Cin * H * W;
        float* sample_input_grad = input_grad + i * Cin * (H << 1) * (W << 1);
        const int* sample_out_mask = out_mask + i * Cin * H * W;
        max_pool_back<<<blocks, threads>>>(sample_output_grad, sample_input_grad, sample_out_mask, Cin, H, W);
    }
}

void expect_close(
    const std::vector<float>& actual, const std::vector<float>& expected,
    const char* name
) {
    for (size_t index = 0; index < actual.size(); ++index) {
        if (std::fabs(actual[index] - expected[index]) > 1e-4f) {
            throw std::runtime_error(std::string(name) + " differs from CPU result");
        }
    }
    std::cout << name << ": PASS\n";
}

__global__ void Find_Max(const float* input, float* Max, int batch_size, int C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= batch_size) return;
    Max[index] = input[index * C];
    for(int i = 1; i < C; i ++) {
        Max[index] = fmaxf(Max[index], input[index * C + i]);
    }
}

__global__ void Find_Sum(const float* input, float* Max, float* output, float* Sum, int batch_size, int C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= batch_size) return;
    Sum[index] = 0;
    for(int i = 0; i < C; i ++) {
        output[index * C + i] = expf(input[index * C + i] - Max[index]);
        Sum[index] += output[index * C + i];
    }
}

__global__ void Normalization(float* output, float* Sum, int batch_size, int C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= batch_size) return;
    for(int i  = 0; i < C; i ++) {
        output[index * C + i] /= Sum[index];
    }
}

__global__ void Find_Label(float* output, int* label, int batch_size, int C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= batch_size) return;
    label[index] = 0;
    for(int i = 1; i < C; i ++) {
        if(output[index * C + i] > output[index * C + label[index]]) {
            label[index] = i;
        }
    }
}

void SoftMax(const float* input,float* output,int* label, int batch_size, int C) {
    DeviceArray Max(batch_size);
    DeviceArray Sum(batch_size);
    int threads = 16;
    int blocks = (batch_size + threads - 1) / threads;
    Find_Max<<<blocks, threads>>>(input, Max.data, batch_size, C);
    Find_Sum<<<blocks, threads>>>(input, Max.data, output, Sum.data, batch_size, C);
    Normalization<<<blocks, threads>>>(output, Sum.data, batch_size, C);
    Find_Label<<<blocks, threads>>>(output, label, batch_size, C);
}

__global__ void cross_entropy(const float* input, const int* label, float* loss, int batch_size, int C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size) return;
    float p = fmaxf(input[index * C + label[index]], 1e-12f);
    atomicAdd(loss, -logf(p) / batch_size);
}

void Cross_Entropy(const float* input, const int* label, float &loss, int batch_size, int C) {
    int threads = 16;
    int blocks = (batch_size + threads - 1) / threads;
    DeviceArray gpu_loss(1);
    cudaMemset(gpu_loss.data, 0, sizeof(float));
    cross_entropy<<<blocks, threads>>>(input, label, gpu_loss.data, batch_size, C);
    cudaMemcpy(&loss, gpu_loss.data, sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void softmax_backward(const float* input, float* output, const int* label, int batch_size, int C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size) return;
    for(int i = 0; i < C; i ++) {
        output[index * C + i] = (input[index * C + i] - (i == label[index])) / batch_size;
    }
}

void SoftMax_backward(const float* input, float* output, const int* label, int batch_size, int C) {
    int threads = 16;
    int blocks = (batch_size + threads - 1) / threads;
    softmax_backward<<<blocks, threads>>>(input, output, label, batch_size, C);
}

int main() {
    constexpr int batch_size = 2;
    constexpr int input_channels = 3;
    constexpr int output_channels = 2;

    const std::vector<float> input = {1.0f, -2.0f, 3.0f, 4.0f, 0.0f, -1.0f};
    const std::vector<float> bias = {0.1f, -0.2f};
    const std::vector<float> output_gradient = {1.0f, 2.0f, -3.0f, 1.0f};

    DeviceArray gpu_input(input.size());
    DeviceArray gpu_weight(output_channels * input_channels);
    DeviceArray gpu_bias(bias.size());
    DeviceArray gpu_output(batch_size * output_channels);
    DeviceArray gpu_output_gradient(output_gradient.size());
    DeviceArray gpu_input_gradient(input.size());
    DeviceArray gpu_weight_gradient(output_channels * input_channels);
    DeviceArray gpu_bias_gradient(bias.size());

    cudaMemcpy(gpu_input.data, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_bias.data, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(
        gpu_output_gradient.data, output_gradient.data(),
        output_gradient.size() * sizeof(float), cudaMemcpyHostToDevice
    );

    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 42);
    curandGenerateUniform(generator, gpu_weight.data, output_channels * input_channels);
    curandDestroyGenerator(generator);

    cublasHandle_t handle;
    cublasCreate(&handle);
    fc_forward(
        handle, gpu_input.data, gpu_weight.data, gpu_bias.data, gpu_output.data,
        batch_size, input_channels, output_channels
    );
    fc_backward(
        handle, gpu_input.data, gpu_weight.data, gpu_output_gradient.data,
        gpu_input_gradient.data, gpu_weight_gradient.data, gpu_bias_gradient.data,
        batch_size, input_channels, output_channels
    );
    cudaDeviceSynchronize();
    cublasDestroy(handle);

    std::vector<float> weight(output_channels * input_channels);
    std::vector<float> output(batch_size * output_channels);
    std::vector<float> input_gradient(input.size());
    std::vector<float> weight_gradient(weight.size());
    std::vector<float> bias_gradient(bias.size());
    cudaMemcpy(weight.data(), gpu_weight.data, weight.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output.data(), gpu_output.data, output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        input_gradient.data(), gpu_input_gradient.data,
        input_gradient.size() * sizeof(float), cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        weight_gradient.data(), gpu_weight_gradient.data,
        weight_gradient.size() * sizeof(float), cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        bias_gradient.data(), gpu_bias_gradient.data,
        bias_gradient.size() * sizeof(float), cudaMemcpyDeviceToHost
    );

    std::vector<float> expected_output(output.size());
    std::vector<float> expected_input_gradient(input.size(), 0.0f);
    std::vector<float> expected_weight_gradient(weight.size(), 0.0f);
    std::vector<float> expected_bias_gradient(bias.size(), 0.0f);
    for (int sample = 0; sample < batch_size; ++sample) {
        for (int out_channel = 0; out_channel < output_channels; ++out_channel) {
            float value = bias[out_channel];
            float gradient = output_gradient[sample * output_channels + out_channel];
            expected_bias_gradient[out_channel] += gradient;
            for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
                int input_index = sample * input_channels + in_channel;
                int weight_index = out_channel * input_channels + in_channel;
                value += input[input_index] * weight[weight_index];
                expected_input_gradient[input_index] += gradient * weight[weight_index];
                expected_weight_gradient[weight_index] += gradient * input[input_index];
            }
            expected_output[sample * output_channels + out_channel] = value;
        }
    }

    expect_close(output, expected_output, "forward");
    expect_close(input_gradient, expected_input_gradient, "input gradient");
    expect_close(weight_gradient, expected_weight_gradient, "weight gradient");
    expect_close(bias_gradient, expected_bias_gradient, "bias gradient");
    return 0;
}



