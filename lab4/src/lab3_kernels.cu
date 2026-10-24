// FC and im2col/GEMM convolution migrated from lab3/code.cu.
#include "lab3_kernels.h"
namespace lab3 {
struct DeviceArray {
    float* data = nullptr;

    explicit DeviceArray(size_t count) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(float)));
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
    check_cublas(cublasSgemm(
        handle, right_op, left_op, cols, rows, shared,
        &one, right, right_leading,
        left, left_leading,
        &beta, result, cols
    ));
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



}  // namespace lab3
