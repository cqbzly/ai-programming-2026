#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define main code_cu_empty_main
#include "code.cu"
#undef main

void require_true(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    float actual,
    float expected,
    const std::string& name,
    float tolerance = 1e-4f
) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            name + ": expected " + std::to_string(expected)
            + ", got " + std::to_string(actual)
        );
    }
}

void require_vector_close(
    const Tensor& tensor,
    const std::vector<float>& expected,
    const std::string& name
) {
    require_true(tensor.device == Device::CPU, name + " must be on CPU");
    require_true(tensor.cpu_data.size() == expected.size(), name + " size mismatch");

    for (std::size_t i = 0; i < expected.size(); ++i) {
        require_close(tensor.cpu_data[i], expected[i], name + "[" + std::to_string(i) + "]");
    }
}

int main() {
    const std::vector<float> input_values = {
        -2.0f, -0.5f, 0.0f, 0.5f, 2.0f
    };
    const std::vector<float> upstream_gradient = {
        0.5f, -1.0f, 2.0f, 3.0f, -0.25f
    };

    Tensor input({5}, Device::CPU);
    Tensor dout({5}, Device::CPU);
    input.cpu_data = input_values;
    dout.cpu_data = upstream_gradient;

    require_true(input.shape == std::vector<std::size_t>{5}, "Tensor shape mismatch");
    require_true(input.sz == 5, "Tensor element count mismatch");
    require_true(input.bytes() == 5 * sizeof(float), "Tensor byte count mismatch");

    Tensor input_round_trip = input.gpu().cpu();
    require_vector_close(input_round_trip, input_values, "CPU-GPU-CPU round trip");

    const std::vector<float> relu_expected = {
        0.0f, 0.0f, 0.0f, 0.5f, 2.0f
    };
    const std::vector<float> relu_backward_expected = {
        0.0f, 0.0f, 0.0f, 3.0f, -0.25f
    };

    Tensor relu_cpu = relu_forward(input);
    Tensor relu_backward_cpu_result = relu_backward(input, dout);
    require_vector_close(relu_cpu, relu_expected, "CPU ReLU forward");
    require_vector_close(
        relu_backward_cpu_result,
        relu_backward_expected,
        "CPU ReLU backward"
    );

    std::vector<float> sigmoid_expected;
    std::vector<float> sigmoid_backward_expected;
    for (std::size_t i = 0; i < input_values.size(); ++i) {
        float value = 1.0f / (1.0f + std::exp(-input_values[i]));
        sigmoid_expected.push_back(value);
        sigmoid_backward_expected.push_back(
            upstream_gradient[i] * value * (1.0f - value)
        );
    }

    Tensor sigmoid_cpu = sigmoid_forward(input);
    Tensor sigmoid_backward_cpu_result = sigmoid_backward(sigmoid_cpu, dout);
    require_vector_close(sigmoid_cpu, sigmoid_expected, "CPU Sigmoid forward");
    require_vector_close(
        sigmoid_backward_cpu_result,
        sigmoid_backward_expected,
        "CPU Sigmoid backward"
    );

    Tensor gpu_input = input.gpu();
    Tensor gpu_dout = dout.gpu();

    Tensor relu_gpu = relu_forward(gpu_input).cpu();
    Tensor relu_backward_gpu_result = relu_backward(gpu_input, gpu_dout).cpu();
    require_vector_close(relu_gpu, relu_expected, "GPU ReLU forward");
    require_vector_close(
        relu_backward_gpu_result,
        relu_backward_expected,
        "GPU ReLU backward"
    );

    Tensor sigmoid_gpu_tensor = sigmoid_forward(gpu_input);
    Tensor sigmoid_gpu = sigmoid_gpu_tensor.cpu();
    Tensor sigmoid_backward_gpu_result =
        sigmoid_backward(sigmoid_gpu_tensor, gpu_dout).cpu();
    require_vector_close(sigmoid_gpu, sigmoid_expected, "GPU Sigmoid forward");
    require_vector_close(
        sigmoid_backward_gpu_result,
        sigmoid_backward_expected,
        "GPU Sigmoid backward"
    );

    std::cout << "Tensor construction: PASS\n";
    std::cout << "CPU-GPU round trip: PASS\n";
    std::cout << "ReLU CPU/GPU forward and backward: PASS\n";
    std::cout << "Sigmoid CPU/GPU forward and backward: PASS\n";
    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
