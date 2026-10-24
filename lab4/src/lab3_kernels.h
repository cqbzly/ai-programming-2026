#pragma once
#include "tensor.h"
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

namespace lab3 {
inline void check_cublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error code " + std::to_string(static_cast<int>(status)));
}

struct BlasHandle {
    cublasHandle_t value = nullptr;
    BlasHandle() { check_cublas(cublasCreate(&value)); }
    ~BlasHandle() { if (value) cublasDestroy(value); }
    BlasHandle(const BlasHandle&) = delete;
    BlasHandle& operator=(const BlasHandle&) = delete;
};

void fc_forward(cublasHandle_t, const float*, const float*, const float*, float*, int, int, int);
void fc_backward(cublasHandle_t, const float*, const float*, const float*, float*, float*, float*, int, int, int);
void Convolution_forward(cublasHandle_t, const float*, const float*, const float*, float*, int, int, int, int, int);
void Convolution_backward(cublasHandle_t, const float*, const float*, const float*, float*, float*, float*, int, int, int, int, int);
}  // namespace lab3
