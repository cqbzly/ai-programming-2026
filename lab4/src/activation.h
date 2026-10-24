#pragma once
#include "tensor.h"

Tensor relu_forward(const Tensor& x);
Tensor relu_backward(const Tensor& x, const Tensor& dout);
Tensor sigmoid_forward(const Tensor& x);
// As in Lab2, sigmoid_backward takes the FORWARD OUTPUT y, not input x.
Tensor sigmoid_backward(const Tensor& y, const Tensor& dout);
