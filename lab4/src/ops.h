#pragma once
#include "tensor.h"
#include <tuple>

using Gradients = std::tuple<Tensor, Tensor, Tensor>;  // dx, dw, db
Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias);
Gradients linear_backward(const Tensor& x, const Tensor& weight, const Tensor& dout);
Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor& bias);
Gradients conv2d_backward(const Tensor& x, const Tensor& weight, const Tensor& dout);
Tensor max_pool2d(const Tensor& x);
Tensor max_pool2d_backward(const Tensor& x, const Tensor& dout);
Tensor softmax(const Tensor& x);
Tensor cross_entropy(const Tensor& logits, const Tensor& labels);
Tensor cross_entropy_backward(const Tensor& logits, const Tensor& labels);
Tensor cross_entropy_from_probs(const Tensor& probabilities, const Tensor& labels);
// Lab3 fused gradient wrt logits, NOT the derivative wrt probabilities.
Tensor softmax_cross_entropy_backward(const Tensor& probabilities, const Tensor& labels);
