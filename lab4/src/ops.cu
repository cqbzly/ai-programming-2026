#include "ops.h"
#include "lab3_kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int threads = 256;
int blocks(size_t count) { return static_cast<int>((count + threads - 1) / threads); }
const float* data(const Tensor& x) { return x.device == Device::CPU ? x.cpu_data.data() : x.gpu_data.get(); }
float* data(Tensor& x) { return x.device == Device::CPU ? x.cpu_data.data() : x.gpu_data.get(); }

void check_rank(const Tensor& x, size_t rank, const char* name) {
    if (x.shape.size() != rank) throw std::invalid_argument(std::string(name) + " has the wrong rank");
    for (auto dimension : x.shape)
        if (dimension == 0) throw std::invalid_argument(std::string(name) + " dimensions must be positive");
    if (x.sz > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(std::string(name) + " exceeds 32-bit operator indexing");
}
void same_device(const Tensor& x, const Tensor& y) {
    if (x.device != y.device) throw std::invalid_argument("All operator inputs must be on the same device");
}
void shape_is(const Tensor& x, const std::vector<size_t>& expected) {
    if (x.shape != expected) throw std::invalid_argument("Unexpected Tensor shape");
}
void check_linear(const Tensor& x, const Tensor& w) {
    check_rank(x, 2, "x"); check_rank(w, 2, "weight"); same_device(x, w);
    if (x.shape[1] != w.shape[1]) throw std::invalid_argument("linear input/weight channels differ");
    if (x.shape[0] * w.shape[0] > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("linear output is too large");
}
void check_conv(const Tensor& x, const Tensor& w) {
    check_rank(x, 4, "x"); check_rank(w, 4, "weight"); same_device(x, w);
    if (w.shape[1] != x.shape[1] || w.shape[2] != 3 || w.shape[3] != 3)
        throw std::invalid_argument("conv2d requires weight (Cout,Cin,3,3)");
    size_t spatial = x.shape[2] * x.shape[3];
    const size_t limit = std::numeric_limits<int>::max();
    if (x.shape[0] * w.shape[0] * spatial > limit || x.shape[1] * 9 * spatial > limit)
        throw std::invalid_argument("convolution output or im2col workspace is too large");
}
void check_pool(const Tensor& x) {
    check_rank(x, 4, "x");
    if (x.shape[2] < 2 || x.shape[3] < 2) throw std::invalid_argument("pooling requires H,W >= 2");
}
void check_labels(const Tensor& x, const Tensor& labels) {
    check_rank(x, 2, "class scores"); check_rank(labels, 1, "labels"); same_device(x, labels);
    shape_is(labels, {x.shape[0]});
    std::vector<float> host(labels.sz);
    labels.copy_to_host(host.data());
    for (float label : host) {
        if (!std::isfinite(label) || label < 0 || label >= x.shape[1] || std::floor(label) != label)
            throw std::invalid_argument("labels must be finite integer class IDs in [0,C)");
    }
}
void check_probabilities(const Tensor& p) {
    std::vector<float> host(p.sz);
    p.copy_to_host(host.data());
    for (size_t n = 0; n < p.shape[0]; ++n) {
        double sum = 0;
        for (size_t c = 0; c < p.shape[1]; ++c) {
            float value = host[n * p.shape[1] + c];
            if (!std::isfinite(value) || value < 0 || value > 1)
                throw std::invalid_argument("probabilities must be finite and in [0,1]");
            sum += value;
        }
        if (std::fabs(sum - 1.0) > 1e-4) throw std::invalid_argument("probability rows must sum to 1");
    }
}

__host__ __device__ int pool_argmax(const float* x, int base, int width) {
    int selected = base;
    for (int kh = 0; kh < 2; ++kh) for (int kw = 0; kw < 2; ++kw) {
        int candidate = base + kh * width + kw;
        // Strict > gives the first maximum on ties; propagate NaNs like PyTorch.
        if (x[candidate] > x[selected] || x[candidate] != x[candidate]) selected = candidate;
    }
    return selected;
}

__global__ void pool_kernel(const float* x, const float* dout, float* output,
                            int total, int height, int width, bool backward) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    int oh = height / 2, ow = width / 2;
    int plane = index / (oh * ow);
    int h = (index / ow) % oh, w = index % ow;
    int selected = pool_argmax(x, plane * height * width + 2 * h * width + 2 * w, width);
    if (backward) output[selected] = dout[index];  // Non-overlapping 2x2 windows.
    else output[index] = x[selected];
}

__global__ void softmax_kernel(const float* x, float* y, int n, int c) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float maximum = x[row * c];
    for (int j = 1; j < c; ++j) maximum = fmaxf(maximum, x[row * c + j]);
    float sum = 0;
    for (int j = 0; j < c; ++j) { float v = expf(x[row * c + j] - maximum); y[row * c + j] = v; sum += v; }
    for (int j = 0; j < c; ++j) y[row * c + j] /= sum;
}

__global__ void loss_rows_kernel(const float* x, const float* labels, float* losses, int n, int c, bool logits) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    int label = static_cast<int>(labels[row]);
    if (!logits) { losses[row] = -logf(fmaxf(x[row * c + label], 1e-12f)); return; }
    float maximum = x[row * c];
    for (int j = 1; j < c; ++j) maximum = fmaxf(maximum, x[row * c + j]);
    float sum = 0;
    for (int j = 0; j < c; ++j) sum += expf(x[row * c + j] - maximum);
    losses[row] = logf(sum) - (x[row * c + label] - maximum);
}
__global__ void mean_kernel(const float* losses, float* result, int n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += losses[i];
    result[0] = static_cast<float>(sum / n);
}
__global__ void loss_backward_kernel(const float* probabilities, const float* labels, float* dx, int total, int n, int c) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total) dx[index] = (probabilities[index] - (index % c == static_cast<int>(labels[index / c]))) / n;
}

Tensor loss_impl(const Tensor& x, const Tensor& labels, bool logits) {
    check_labels(x, labels);
    if (!logits) check_probabilities(x);
    int n = static_cast<int>(x.shape[0]), c = static_cast<int>(x.shape[1]);
    Tensor result({}, x.device);
    if (x.device == Device::GPU) {
        Tensor rows({x.shape[0]}, x.device);
        loss_rows_kernel<<<blocks(n), threads>>>(data(x), data(labels), data(rows), n, c, logits);
        mean_kernel<<<1, 1>>>(data(rows), data(result), n);
        check_kernel();
    } else {
        double total = 0;
        for (int i = 0; i < n; ++i) {
            int label = static_cast<int>(labels.cpu_data[i]);
            if (!logits) { total -= std::log(std::max(x.cpu_data[i * c + label], 1e-12f)); continue; }
            float maximum = *std::max_element(data(x) + i * c, data(x) + (i + 1) * c);
            float sum = 0;
            for (int j = 0; j < c; ++j) sum += std::exp(x.cpu_data[i * c + j] - maximum);
            total += std::log(sum) - (x.cpu_data[i * c + label] - maximum);
        }
        result.cpu_data[0] = static_cast<float>(total / n);
    }
    return result;
}
}  // namespace

Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b) {
    check_linear(x, w); check_rank(b, 1, "bias"); same_device(x, b); shape_is(b, {w.shape[0]});
    int n = static_cast<int>(x.shape[0]), in = static_cast<int>(x.shape[1]), out = static_cast<int>(w.shape[0]);
    Tensor y({x.shape[0], w.shape[0]}, x.device);
    if (x.device == Device::GPU) {
        lab3::BlasHandle handle;
        lab3::fc_forward(handle.value, data(x), data(w), data(b), data(y), n, in, out);
        check_kernel();
    } else {
        for (int i = 0; i < n; ++i) for (int o = 0; o < out; ++o) {
            float sum = b.cpu_data[o];
            for (int k = 0; k < in; ++k) sum += x.cpu_data[i * in + k] * w.cpu_data[o * in + k];
            y.cpu_data[i * out + o] = sum;
        }
    }
    return y;
}

Gradients linear_backward(const Tensor& x, const Tensor& w, const Tensor& dout) {
    check_linear(x, w); check_rank(dout, 2, "dout"); same_device(x, dout); shape_is(dout, {x.shape[0], w.shape[0]});
    int n = static_cast<int>(x.shape[0]), in = static_cast<int>(x.shape[1]), out = static_cast<int>(w.shape[0]);
    Tensor dx(x.shape, x.device), dw(w.shape, x.device), db({w.shape[0]}, x.device);
    if (x.device == Device::GPU) {
        lab3::BlasHandle handle;
        lab3::fc_backward(handle.value, data(x), data(w), data(dout), data(dx), data(dw), data(db), n, in, out);
        check_kernel();
    } else {
        for (int i = 0; i < n; ++i) for (int o = 0; o < out; ++o) {
            float g = dout.cpu_data[i * out + o]; db.cpu_data[o] += g;
            for (int k = 0; k < in; ++k) {
                dx.cpu_data[i * in + k] += g * w.cpu_data[o * in + k];
                dw.cpu_data[o * in + k] += g * x.cpu_data[i * in + k];
            }
        }
    }
    return {std::move(dx), std::move(dw), std::move(db)};
}

Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& b) {
    check_conv(x, w); check_rank(b, 1, "bias"); same_device(x, b); shape_is(b, {w.shape[0]});
    int n = static_cast<int>(x.shape[0]), in = static_cast<int>(x.shape[1]), out = static_cast<int>(w.shape[0]);
    int h = static_cast<int>(x.shape[2]), width = static_cast<int>(x.shape[3]), spatial = h * width;
    Tensor y({x.shape[0], w.shape[0], x.shape[2], x.shape[3]}, x.device);
    if (x.device == Device::GPU) {
        lab3::BlasHandle handle;
        lab3::Convolution_forward(handle.value, data(x), data(w), data(b), data(y), n, in, out, h, width);
        check_kernel();
    } else {
        for (int i = 0; i < n; ++i) for (int o = 0; o < out; ++o)
        for (int row = 0; row < h; ++row) for (int col = 0; col < width; ++col) {
            float sum = b.cpu_data[o];
            for (int c = 0; c < in; ++c) for (int kh = 0; kh < 3; ++kh) for (int kw = 0; kw < 3; ++kw) {
                int ih = row + kh - 1, iw = col + kw - 1;
                if (ih >= 0 && ih < h && iw >= 0 && iw < width)
                    sum += x.cpu_data[(i * in + c) * spatial + ih * width + iw] * w.cpu_data[(o * in + c) * 9 + kh * 3 + kw];
            }
            y.cpu_data[(i * out + o) * spatial + row * width + col] = sum;
        }
    }
    return y;
}

Gradients conv2d_backward(const Tensor& x, const Tensor& w, const Tensor& dout) {
    check_conv(x, w); check_rank(dout, 4, "dout"); same_device(x, dout);
    shape_is(dout, {x.shape[0], w.shape[0], x.shape[2], x.shape[3]});
    int n = static_cast<int>(x.shape[0]), in = static_cast<int>(x.shape[1]), out = static_cast<int>(w.shape[0]);
    int h = static_cast<int>(x.shape[2]), width = static_cast<int>(x.shape[3]), spatial = h * width;
    Tensor dx(x.shape, x.device), dw(w.shape, x.device), db({w.shape[0]}, x.device);
    if (x.device == Device::GPU) {
        lab3::BlasHandle handle;
        lab3::Convolution_backward(handle.value, data(x), data(w), data(dout), data(dw), data(db), data(dx), n, in, out, h, width);
        check_kernel();
    } else {
        for (int i = 0; i < n; ++i) for (int o = 0; o < out; ++o)
        for (int row = 0; row < h; ++row) for (int col = 0; col < width; ++col) {
            float g = dout.cpu_data[(i * out + o) * spatial + row * width + col]; db.cpu_data[o] += g;
            for (int c = 0; c < in; ++c) for (int kh = 0; kh < 3; ++kh) for (int kw = 0; kw < 3; ++kw) {
                int ih = row + kh - 1, iw = col + kw - 1;
                if (ih < 0 || ih >= h || iw < 0 || iw >= width) continue;
                int xi = (i * in + c) * spatial + ih * width + iw, wi = (o * in + c) * 9 + kh * 3 + kw;
                dx.cpu_data[xi] += g * w.cpu_data[wi]; dw.cpu_data[wi] += g * x.cpu_data[xi];
            }
        }
    }
    return {std::move(dx), std::move(dw), std::move(db)};
}

Tensor max_pool2d(const Tensor& x) {
    check_pool(x);
    int h = static_cast<int>(x.shape[2]), w = static_cast<int>(x.shape[3]), oh = h / 2, ow = w / 2;
    Tensor y({x.shape[0], x.shape[1], static_cast<size_t>(oh), static_cast<size_t>(ow)}, x.device);
    if (x.device == Device::GPU) {
        pool_kernel<<<blocks(y.sz), threads>>>(data(x), nullptr, data(y), static_cast<int>(y.sz), h, w, false); check_kernel();
    } else for (int i = 0; i < static_cast<int>(y.sz); ++i) {
        int base = (i / (oh * ow)) * h * w + 2 * ((i / ow) % oh) * w + 2 * (i % ow);
        y.cpu_data[i] = x.cpu_data[pool_argmax(data(x), base, w)];
    }
    return y;
}

Tensor max_pool2d_backward(const Tensor& x, const Tensor& dout) {
    check_pool(x); check_rank(dout, 4, "dout"); same_device(x, dout);
    shape_is(dout, {x.shape[0], x.shape[1], x.shape[2] / 2, x.shape[3] / 2});
    int h = static_cast<int>(x.shape[2]), w = static_cast<int>(x.shape[3]), oh = h / 2, ow = w / 2;
    Tensor dx(x.shape, x.device);
    if (x.device == Device::GPU) {
        pool_kernel<<<blocks(dout.sz), threads>>>(data(x), data(dout), data(dx), static_cast<int>(dout.sz), h, w, true); check_kernel();
    } else for (int i = 0; i < static_cast<int>(dout.sz); ++i) {
        int base = (i / (oh * ow)) * h * w + 2 * ((i / ow) % oh) * w + 2 * (i % ow);
        dx.cpu_data[pool_argmax(data(x), base, w)] = dout.cpu_data[i];
    }
    return dx;
}

Tensor softmax(const Tensor& x) {
    check_rank(x, 2, "x"); Tensor y(x.shape, x.device);
    int n = static_cast<int>(x.shape[0]), c = static_cast<int>(x.shape[1]);
    if (x.device == Device::GPU) {
        softmax_kernel<<<blocks(n), threads>>>(data(x), data(y), n, c); check_kernel();
    } else for (int i = 0; i < n; ++i) {
        float maximum = *std::max_element(data(x) + i * c, data(x) + (i + 1) * c), sum = 0;
        for (int j = 0; j < c; ++j) { float v = std::exp(x.cpu_data[i * c + j] - maximum); y.cpu_data[i * c + j] = v; sum += v; }
        for (int j = 0; j < c; ++j) y.cpu_data[i * c + j] /= sum;
    }
    return y;
}

Tensor cross_entropy(const Tensor& logits, const Tensor& labels) { return loss_impl(logits, labels, true); }
Tensor cross_entropy_from_probs(const Tensor& p, const Tensor& labels) { return loss_impl(p, labels, false); }

Tensor softmax_cross_entropy_backward(const Tensor& p, const Tensor& labels) {
    check_labels(p, labels); check_probabilities(p);
    int n = static_cast<int>(p.shape[0]), c = static_cast<int>(p.shape[1]);
    Tensor dx(p.shape, p.device);
    if (p.device == Device::GPU) {
        loss_backward_kernel<<<blocks(p.sz), threads>>>(data(p), data(labels), data(dx), static_cast<int>(p.sz), n, c); check_kernel();
    } else for (int i = 0; i < n; ++i) for (int j = 0; j < c; ++j)
        dx.cpu_data[i * c + j] = (p.cpu_data[i * c + j] - (j == static_cast<int>(labels.cpu_data[i]))) / n;
    return dx;
}
Tensor cross_entropy_backward(const Tensor& logits, const Tensor& labels) {
    check_labels(logits, labels);
    return softmax_cross_entropy_backward(softmax(logits), labels);
}
