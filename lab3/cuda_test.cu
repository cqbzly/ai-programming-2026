#define main fc_demo_main
#include "code.cu"
#undef main

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void check_curand(curandStatus_t status) {
    if (status != CURAND_STATUS_SUCCESS) throw std::runtime_error("cuRAND call failed");
}

void check_equal(const std::vector<int>& actual, const std::vector<int>& expected, const char* name) {
    if (actual != expected) throw std::runtime_error(std::string(name) + " differs from CPU result");
    std::cout << name << ": PASS\n";
}

void test_convolution(int N, int Cin, int Cout, int H, int W) {
    const int S = H * W;
    std::vector<float> x(N * Cin * S), weight(Cout * Cin * 9), bias(Cout), dy(N * Cout * S);
    for (int i = 0; i < (int)x.size(); ++i) x[i] = ((i * 7) % 17 - 8) * 0.125f;
    for (int i = 0; i < Cout; ++i) bias[i] = (i - 1) * 0.25f;
    for (int i = 0; i < (int)dy.size(); ++i) dy[i] = ((i * 3) % 11 - 5) * 0.125f;

    DeviceArray gx(x.size()), gw(weight.size()), gb(bias.size()), gy(dy.size());
    DeviceArray gdy(dy.size()), gdx(x.size()), gdw(weight.size()), gdb(bias.size());
    curandGenerator_t generator;
    check_curand(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    check_curand(curandSetPseudoRandomGeneratorSeed(generator, 42));
    check_curand(curandGenerateUniform(generator, gw.data, weight.size()));
    check_curand(curandDestroyGenerator(generator));
    // Use the exact same cuRAND weights for the independent CPU reference.
    check_cuda(cudaMemcpy(weight.data(), gw.data, weight.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected_y(dy.size(), 0), expected_dx(x.size(), 0);
    std::vector<float> expected_dw(weight.size(), 0), expected_db(Cout, 0);
    for (int n = 0; n < N; ++n) for (int co = 0; co < Cout; ++co)
    for (int h = 0; h < H; ++h) for (int w = 0; w < W; ++w) {
        int y_index = (n * Cout + co) * S + h * W + w;
        expected_y[y_index] = bias[co];
        expected_db[co] += dy[y_index];
        for (int ci = 0; ci < Cin; ++ci)
        for (int kh = 0; kh < 3; ++kh) for (int kw = 0; kw < 3; ++kw) {
            int ih = h + kh - 1, iw = w + kw - 1;
            if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
            int x_index = (n * Cin + ci) * S + ih * W + iw;
            int w_index = (co * Cin + ci) * 9 + kh * 3 + kw;
            expected_y[y_index] += x[x_index] * weight[w_index];
            expected_dx[x_index] += dy[y_index] * weight[w_index];
            expected_dw[w_index] += dy[y_index] * x[x_index];
        }
    }

    check_cuda(cudaMemcpy(gx.data, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(gb.data, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(gdy.data, dy.data(), dy.size() * sizeof(float), cudaMemcpyHostToDevice));
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
    Convolution_forward(handle, gx.data, gw.data, gb.data, gy.data, N, Cin, Cout, H, W);
    Convolution_backward(handle, gx.data, gw.data, gdy.data, gdw.data, gdb.data, gdx.data, N, Cin, Cout, H, W);
    check_cuda(cudaDeviceSynchronize());
    cublasDestroy(handle);
    std::vector<float> actual_y(expected_y.size()), actual_dx(expected_dx.size());
    std::vector<float> actual_dw(expected_dw.size()), actual_db(expected_db.size());
    check_cuda(cudaMemcpy(actual_y.data(), gy.data, actual_y.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(actual_dx.data(), gdx.data, actual_dx.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(actual_dw.data(), gdw.data, actual_dw.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(actual_db.data(), gdb.data, actual_db.size() * sizeof(float), cudaMemcpyDeviceToHost));
    expect_close(actual_y, expected_y, "convolution forward");
    expect_close(actual_dx, expected_dx, "convolution input gradient");
    expect_close(actual_dw, expected_dw, "convolution weight gradient");
    expect_close(actual_db, expected_db, "convolution bias gradient");
}

void test_pool() {
    const int N = 2, C = 2, H = 4, W = 6, OH = H / 2, OW = W / 2;
    std::vector<float> x(N * C * H * W), dy(N * C * OH * OW);
    for (int i = 0; i < (int)x.size(); ++i) x[i] = ((i * 13) % 29 - 14) * 0.25f;
    for (int i = 0; i < (int)dy.size(); ++i) dy[i] = (i - 7) * 0.125f;
    std::vector<float> expected_y(dy.size()), expected_dx(x.size(), 0);
    std::vector<int> expected_mask(dy.size());
    for (int n = 0; n < N; ++n) for (int c = 0; c < C; ++c)
    for (int h = 0; h < OH; ++h) for (int w = 0; w < OW; ++w) {
        int oi = ((n * C + c) * OH + h) * OW + w;
        int ii = ((n * C + c) * H + 2 * h) * W + 2 * w;
        int winner = 0;
        for (int kh = 0; kh < 2; ++kh) for (int kw = 0; kw < 2; ++kw) {
            int candidate = kh * 2 + kw;
            if (x[ii + kh * W + kw] > x[ii + (winner / 2) * W + winner % 2]) winner = candidate;
        }
        expected_y[oi] = x[ii + (winner / 2) * W + winner % 2];
        expected_mask[oi] = winner;
        expected_dx[ii + (winner / 2) * W + winner % 2] = dy[oi];
    }
    DeviceArray gx(x.size()), gy(dy.size()), gdy(dy.size()), gdx(x.size());
    int* gmask = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&gmask), dy.size() * sizeof(int)));
    check_cuda(cudaMemcpy(gx.data, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(gdy.data, dy.data(), dy.size() * sizeof(float), cudaMemcpyHostToDevice));
    max_pool_forward(gx.data, gy.data, N, C, H, W, gmask);
    max_pool_backward(gdy.data, gdx.data, N, C, OH, OW, gmask);
    check_cuda(cudaDeviceSynchronize());
    std::vector<float> actual_y(dy.size()), actual_dx(x.size());
    std::vector<int> actual_mask(dy.size());
    check_cuda(cudaMemcpy(actual_y.data(), gy.data, actual_y.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(actual_dx.data(), gdx.data, actual_dx.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(actual_mask.data(), gmask, actual_mask.size() * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(gmask);
    expect_close(actual_y, expected_y, "pool forward");
    expect_close(actual_dx, expected_dx, "pool backward");
    check_equal(actual_mask, expected_mask, "pool mask");
}

void test_softmax_loss() {
    const int N = 3, C = 5;
    const std::vector<float> z = {100, 99, 98, 97, 96, -100, -99, -98, -97, -96, 0, 1, 2, 3, 4};
    const std::vector<int> truth = {2, 1, 3};
    std::vector<float> expected_p(N * C), expected_grad(N * C);
    std::vector<int> expected_prediction(N);
    float expected_loss = 0;
    for (int n = 0; n < N; ++n) {
        float max_value = z[n * C];
        for (int c = 1; c < C; ++c) max_value = std::max(max_value, z[n * C + c]);
        float sum = 0;
        for (int c = 0; c < C; ++c) sum += std::exp(z[n * C + c] - max_value);
        for (int c = 0; c < C; ++c) {
            expected_p[n * C + c] = std::exp(z[n * C + c] - max_value) / sum;
            expected_grad[n * C + c] = (expected_p[n * C + c] - (c == truth[n])) / N;
        }
        expected_prediction[n] = (n == 1) ? 4 : (n == 2 ? 4 : 0);
        expected_loss -= std::log(expected_p[n * C + truth[n]]) / N;
    }
    DeviceArray gz(z.size()), gp(z.size()), gg(z.size());
    int *gtruth = nullptr, *gprediction = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&gtruth), N * sizeof(int)));
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&gprediction), N * sizeof(int)));
    check_cuda(cudaMemcpy(gz.data, z.data(), z.size() * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(gtruth, truth.data(), truth.size() * sizeof(int), cudaMemcpyHostToDevice));
    SoftMax(gz.data, gp.data, gprediction, N, C);
    float actual_loss = 0;
    Cross_Entropy(gp.data, gtruth, actual_loss, N, C);
    SoftMax_backward(gp.data, gg.data, gtruth, N, C);
    check_cuda(cudaDeviceSynchronize());
    std::vector<float> actual_p(z.size()), actual_grad(z.size());
    std::vector<int> actual_prediction(N);
    check_cuda(cudaMemcpy(actual_p.data(), gp.data, actual_p.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(actual_grad.data(), gg.data, actual_grad.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(actual_prediction.data(), gprediction, N * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(gtruth); cudaFree(gprediction);
    expect_close(actual_p, expected_p, "softmax probability");
    check_equal(actual_prediction, expected_prediction, "softmax predicted label");
    if (std::fabs(actual_loss - expected_loss) > 1e-4f) throw std::runtime_error("cross entropy loss differs from CPU result");
    std::cout << "cross entropy loss: PASS\n";
    expect_close(actual_grad, expected_grad, "cross entropy with softmax gradient");
}

int main() {
    fc_demo_main();
    test_convolution(2, 2, 3, 3, 4);
    test_convolution(2, 1, 2, 1, 1);
    test_pool();
    test_softmax_loss();
    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
