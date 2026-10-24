// Large-input CPU/GPU comparison. Compile separately from cuda_test.cu.
#define main code_example_main
#include "code.cu"
#undef main

#include <algorithm>
#include <chrono>
#include <iomanip>

static void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

static void copy_to_gpu(float* destination, const std::vector<float>& source) {
    cuda_check(cudaMemcpy(destination, source.data(), source.size() * sizeof(float), cudaMemcpyHostToDevice));
}

static std::vector<float> copy_from_gpu(const float* source, size_t count) {
    std::vector<float> result(count);
    cuda_check(cudaMemcpy(result.data(), source, count * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

static void compare(const std::vector<float>& gpu, const std::vector<float>& cpu, const char* name) {
    if (gpu.size() != cpu.size()) throw std::runtime_error("size mismatch");
    float maximum_error = 0.0f;
    for (size_t i = 0; i < gpu.size(); ++i) {
        float error = std::fabs(gpu[i] - cpu[i]);
        maximum_error = std::max(maximum_error, error);
        if (!std::isfinite(gpu[i]) || error > 2e-3f + 1e-3f * std::fabs(cpu[i])) {
            throw std::runtime_error(std::string(name) + " differs from CPU at index " + std::to_string(i));
        }
    }
    std::cout << name << " max_abs_error=" << maximum_error << '\n';
}

template <typename Function>
static double time_cpu(Function run, int repetitions = 5) {
    run(); // warm-up
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; ++i) run();
    auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() / repetitions;
}

template <typename Function>
static double time_gpu(Function run, int repetitions = 5) {
    run(); // warm-up
    cuda_check(cudaDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; ++i) {
        run();
        cuda_check(cudaDeviceSynchronize());
    }
    auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() / repetitions;
}

static void show(const char* name, double cpu_ms, double gpu_ms) {
    std::cout << std::left << std::setw(22) << name << std::right
              << " CPU " << std::setw(9) << cpu_ms << " ms"
              << "  GPU " << std::setw(9) << gpu_ms << " ms"
              << "  speedup " << cpu_ms / gpu_ms << "x\n";
}

static void benchmark_fc(cublasHandle_t handle) {
    const int N = 512, Cin = 512, Cout = 256;
    std::vector<float> x(N * Cin), w(Cout * Cin), b(Cout), dy(N * Cout);
    for (size_t i = 0; i < x.size(); ++i) x[i] = ((int)(i % 17) - 8) * 0.003f;
    for (size_t i = 0; i < w.size(); ++i) w[i] = ((int)(i % 19) - 9) * 0.004f;
    for (int i = 0; i < Cout; ++i) b[i] = i * 0.001f;
    for (size_t i = 0; i < dy.size(); ++i) dy[i] = ((int)(i % 13) - 6) * 0.003f;
    std::vector<float> y(N * Cout), dx(x.size()), dw(w.size()), db(b.size());
    DeviceArray gx(x.size()), gw(w.size()), gb(b.size()), gdy(dy.size());
    DeviceArray gy(y.size()), gdx(dx.size()), gdw(dw.size()), gdb(db.size());
    copy_to_gpu(gx.data, x); copy_to_gpu(gw.data, w);
    copy_to_gpu(gb.data, b); copy_to_gpu(gdy.data, dy);
    auto cpu = [&] {
        std::fill(y.begin(), y.end(), 0.0f); std::fill(dx.begin(), dx.end(), 0.0f);
        std::fill(dw.begin(), dw.end(), 0.0f); std::fill(db.begin(), db.end(), 0.0f);
        for (int n = 0; n < N; ++n) for (int o = 0; o < Cout; ++o) {
            float output = b[o], gradient = dy[n * Cout + o];
            db[o] += gradient;
            for (int c = 0; c < Cin; ++c) {
                output += x[n * Cin + c] * w[o * Cin + c];
                dx[n * Cin + c] += gradient * w[o * Cin + c];
                dw[o * Cin + c] += gradient * x[n * Cin + c];
            }
            y[n * Cout + o] = output;
        }
    };
    auto gpu = [&] {
        fc_forward(handle, gx.data, gw.data, gb.data, gy.data, N, Cin, Cout);
        fc_backward(handle, gx.data, gw.data, gdy.data, gdx.data, gdw.data, gdb.data, N, Cin, Cout);
    };
    double cpu_ms = time_cpu(cpu), gpu_ms = time_gpu(gpu);
    compare(copy_from_gpu(gy.data, y.size()), y, "FC forward");
    compare(copy_from_gpu(gdx.data, dx.size()), dx, "FC dX");
    compare(copy_from_gpu(gdw.data, dw.size()), dw, "FC dW");
    compare(copy_from_gpu(gdb.data, db.size()), db, "FC db");
    std::cout << "FC shape: N=512 Cin=512 Cout=256\n";
    show("FC forward+backward", cpu_ms, gpu_ms);
}

static void benchmark_convolution(cublasHandle_t handle) {
    const int N = 8, Cin = 16, Cout = 32, H = 32, W = 32, S = H * W;
    std::vector<float> x(N * Cin * S), w(Cout * Cin * 9), b(Cout), dy(N * Cout * S);
    for (size_t i = 0; i < x.size(); ++i) x[i] = ((int)(i % 17) - 8) * 0.003f;
    for (size_t i = 0; i < w.size(); ++i) w[i] = ((int)(i % 19) - 9) * 0.004f;
    for (int i = 0; i < Cout; ++i) b[i] = i * 0.001f;
    for (size_t i = 0; i < dy.size(); ++i) dy[i] = ((int)(i % 13) - 6) * 0.003f;
    std::vector<float> y(dy.size()), dx(x.size()), dw(w.size()), db(b.size());
    DeviceArray gx(x.size()), gw(w.size()), gb(b.size()), gdy(dy.size());
    DeviceArray gy(y.size()), gdx(dx.size()), gdw(dw.size()), gdb(db.size());
    copy_to_gpu(gx.data, x); copy_to_gpu(gw.data, w);
    copy_to_gpu(gb.data, b); copy_to_gpu(gdy.data, dy);
    auto cpu = [&] {
        std::fill(y.begin(), y.end(), 0.0f); std::fill(dx.begin(), dx.end(), 0.0f);
        std::fill(dw.begin(), dw.end(), 0.0f); std::fill(db.begin(), db.end(), 0.0f);
        for (int n = 0; n < N; ++n) for (int o = 0; o < Cout; ++o)
        for (int h = 0; h < H; ++h) for (int v = 0; v < W; ++v) {
            int yi = (n * Cout + o) * S + h * W + v;
            y[yi] = b[o]; db[o] += dy[yi];
            for (int c = 0; c < Cin; ++c)
            for (int kh = 0; kh < 3; ++kh) for (int kw = 0; kw < 3; ++kw) {
                int ih = h + kh - 1, iw = v + kw - 1;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                int xi = (n * Cin + c) * S + ih * W + iw;
                int wi = (o * Cin + c) * 9 + kh * 3 + kw;
                y[yi] += x[xi] * w[wi];
                dx[xi] += dy[yi] * w[wi];
                dw[wi] += dy[yi] * x[xi];
            }
        }
    };
    auto gpu = [&] {
        Convolution_forward(handle, gx.data, gw.data, gb.data, gy.data, N, Cin, Cout, H, W);
        Convolution_backward(handle, gx.data, gw.data, gdy.data, gdw.data, gdb.data, gdx.data, N, Cin, Cout, H, W);
    };
    double cpu_ms = time_cpu(cpu), gpu_ms = time_gpu(gpu);
    compare(copy_from_gpu(gy.data, y.size()), y, "Conv forward");
    compare(copy_from_gpu(gdx.data, dx.size()), dx, "Conv dX");
    compare(copy_from_gpu(gdw.data, dw.size()), dw, "Conv dW");
    compare(copy_from_gpu(gdb.data, db.size()), db, "Conv db");
    std::cout << "Conv shape: N=8 Cin=16 Cout=32 H=W=32\n";
    show("Conv forward+backward", cpu_ms, gpu_ms);
}

static void benchmark_pool() {
    const int N = 64, C = 64, H = 64, W = 64, OH = H / 2, OW = W / 2;
    const size_t input_count = (size_t)N * C * H * W, output_count = (size_t)N * C * OH * OW;
    std::vector<float> x(input_count), dy(output_count), y(output_count), dx(input_count);
    for (size_t i = 0; i < x.size(); ++i) x[i] = ((int)(i % 97) - 48) * 0.01f;
    for (size_t i = 0; i < dy.size(); ++i) dy[i] = ((int)(i % 23) - 11) * 0.01f;
    DeviceArray gx(x.size()), gdy(dy.size()), gy(y.size()), gdx(dx.size());
    int* mask = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&mask), output_count * sizeof(int)));
    copy_to_gpu(gx.data, x); copy_to_gpu(gdy.data, dy);
    auto cpu = [&] {
        std::fill(dx.begin(), dx.end(), 0.0f);
        for (int n = 0; n < N; ++n) for (int c = 0; c < C; ++c)
        for (int h = 0; h < OH; ++h) for (int w = 0; w < OW; ++w) {
            int oi = ((n * C + c) * OH + h) * OW + w;
            int ii = ((n * C + c) * H + 2 * h) * W + 2 * w;
            int best = 0;
            for (int kh = 0; kh < 2; ++kh) for (int kw = 0; kw < 2; ++kw) {
                int k = kh * 2 + kw;
                if (x[ii + kh * W + kw] > x[ii + (best / 2) * W + best % 2]) best = k;
            }
            int selected = ii + (best / 2) * W + best % 2;
            y[oi] = x[selected]; dx[selected] = dy[oi];
        }
    };
    auto gpu = [&] {
        max_pool_forward(gx.data, gy.data, N, C, H, W, mask);
        max_pool_backward(gdy.data, gdx.data, N, C, OH, OW, mask);
    };
    double cpu_ms = time_cpu(cpu), gpu_ms = time_gpu(gpu);
    compare(copy_from_gpu(gy.data, y.size()), y, "Pool forward");
    compare(copy_from_gpu(gdx.data, dx.size()), dx, "Pool backward");
    cuda_check(cudaFree(mask));
    std::cout << "Pool shape: N=64 C=64 H=W=64\n";
    show("Pool forward+backward", cpu_ms, gpu_ms);
}

static void benchmark_softmax_loss() {
    const int N = 32768, C = 100;
    std::vector<float> z((size_t)N * C), p(z.size()), dz(z.size());
    std::vector<int> truth(N), prediction(N);
    for (size_t i = 0; i < z.size(); ++i) z[i] = ((int)(i % 53) - 26) * 0.05f;
    for (int n = 0; n < N; ++n) truth[n] = (n * 7) % C;
    DeviceArray gz(z.size()), gp(p.size()), gdz(dz.size());
    int *gtruth = nullptr, *gprediction = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&gtruth), N * sizeof(int)));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&gprediction), N * sizeof(int)));
    copy_to_gpu(gz.data, z);
    cuda_check(cudaMemcpy(gtruth, truth.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    float cpu_loss = 0, gpu_loss = 0;
    auto cpu = [&] {
        cpu_loss = 0;
        for (int n = 0; n < N; ++n) {
            float maximum = z[n * C]; int label = 0;
            for (int c = 1; c < C; ++c) {
                if (z[n * C + c] > maximum) { maximum = z[n * C + c]; label = c; }
            }
            prediction[n] = label;
            float sum = 0;
            for (int c = 0; c < C; ++c) { p[n * C + c] = std::exp(z[n * C + c] - maximum); sum += p[n * C + c]; }
            for (int c = 0; c < C; ++c) {
                p[n * C + c] /= sum;
                dz[n * C + c] = (p[n * C + c] - (c == truth[n])) / N;
            }
            cpu_loss -= std::log(std::max(p[n * C + truth[n]], 1e-12f)) / N;
        }
    };
    auto gpu = [&] {
        SoftMax(gz.data, gp.data, gprediction, N, C);
        Cross_Entropy(gp.data, gtruth, gpu_loss, N, C);
        SoftMax_backward(gp.data, gdz.data, gtruth, N, C);
    };
    double cpu_ms = time_cpu(cpu), gpu_ms = time_gpu(gpu);
    compare(copy_from_gpu(gp.data, p.size()), p, "SoftMax probability");
    compare(copy_from_gpu(gdz.data, dz.size()), dz, "SoftMax+CE gradient");
    std::vector<int> gpu_prediction(N);
    cuda_check(cudaMemcpy(gpu_prediction.data(), gprediction, N * sizeof(int), cudaMemcpyDeviceToHost));
    if (gpu_prediction != prediction) throw std::runtime_error("SoftMax predicted labels differ");
    if (std::fabs(gpu_loss - cpu_loss) > 2e-3f) throw std::runtime_error("Cross Entropy loss differs");
    std::cout << "SoftMax labels: PASS; CE loss: CPU=" << cpu_loss << " GPU=" << gpu_loss << '\n';
    cuda_check(cudaFree(gtruth)); cuda_check(cudaFree(gprediction));
    std::cout << "SoftMax/CE shape: N=32768 C=100\n";
    show("SoftMax+CE+backward", cpu_ms, gpu_ms);
}

int main() {
    try {
        cudaDeviceProp properties;
        cuda_check(cudaGetDeviceProperties(&properties, 0));
        std::cout << "GPU: " << properties.name << '\n';
        std::cout << "Average of 5 runs after 1 warm-up; inputs already resident; "
                     "GPU timing includes internal allocation, synchronization and CE scalar copy.\n";
        cublasHandle_t handle;
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
        benchmark_fc(handle);
        benchmark_convolution(handle);
        benchmark_pool();
        benchmark_softmax_loss();
        cublasDestroy(handle);
        std::cout << "ALL LARGE-INPUT CHECKS PASSED\n";
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
