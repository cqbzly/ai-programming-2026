# 人工智能中的编程第三次作业

在本次作业中，请实现卷积神经网络中常用的 5 种 Module 的正向与反向传播。

## 全连接层（Fully Connected Layer）

请按照第六讲第 3 页的公式，及 36–42 页的示例，实现全连接层的正向与反向传播。

- **正向传播**：给定 input $X$、全连接参数 $W$、偏置 $b$，计算 output $Y$。
- **反向传播**：给定 $\frac{\partial L}{\partial Y}$，计算全连接层参数的梯度 $\frac{\partial L}{\partial W}$、$\frac{\partial L}{\partial b}$ 以及输入的梯度 $\frac{\partial L}{\partial X}$。

输入 Tensor 的 shape 定义为 $(N, C_{in})$，其中 $N$ 为 batch size，$C_{in}$ 为输入 channel 数；输出 Tensor 的 shape 定义为 $(N, C_{out})$，$C_{out}$ 为输出 channel 数。在进行实例测试时，$N, C_{in}, C_{out}$ 可自行定义。


## 卷积层（Convolution Layer）

请采用 **im2col + GEMM** 的方案（见[附录 A](#附录-a卷积层im2col--gemm)）实现 `im2col` 以及 `col2im`，并在此基础上实现卷积层的正向与反向传播。

为简化起见，输入 Tensor 的 shape 定义为 $(N, C_{in}, H, W)$，其中 $N$ 为 batch size，$C_{in}$ 为输入 channel 数，$H, W$ 为 feature map 的高和宽。在此基础上实现 2D 卷积，卷积核大小为 $3 \times 3$，padding 策略为 zero padding，stride 为 1，因此输出 Tensor 的 shape 为 $(N, C_{out}, H, W)$。在进行实例测试时，$N, C_{in}, C_{out}, H, W$ 可自行定义。

Depthwise Convolution、Dilated Convolution 以及 Transposed Convolution 不做要求。

为简化起见，FC 层和卷积层都只使用单精度的 `cublasSgemm` 即可，并使用 cuRAND 对网络参数进行初始化。

## 池化层（Pooling Layer）

为简化起见，请仿照[附录 B](#附录-bmax-pooling-的函数接口) 给出的函数接口，实现 kernel size 为 $2 \times 2$、stride 为 2 的 Max Pooling Layer 的正向与反向传播。输入 Tensor 的 shape 定义为 $(N, C, H, W)$，输出 Tensor 的 shape 为 $(N, C, H/2, W/2)$。在进行实例测试时，$N, C, H, W$ 可自行定义。Unpooling Layer 不做要求。

## SoftMax 层

请按照[附录 C](#附录-csoftmax-的计算方式) 的计算方式实现 SoftMax 的正向传播。输入 Tensor 的 shape 定义为 $(N, C)$，其中 $N$ 为 batch size，$C$ 为要进行分类的类别数量，例如对于 MNIST 和 Cifar10，$C=10$，对于 Cifar100，$C=100$。输出 Tensor 的 shape 同样为 $(N, C)$，代表了对于一个 batch 内 $N$ 个样本分到每个类别的概率。

注意，不需要单独对 SoftMax 层实现反向传播。在进行实例测试时，$N, C$ 可自行定义。

## Cross Entropy Loss

请对 SoftMax 层输出的概率值和真实 label 计算 Cross Entropy Loss。SoftMax 层输出的 Tensor 的 shape 大小为 $(N, C)$，真实 label 的 shape 为 $(N,)$，取值范围为 $[0, C-1]$。在进行实例测试时，$N, C$ 可自行定义，真实 label 的值可在 $[0, C-1]$ 中随机选取。

请实现 Cross Entropy Loss with Softmax 的反向传播，即按照[附录 D](#附录-dcross-entropy-loss-with-softmax-的反向传播) 的推导，计算 Cross Entropy Loss 对 SoftMax 层输入神经元的梯度。将两者的梯度进行结合更有利于数值的稳定性。

---

> **注**：本次作业只需要实现所有模块的正向与反向传播的核心函数，不需要进行封装和拼接。需要在 `main` 函数中设计并进行实例化测试，验证结果的正确性。另外为简化起见，可以不使用模板类型参数，所有函数只需要接受单精度浮点数类型指针作为输入和输出。

## 评分标准

本次作业占最终课程总成绩的 **25 分**，其中：

| 模块 | 分值 |
| --- | --- |
| 全连接层 | 5 分 |
| 卷积层 | 5 分 |
| 池化层 | 5 分 |
| SoftMax 层 | 5 分 |
| Cross Entropy Loss | 5 分 |

请提交**源码及作业报告**。在报告中，需要写明每个模块中核心函数的设计思路、你如何进行正确性测试，以及测试结果。

本次作业的截止日期为 **2026 年 11 月 11 日 23:59**，每迟交一天扣一分，扣一半为止。

本次作业内容较多，请大家务必规划好时间，尽早开始。

---

# 附录

## 附录 A：卷积层（im2col + GEMM）

### A.1 基本思想

im2col 将输入中每个卷积窗口展开成矩阵 $\hat{X}$ 的一列（或一行），从而把卷积转化为一次矩阵乘法（GEMM）。以一维输入 $x_1, \dots, x_5$、长度为 3 的卷积核 $w_1, w_2, w_3$、zero padding 为 1 为例：

$$
\begin{bmatrix} y_1 \\ y_2 \\ y_3 \\ y_4 \\ y_5 \end{bmatrix}
=
\underbrace{\begin{bmatrix}
0   & x_1 & x_2 \\
x_1 & x_2 & x_3 \\
x_2 & x_3 & x_4 \\
x_3 & x_4 & x_5 \\
x_4 & x_5 & 0
\end{bmatrix}}_{\hat{X}}
\begin{bmatrix} w_1 \\ w_2 \\ w_3 \end{bmatrix}
$$

- 优点：实现简单，且借助高度优化的 GEMM 非常高效。
- 缺点：$\hat{X}$ 会把输入复制约 $k \times k$ 倍，**占用大量显存**。因此对于形状为 $[N, C, H, W]$ 的输入，应**在 $N$ 维度上循环**，每次只对一个样本构造 $\hat{X}$。
- 若需要 stride 为 2 的下采样卷积，只需在 im2col 构造 $\hat{X}$ 时跳过部分窗口，再执行 GEMM 即可（本次作业 stride 固定为 1）。

### A.2 多通道情形下的矩阵形状

对 batch 中的第 $n$ 个样本，卷积核大小为 $k \times k$（本作业 $k=3$）：

| 矩阵 | 含义 | 形状 |
| --- | --- | --- |
| $\hat{X}$ | im2col 的结果，每一列对应一个输出位置 | $(C_{in} \cdot k \cdot k,\ H \cdot W)$ |
| $W$ | 卷积核参数 | $(C_{out},\ C_{in} \cdot k \cdot k)$ |
| $b$ | 偏置 | $(C_{out})$ |
| $Y$ | 输出 | $(C_{out},\ H \cdot W)$ |

### A.3 实现步骤

1. **实现 `im2col` 与 `col2im`**。`col2im` 是 `im2col` 的逆过程：把 $\hat{X}$ 形状的矩阵中的每个元素**累加**回其在输入 feature map 中对应的位置（同一输入像素会出现在多个窗口中，因此必须累加而非覆盖；padding 位置直接丢弃）。
2. **正向传播**：结合 im2col 与 GEMM：
   $$Y = W \hat{X}$$
3. **对权重的反向传播**：结合 im2col 与 GEMM：
   $$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \times \hat{X}^T$$
   注意需要在 batch 的 $N$ 个样本上累加。
4. **对输入的反向传播**：结合 col2im 与 GEMM：
   $$\frac{\partial L}{\partial \hat{X}} = W^T \times \frac{\partial L}{\partial Y}$$
   再通过 col2im 将 $\frac{\partial L}{\partial \hat{X}}$ 还原（累加）为 $\frac{\partial L}{\partial X}$。
5. **用 GEMM 加上卷积偏置**。偏置也可以看作一个独立的神经网络层：
   - 正向：$Y \mathrel{+}= b \cdot \mathbf{1}^T$，其中 $\mathbf{1}$ 是长度为 $H \cdot W$ 的全 1 向量；
   - 反向：$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial Y} \cdot \mathbf{1}$，同样在 $N$ 个样本上累加。

> **提示**：cuBLAS 默认采用列主序（column-major）存储，而 Tensor 一般按行主序存储，调用 `cublasSgemm` 时请注意转置参数与 leading dimension 的设置。

## 附录 B：Max Pooling 的函数接口

正向传播时，每个线程负责一个输出位置所对应的局部窗口：求出窗口内的最大值写入 `out_data[index]`，同时将最大值在输入中的位置写入 `out_mask[index]`，供反向传播使用。

```cuda
__global__ void max_pool_forward(
    float* in_data, int nthreads, int num, int channels,
    int in_h, int in_w, int out_h, int out_w, int kernel_h,
    int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, float* out_data, float* out_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n  = index / out_w / out_h / channels;
    int c  = (index / out_w / out_h) % channels;
    int ph = (index / out_w) % out_h;
    int pw = index % out_w;

    // implement max pooling for each local window,
    // store the max value and mask to out_data[index] & out_mask[index]
  }
}
```

**反向传播**：请仿照上述接口自行设计 `max_pool_backward`。给定 $\frac{\partial L}{\partial Y}$ 与正向传播保存的 `out_mask`，将每个输出位置的梯度传回其最大值所在的输入位置，窗口内其余位置的梯度为 0。

## 附录 C：SoftMax 的计算方式

输入 $[N, C]$，输出 $[N, C]$（例如 MNIST 分类中 $C = 10$）：

$$
\text{softmax}(z)_{n,i} = \frac{e^{z_{n,i} - m_n}}{\sum_{j=1}^{C} e^{z_{n,j} - m_n}}, \quad m_n = \max_j z_{n,j}
$$

实现步骤：

1. **计算每一行的最大值**：使用 max reduce，或直接让每个线程负责一行进行计算；
2. **每行减去该行的最大值**（Map），防止指数运算溢出；
3. **对每个元素计算指数**（Map）；
4. **对每行求和得到归一化因子**：使用 sum reduce，或直接让每个线程负责一行进行计算；
5. **归一化**（Map）：每个元素除以所在行的归一化因子。

## 附录 D：Cross Entropy Loss with Softmax 的反向传播

设 SoftMax 层的输入为 $z \in \mathbb{R}^{N \times C}$，输出概率为 $p = \text{softmax}(z)$，第 $n$ 个样本的真实 label 为 $y_n \in [0, C-1]$。batch 平均的 Cross Entropy Loss 为：

$$
L = -\frac{1}{N} \sum_{n=1}^{N} \log p_{n, y_n}
$$

将 SoftMax 与 Cross Entropy 的梯度结合后，Loss 对 SoftMax 输入的梯度有简洁的形式：

$$
\frac{\partial L}{\partial z_{n,i}} = \frac{1}{N} \left( p_{n,i} - \mathbb{1}[i = y_n] \right)
$$

即用概率减去 one-hot 编码的真实 label，再除以 $N$。这样既避免了单独计算 $\frac{\partial L}{\partial p}$ 时 $\frac{1}{p}$ 带来的数值不稳定，计算量也更小。计算 $\log p$ 时，可对 $p$ 加一个很小的下限（如 $10^{-12}$）以避免 $\log 0$。
