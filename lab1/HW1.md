# 《人工智能中的编程》第一次作业

## 使用 LeNet 完成 CIFAR-10 图像分类

## Part 0：环境配置

在开始编程前，请准备一个独立、可复现的开发环境。本部分不单独计分，但提交的程序必须能够在配置正确的环境中运行。

#### 0.1 Python 与 Conda

推荐使用 Miniconda 或 Anaconda 管理虚拟环境，避免不同项目的依赖相互冲突。

参考资料：

- [Miniconda 安装说明](https://www.anaconda.com/docs/getting-started/miniconda/main)
- [Conda 环境管理文档](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

可以使用下面的命令创建本课程的环境：

```bash
conda create -n myenv python=3.10
conda activate myenv
```

常用命令如下：

```bash
conda env list                 # 查看所有 Conda 环境
conda list                     # 查看当前环境中的软件包
conda install <package_name>   # 在当前环境中安装软件包
conda deactivate               # 退出当前环境
```

#### 0.2 安装 CUDA Toolkit

在安装Cuda Toolkit前，请先确认⾃⼰的平台拥有⼀块⽀持CUDA的Nvidia显卡，并检查对应的CUDA Version，这可以通过 nvidia-smi 命令查看，会得到类似下图的输出

![CleanShot 2026-09-14 at 09.42.16@2x](/Users/turbo/Library/Application Support/CleanShot/media/media_ouTD3SK22B/CleanShot 2026-09-14 at 09.42.16@2x.png)

上图右上⻆的 `CUDA Version: 12.2` 即为所需，不同的显卡可能对应不同的CUDA Version。在确定版本后，请到[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)中下载对应版本的⼯具包。完成安装后，可以通过运⾏`nvcc --version`来确认是否安装成功。

#### 0.3 安装 PyTorch

请根据自己的操作系统、Python 版本和 CUDA 环境，使用 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 生成安装命令，并选择支持 CUDA 的 PyTorch 版本。

安装完成后，可运行：

```bash
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
```

#### 0.4 C++

课程前半部分内容涉及CUDA C++代码的编写，需要⼤家准备可⽤的C++编译器。请⼤家根据⾃⼰的平台，准备[MSVC](https://visualstudio.microsoft.com/zh-hans/downloads/)/[g++](https://learnubuntu.com/install-g-plus-plus/)/任意喜欢的编译器即可.

为了更便捷地进⾏多⽂件编译，推荐同学们使⽤CMake或类似⼯具管理项⽬。这⾥给出[CMake](https://cmake.org/download/)官⽅的安装链接与[Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)。CMake也可以通过conda进⾏安装，请参考[这个链接](https://anaconda.org/anaconda/cmake)。

#### 0.5 其他

确保实验在GPU运行，一个常见的设备选择方式是：

```python
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this assignment.")

device = torch.device("cuda")
print("device:", device)
```

W&B 的安装和使用方式见文末的 [W&B 使用参考](#wandb-reference)。

---

## Part 1：CIFAR-10 图像分类

请参考第一讲课程内容和 [PyTorch 官方教程](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)，使用 PyTorch 实现 LeNet，在 CIFAR-10 数据集上完成图像分类，并撰写一份简要实验报告。

#### 1.1 数据集加载与处理

使用 `torchvision.datasets.CIFAR10` 下载并处理 CIFAR-10 数据集，通过 `DataLoader` 分别加载训练集和测试集。

#### 1.2 使用 PyTorch 实现 LeNet

继承 `torch.nn.Module` 实现适用于 CIFAR-10 的 LeNet，网络结构如下：

| 层 | 参数 | 输出形状（不含 Batch 维度） |
| --- | --- | --- |
| Input | CIFAR-10 RGB 图像 | `3 × 32 × 32` |
| Conv1 | `3 → 6`，Kernel `5 × 5` | `6 × 28 × 28` |
| ReLU + MaxPool | Kernel `2 × 2`，Stride 2 | `6 × 14 × 14` |
| Conv2 | `6 → 16`，Kernel `5 × 5` | `16 × 10 × 10` |
| ReLU + MaxPool | Kernel `2 × 2`，Stride 2 | `16 × 5 × 5` |
| Flatten | - | `400` |
| Linear1 + ReLU | `400 → 120` | `120` |
| Linear2 + ReLU | `120 → 84` | `84` |
| Linear3 | `84 → 10` | `10` |

#### 1.3 损失函数与优化器

损失函数使用 `torch.nn.CrossEntropyLoss`，优化器使用 `torch.optim.SGD`。Learning Rate、Momentum 和 Batch Size 可自行设置。

#### 1.4 模型训练

训练模型 10 个 Epoch，且训练过程必须在 CUDA 上进行。第一次前向计算前，输出模型参数、输入图像和标签所在的 Device，并确保三者均为 CUDA。

#### 1.5 使用 W&B 记录实验

使用 W&B 记录每个 Batch 的 `loss` 和每个 Epoch 的平均 `epoch_loss`。使用方式见文末的 [W&B 使用参考](#wandb-reference)。

#### 1.6 模型测试

在测试集上使用 CUDA 评估模型，计算整体平均准确率以及 10 个类别各自的准确率。

---

## Part 2：实验报告

提交一份简要实验报告，内容应包括：

1. **实验结果**：测试集整体平均准确率，以及 10 个类别各自的准确率。
2. **损失曲线**：由 W&B 记录的 Loss Curve。
3. **问题回答**：SGD 优化器的 Momentum 参数表示什么？至少选取两个不同的 Momentum 值进行实验，在其他设置相同的条件下比较对应的 Loss Curve，并分析差异。

---

### 评分标准

本次作业满分 10 分。

#### 代码：6 分

| 项目 | 分值 |
| --- | ---: |
| 正确下载、预处理并加载 CIFAR-10 | 1 |
| 正确实现指定的 LeNet | 1 |
| 正确使用 CrossEntropyLoss 和 SGD | 1 |
| 使用 CUDA 完成 10 个 Epoch 的训练，并正确输出 Device | 1 |
| 正确计算整体准确率和各类别准确率 | 1 |
| 使用 W&B 正确记录 `loss` 和 `epoch_loss` | 1 |

#### 实验报告：4 分

| 项目 | 分值 |
| --- | ---: |
| 完整报告整体及各类别测试结果 | 1 |
| 提供清晰的 W&B Loss Curve | 1 |
| 正确解释 Momentum 的含义 | 1 |
| 公平完成两组实验并结合曲线分析 Momentum 的影响 | 1 |

---

### 提交要求

- 代码需要提交可运行的 `.py` 源码；
- README 中应写明依赖安装方式、运行命令、输出位置和 W&B Run 链接；
- 作业提交截止时间为2026年9月23日23:59:59，每迟交一天扣 1 分，最多扣至本次作业分数的一半。

---

<a id="wandb-reference"></a>

### 附录：W&B 使用参考

安装并登录：

```bash
python -m pip install wandb
wandb login # 创建并输入自己的API Key
```

在训练代码中初始化 W&B，并且只记录 `loss` 和 `epoch_loss`：

```python
import wandb

wandb.init(project="aipro-hw1-cifar10")

for epoch in range(epochs):
    running_loss = 0.0

    for images, labels in trainloader:
        # 省略前向传播、反向传播和参数更新
        running_loss += loss.item()
        wandb.log({"loss": loss.item()})

    epoch_loss = running_loss / len(trainloader)
    wandb.log({"epoch_loss": epoch_loss})

wandb.finish()
```

参考：[W&B Quickstart](https://docs.wandb.ai/quickstart/)。
