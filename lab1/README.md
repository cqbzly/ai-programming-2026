# HW1：使用 LeNet 完成 CIFAR-10 图像分类

本实验使用 PyTorch 实现 LeNet，在 CIFAR-10 数据集上进行图像分类，并比较 SGD 在不同 Momentum 参数下的训练效果。代码文件为 LeNet.py，实验报告为 lab1实验报告.docx。

## 1. 环境与依赖

实验在 Windows + CUDA 环境下完成，主要配置如下：

| 项目 | 配置 |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.14.0+cu130 |
| torchvision | 0.29.0+cu130 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| CUDA Toolkit | 13.0 |
| W&B | 0.30.0 |

创建或激活 Conda 环境后安装依赖：

~~~powershell
conda activate aip
python -m pip install torch torchvision wandb
wandb login
~~~

检查 PyTorch 是否使用 CUDA：

~~~powershell
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
~~~

本实验必须输出 True，否则 LeNet.py 会主动停止，因为作业要求训练和测试均使用 CUDA。

## 2. 数据集

程序使用 torchvision.datasets.CIFAR10 加载数据，并使用以下预处理：

- ToTensor()：将图像转换为 Tensor；
- Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))：对 RGB 三个通道进行归一化。

默认数据目录为 lab1/data。运行时 download=True 会检查数据是否存在；数据已经完整下载并解压时不会重复下载。

## 3. 模型与训练设置

LeNet 的结构为：

~~~text
输入 3×32×32
→ Conv2d(3, 6, kernel_size=5)
→ ReLU → MaxPool2d(2, 2)
→ Conv2d(6, 16, kernel_size=5)
→ ReLU → MaxPool2d(2, 2)
→ Flatten(400)
→ Linear(400, 120) → ReLU
→ Linear(120, 84) → ReLU
→ Linear(84, 10)
~~~

损失函数使用 CrossEntropyLoss，优化器使用 SGD。两组实验只改变 Momentum，其余参数保持一致：

| 参数 | 设置 |
| --- | ---: |
| Epochs | 10 |
| Batch Size | 128 |
| Learning Rate | 0.01 |
| Random Seed | 42 |
| DataLoader workers | 0 |
| Momentum | 0.0 / 0.9 |

运行两组实验：

~~~powershell
cd D:\projects\AIP\lab1
python LeNet.py --momentum 0.0 --learning-rate 0.01 --batch-size 128 --epochs 10 --seed 42
python LeNet.py --momentum 0.9 --learning-rate 0.01 --batch-size 128 --epochs 10 --seed 42
~~~

程序第一次前向计算前会输出模型参数、输入图像和标签的设备，并验证三者均为 CUDA。

## 4. W&B 记录

程序将以下指标同步到 W&B 项目 aipro-hw1-cifar10：

- loss：每个 Batch 的损失；
- epoch_loss：每个 Epoch 的平均损失。

两次真实实验链接如下：

| Momentum | W&B Run |
| ---: | --- |
| 0.0 | [lenet-momentum-0-seed-42](https://wandb.ai/1729229821-peking-university/aipro-hw1-cifar10/runs/jd5ngu0n) |
| 0.9 | [lenet-momentum-0.9-seed-42](https://wandb.ai/1729229821-peking-university/aipro-hw1-cifar10/runs/d1h280ue) |

在 W&B 中打开两个 Run 后，将 epoch_loss 曲线放在同一图中即可比较两种 Momentum 的收敛速度；loss 曲线可以用于观察 Batch 级别的波动。

## 5. 实验结果

### 5.1 每轮平均损失

| Epoch | Momentum = 0.0 | Momentum = 0.9 |
| ---: | ---: | ---: |
| 1 | 2.3021 | 1.9890 |
| 2 | 2.2873 | 1.5139 |
| 3 | 2.1680 | 1.3454 |
| 4 | 1.9762 | 1.2301 |
| 5 | 1.8589 | 1.1449 |
| 6 | 1.7599 | 1.0764 |
| 7 | 1.6588 | 1.0191 |
| 8 | 1.5894 | 0.9701 |
| 9 | 1.5427 | 0.9222 |
| 10 | 1.5111 | 0.8742 |

### 5.2 测试集准确率

| 类别 | Momentum = 0.0 | Momentum = 0.9 |
| --- | ---: | ---: |
| Overall | **45.48%** | **62.43%** |
| airplane | 61.20% | 58.90% |
| automobile | 54.20% | 77.80% |
| bird | 28.80% | 40.80% |
| cat | 23.70% | 36.20% |
| deer | 55.40% | 65.50% |
| dog | 46.10% | 57.20% |
| frog | 37.40% | 58.70% |
| horse | 50.30% | 76.10% |
| ship | 52.10% | 73.30% |
| truck | 45.60% | 79.80% |

### 5.3 Momentum 分析

Momentum 是 SGD 中用于累积历史梯度方向的系数。可用下面的简化公式表示：

~~~text
velocity = momentum × previous_velocity + gradient
parameter = parameter − learning_rate × velocity
~~~

当连续多个 Batch 的梯度方向较一致时，历史方向会帮助参数更快前进；当梯度方向来回变化时，历史方向可以减轻更新的振荡。Momentum = 0.0 等价于普通 SGD，不保留历史动量；Momentum = 0.9 会保留较强的历史方向影响。

在本实验中，Momentum = 0.9 的平均损失从 1.9890 下降到 0.8742，而 Momentum = 0.0 从 2.3021 下降到 1.5111。同时，Momentum = 0.9 的整体测试准确率为 62.43%，高于 Momentum = 0.0 的 45.48%。因此，在本次固定学习率、Batch Size、Epoch 数、随机种子和网络结构的条件下，Momentum = 0.9 收敛更快，最终训练损失更低，测试集表现也更好。

## 6. 输出文件

每次运行会在 outputs 下创建独立目录：

~~~text
outputs/
├── momentum_0_seed_42_jd5ngu0n/
│   ├── lenet_cifar10.pth
│   └── results.json
└── momentum_0.9_seed_42_d1h280ue/
    ├── lenet_cifar10.pth
    └── results.json
~~~

lenet_cifar10.pth 保存模型参数，results.json 保存整体准确率、10 个类别的准确率、每轮平均损失、实验配置和 W&B Run 链接。
