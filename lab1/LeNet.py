import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_loaders(data_dir, batch_size, num_workers):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    return (
        torch.utils.data.DataLoader(trainset, shuffle=True, **loader_args),
        torch.utils.data.DataLoader(testset, shuffle=False, **loader_args),
        trainset.classes,
    )


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, run):
    model.train()
    running_loss = 0.0
    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if epoch == 1 and batch_index == 0:
            print("model parameters device:", next(model.parameters()).device)
            print("input images device:", images.device)
            print("labels device:", labels.device)
            if not all(parameter.is_cuda for parameter in model.parameters()):
                raise RuntimeError("All model parameters must be on CUDA.")
            if not images.is_cuda or not labels.is_cuda:
                raise RuntimeError("Images and labels must be on CUDA.")

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss
        run.log({"loss": batch_loss})

    epoch_loss = running_loss / len(loader)
    run.log({"epoch_loss": epoch_loss})
    return epoch_loss


@torch.no_grad()
def evaluate(model, loader, device, classes):
    model.eval()
    class_correct = torch.zeros(len(classes), dtype=torch.long, device=device)
    class_total = torch.zeros(len(classes), dtype=torch.long, device=device)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = model(images).argmax(dim=1)
        class_total += torch.bincount(labels, minlength=len(classes))
        class_correct += torch.bincount(
            labels[predictions == labels], minlength=len(classes)
        )

    class_correct = class_correct.cpu().tolist()
    class_total = class_total.cpu().tolist()
    correct = sum(class_correct)
    total = sum(class_total)
    accuracy = 100.0 * correct / total
    per_class_accuracy = {}
    print(f"Overall accuracy: {accuracy:.2f}%")
    for class_name, class_hits, class_count in zip(
        classes, class_correct, class_total
    ):
        class_accuracy = 100.0 * class_hits / class_count
        per_class_accuracy[class_name] = class_accuracy
        print(f"{class_name:>10}: {class_accuracy:.2f}%")
    return {
        "overall_accuracy_percent": accuracy,
        "per_class_accuracy_percent": per_class_accuracy,
    }


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train LeNet on CIFAR-10")
    parser.add_argument("--data-dir", default=str(script_dir / "data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(script_dir / "outputs"))
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive.")
    if args.learning_rate <= 0 or args.momentum < 0 or args.num_workers < 0:
        parser.error("learning-rate must be positive; momentum and num-workers cannot be negative.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this assignment. Activate an environment "
            "with a CUDA-enabled PyTorch build (for example: conda activate aip)."
        )
    if not callable(getattr(wandb, "init", None)):
        raise RuntimeError("W&B is required. Install it with: python -m pip install wandb")
    set_seed(args.seed)
    device = torch.device("cuda")
    print("device:", device)
    print("GPU:", torch.cuda.get_device_name(device))

    trainloader, testloader, classes = build_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum
    )

    with wandb.init(
        project="aipro-hw1-cifar10",
        name=f"lenet-momentum-{args.momentum:g}-seed-{args.seed}",
        config=vars(args),
    ) as run:
        output_dir = (
            Path(args.output_dir)
            / f"momentum_{args.momentum:g}_seed_{args.seed}_{run.id}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        epoch_losses = []
        for epoch in range(1, args.epochs + 1):
            epoch_loss = train_one_epoch(
                model, trainloader, criterion, optimizer, device, epoch, run
            )
            epoch_losses.append(epoch_loss)
            print(f"Epoch {epoch}/{args.epochs}, loss: {epoch_loss:.4f}")

        results = evaluate(model, testloader, device, classes)
        results.update({
            "config": vars(args),
            "epoch_losses": epoch_losses,
            "wandb_run_id": run.id,
            "wandb_run_url": run.url,
        })
        torch.save(model.state_dict(), output_dir / "lenet_cifar10.pth")
        (output_dir / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        print(f"Saved model and results to: {output_dir.resolve()}")
        print("W&B Run:", results["wandb_run_url"])


if __name__ == "__main__":
    main()
