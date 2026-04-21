import csv
import os
import random

os.makedirs("./results/mpl_cache", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "./results/mpl_cache")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Basic settings
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SEED = 42

EXPERIMENTS = ["baseline", "flip", "crop", "flip_crop"]


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_transform(name):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    if name == "baseline":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif name == "flip":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif name == "crop":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif name == "flip_crop":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, test_transform


def get_data_loaders(experiment_name):
    train_transform, test_transform = build_transform(experiment_name)

    trainset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    testset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    return trainloader, testloader


def train_one_epoch(model, trainloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0

    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / total, correct / total


def test_model(model, testloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / total, correct / total


def run_experiment(experiment_name, device):
    print("\nExperiment:", experiment_name)

    trainloader, testloader = get_data_loaders(experiment_name)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device
        )
        test_loss, test_acc = test_model(model, testloader, criterion, device)

        result = {
            "experiment": experiment_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        }
        history.append(result)

        print(
            f"Epoch {epoch:02d}: "
            f"train acc={train_acc:.4f}, test acc={test_acc:.4f}, "
            f"train loss={train_loss:.4f}, test loss={test_loss:.4f}"
        )

    return history


def save_results(all_results):
    os.makedirs("./results", exist_ok=True)

    with open("./results/experiment_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "epoch", "train_loss", "train_acc", "test_loss", "test_acc"]
        )
        writer.writeheader()
        writer.writerows(all_results)


def plot_results(all_results):
    os.makedirs("./results", exist_ok=True)

    plt.figure(figsize=(9, 5))
    for name in EXPERIMENTS:
        rows = [row for row in all_results if row["experiment"] == name]
        epochs = [row["epoch"] for row in rows]
        test_acc = [row["test_acc"] for row in rows]
        plt.plot(epochs, test_acc, marker="o", label=name)

    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./results/test_accuracy.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    for name in EXPERIMENTS:
        rows = [row for row in all_results if row["experiment"] == name]
        epochs = [row["epoch"] for row in rows]
        train_loss = [row["train_loss"] for row in rows]
        test_loss = [row["test_loss"] for row in rows]
        plt.plot(epochs, train_loss, linestyle="--", label=f"{name} train")
        plt.plot(epochs, test_loss, label=f"{name} test")

    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("./results/loss_curves.png", dpi=200)
    plt.close()

    names = []
    best_acc = []
    for name in EXPERIMENTS:
        rows = [row for row in all_results if row["experiment"] == name]
        names.append(name)
        best_acc.append(max(row["test_acc"] for row in rows))

    plt.figure(figsize=(8, 5))
    plt.bar(names, best_acc)
    plt.title("Best Test Accuracy")
    plt.xlabel("Experiment")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("./results/best_accuracy.png", dpi=200)
    plt.close()


def main():
    set_seed()
    device = torch.device("mps")
    print("Using device:", device)

    all_results = []
    for experiment_name in EXPERIMENTS:
        results = run_experiment(experiment_name, device)
        all_results.extend(results)

    save_results(all_results)
    plot_results(all_results)

    print("\nFinished. Results are saved in the results folder.")
    print("- results/experiment_results.csv")
    print("- results/test_accuracy.png")
    print("- results/loss_curves.png")
    print("- results/best_accuracy.png")


if __name__ == "__main__":
    main()
