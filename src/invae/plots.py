from pathlib import Path

import matplotlib.pyplot as plt


def plot_losses(train_losses: list[float], val_losses: list[float], path: Path) -> None:
    """Plot training and validation losses and save to path."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
