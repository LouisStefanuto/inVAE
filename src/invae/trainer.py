from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import mlflow
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from invae.mlflow_utils import get_next_run_name
from invae.plots import plot_losses


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: str,
) -> float:
    """Train the model for one epoch and return average loss."""
    model.train()
    total_loss: float = 0.0
    for batch in tqdm(dataloader, desc="[Train]"):
        x = batch.X.to(device)

        optimizer.zero_grad()
        outputs = model(x)

        if isinstance(outputs, tuple) and len(outputs) == 3:
            recon, mu, logvar = outputs
            loss = loss_fn(recon, x, mu, logvar)
        else:
            loss = loss_fn(outputs, x)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss: float = total_loss / len(dataloader.dataset)
    return avg_loss


def validate(
    model: torch.nn.Module, dataloader: DataLoader, loss_fn: Callable, device: str
) -> float:
    """Validate the model and return average loss."""
    model.eval()
    total_loss: float = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Val]"):
            x = batch.X.to(device)
            outputs = model(x)

            if isinstance(outputs, tuple) and len(outputs) == 3:
                recon, mu, logvar = outputs
                loss = loss_fn(recon, x, mu, logvar)
            else:
                loss = loss_fn(outputs, x)

            total_loss += loss.item()

    avg_loss: float = total_loss / len(dataloader.dataset)
    return avg_loss


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    """Save model state dict to the specified path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    num_epochs: int = 20,
    device: str = "cuda",
    ckpt_dir: str = "checkpoints",
    metrics: Optional[Dict[str, Callable]] = None,
    experiment_name: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Training loop with MLflow logging, loss plotting, and checkpointing.

    Returns:
        run_id (str), checkpoint path (str)
    """
    model.to(device)

    if experiment_name:
        mlflow.set_experiment(experiment_name)
        run_name = get_next_run_name(experiment_name)
    else:
        run_name = "run_1"  # default if no experiment provided

    train_losses: List[float] = []
    val_losses: List[float] = []

    with mlflow.start_run() as run:
        run_id: str = run.info.run_id
        mlflow.set_tag("mlflow.runName", run_name)

        # Log hyperparameters
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("optimizer", optimizer.__class__.__name__)
        mlflow.log_param(
            "loss_fn",
            loss_fn.__name__ if hasattr(loss_fn, "__name__") else str(loss_fn),
        )

        # Training loop
        for epoch in range(num_epochs):
            avg_train_loss: float = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device
            )
            avg_val_loss: float = validate(model, val_loader, loss_fn, device)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Log metrics
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            print(
                f"Epoch {epoch + 1}/{num_epochs} "
                f"Train Loss: {avg_train_loss:.4f} "
                f"Val Loss: {avg_val_loss:.4f}"
            )

        # Save checkpoint and plot
        ckpt_path: Path = Path(ckpt_dir) / f"model_{run_id}.pt"
        save_checkpoint(model, ckpt_path)
        mlflow.log_artifact(str(ckpt_path))

        plot_path: Path = Path(ckpt_dir) / f"loss_plot_{run_id}.png"
        plot_losses(train_losses, val_losses, plot_path)
        mlflow.log_artifact(str(plot_path))

        # Register model
        mlflow.pytorch.log_model(model, name="model", registered_model_name="model")
        print(f"Model saved to {ckpt_path} and loss plot saved to {plot_path}")

    return run_id, str(ckpt_path)
