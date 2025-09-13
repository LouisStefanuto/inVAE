from pathlib import Path
from typing import Callable, Dict, Optional

import mlflow
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


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
) -> tuple[str, str]:
    """
    Generic training loop with MLflow logging.

    Args:
        model: PyTorch model
        train_loader: training dataloader
        val_loader: validation dataloader
        optimizer: optimizer
        loss_fn: loss function
        num_epochs: number of epochs
        device: 'cuda', 'cpu', or 'mps'
        ckpt_dir: directory to save model checkpoint
        metrics: dict of additional metrics {name: fn(y_pred, y_true)}

    Returns:
        run_id (str): The MLflow run ID for this training run
    """
    model.to(device)

    if experiment_name is not None:
        mlflow.set_experiment(experiment_name)

    # MLflow logging
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        run_name = f"run_{run_id[:8]}"  # shorter display name
        mlflow.set_tag("mlflow.runName", run_name)

        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("optimizer", optimizer.__class__.__name__)
        mlflow.log_param(
            "loss_fn",
            loss_fn.__name__ if hasattr(loss_fn, "__name__") else str(loss_fn),
        )

        for epoch in range(num_epochs):
            # ---------------- Train ----------------
            model.train()
            total_train_loss = 0.0
            for batch in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"
            ):
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
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader.dataset)

            # ---------------- Validation ----------------
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(
                    val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"
                ):
                    x = batch.X.to(device)
                    outputs = model(x)

                    if isinstance(outputs, tuple) and len(outputs) == 3:
                        recon, mu, logvar = outputs
                        loss = loss_fn(recon, x, mu, logvar)
                    else:
                        loss = loss_fn(outputs, x)

                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader.dataset)

            # Log to MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            print(
                f"Epoch {epoch + 1}/{num_epochs} "
                f"Train Loss: {avg_train_loss:.4f} "
                f"Val Loss: {avg_val_loss:.4f}"
            )

        # ---------------- Save checkpoint ----------------
        ckpt_path = Path(ckpt_dir) / f"model_{run_id}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        mlflow.log_artifact(str(ckpt_path))

        # Register model in MLflow with run-based name
        model_name = f"{run_name}_{run_id[:8]}"
        mlflow.pytorch.log_model(model, name="model", registered_model_name="model")
        print(f"Model saved to {ckpt_path} and registered as {model_name}")

    return run_id, ckpt_path
