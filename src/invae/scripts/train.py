import torch
from anndata import AnnData
from torch.utils.data import DataLoader
from torchsummary import summary

from invae.config import Config
from invae.trainer import train_model
from invae.vae import VAE, vae_loss


def train(
    cfg: Config, adata: AnnData, train_loader: DataLoader, val_loader: DataLoader
) -> str:
    # Init model
    input_dim = adata.n_vars
    model = VAE(input_dim, hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim)
    summary(model, input_size=(input_dim,))

    # Define optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = vae_loss

    # Train (agnostic loop)
    run_id, ckpt_path = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        num_epochs=cfg.num_epochs,
        device=cfg.device,
        ckpt_dir=cfg.ckpt_dir,
        experiment_name=cfg.experiment_name,
    )

    return ckpt_path
