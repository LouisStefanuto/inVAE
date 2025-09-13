import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class VAE(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 10
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc_out(F.relu(self.fc2(z))))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


def load_model(
    ckpt_path: str,
    input_dim: int,
    hidden_dim: int = 128,
    latent_dim: int = 10,
    device: str = "cuda",
) -> VAE:
    """Load a trained VAE model from checkpoint."""
    model = VAE(input_dim, hidden_dim, latent_dim)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def extract_latent(model: VAE, dataloader, device: str = "cuda") -> np.ndarray:
    """Extract latent mean (mu) for all samples in the dataloader."""
    model.eval()
    zs: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting latent space"):
            x = batch.X.to(device)
            mu, _ = model.encode(x)
            zs.append(mu.cpu())
    return torch.cat(zs, dim=0).numpy()
