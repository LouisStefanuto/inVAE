import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class VAE(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 10
    ) -> None:
        super().__init__()
        # encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc_out(h))

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


def train_vae(
    model: VAE,
    dataloader: DataLoader,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
) -> None:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss: float = 0.0
        for batch in tqdm(dataloader):
            x = batch.X.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader.dataset):.4f}")


def extract_latent(
    model: VAE, dataloader: DataLoader, device: str = "cuda"
) -> np.ndarray:
    model.eval()
    zs: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch.X.to(device)
            mu, _ = model.encode(x)
            zs.append(mu.cpu())
    return torch.cat(zs, dim=0).numpy()
