from anndata import AnnData
from numpy import ndarray
from torch.utils.data import DataLoader

from invae.config import Config
from invae.vae import extract_latent, load_model


def inference(
    cfg: Config,
    adata: AnnData,
    test_loader: DataLoader,
    ckpt_path: str,
) -> ndarray:
    # Reload trained model
    input_dim = adata.n_vars
    model = load_model(
        ckpt_path,
        input_dim,
        cfg.hidden_dim,
        cfg.latent_dim,
        device=cfg.device,
    )

    # Extract latent embeddings
    latent = extract_latent(model, test_loader, device=cfg.device)
    print("Latent embeddings (test) shape:", latent.shape)

    return latent
