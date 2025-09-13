from invae.config import Config
from invae.dataloader import build_dataloader, load_pancreas_data
from invae.vae import VAE, extract_latent, train_vae


def main() -> None:
    cfg = Config()
    print(cfg.model_dump_json())

    adata = load_pancreas_data(cfg.filename, cfg.backup_url)
    dataloader = build_dataloader(adata, None, use_cuda=False)

    input_dim = adata.n_vars
    model = VAE(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
    )
    train_vae(model, dataloader, num_epochs=cfg.num_epochs, device=cfg.device)

    # save latent space into adata
    latent = extract_latent(model, dataloader, device=cfg.device)
    adata.obsm["X_vae"] = latent
    print("Latent embeddings shape:", latent.shape)


if __name__ == "__main__":
    main()
