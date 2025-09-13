from invae.config import Config
from invae.dataloader import get_dataloaders, load_pancreas_data
from invae.scripts.inference import inference
from invae.scripts.train import train


def main() -> None:
    cfg = Config()

    print("----- Starting run ------")
    print(cfg.model_dump_json())
    print("-------------------------")

    # Load data
    adata = load_pancreas_data(cfg.filename, cfg.backup_url)
    train_loader, val_loader, test_loader = get_dataloaders(
        adata,
        batch_size=cfg.batch_size,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        use_cuda=False,
    )

    print("----- Train -------------")
    ckpt_path = train(cfg, adata, train_loader, val_loader)
    print("----- Inference ---------")
    inference(cfg, adata, test_loader, ckpt_path)
    print("-------------------------")


if __name__ == "__main__":
    main()
