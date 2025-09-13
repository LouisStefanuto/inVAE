from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    # Dataset
    filename: str = "./data/pancreas.h5ad"
    backup_url: str = "https://drive.google.com/uc?id=1ehxgfHTsMZXy6YzlFKGJOsBKQ5rrvMnd"

    # Training
    batch_size: int = 32
    num_epochs: int = 200

    # Model
    hidden_dim: int = 128
    latent_dim: int = 10

    # Hardware
    device: str = "mps"

    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )
