from typing import Any, Dict, Tuple

import numpy as np
import scanpy as sc
from anndata import AnnData
from anndata.experimental.pytorch import AnnLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def load_pancreas_data(filename: str, backup_url: str) -> AnnData:
    adata = sc.read(filename, backup_url=backup_url)
    return adata


def split_adata(
    adata: AnnData,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[AnnData, AnnData, AnnData]:
    """
    Split AnnData into train, val, test sets.
    """
    idx = np.arange(adata.n_obs)
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size, random_state=random_state
    )

    adata_train = adata[train_idx].copy()
    adata_val = adata[val_idx].copy()
    adata_test = adata[test_idx].copy()

    return adata_train, adata_val, adata_test


def build_dataloader(
    adata: AnnData,
    encoders: Dict[str, Dict[str, Any]] = None,
    batch_size: int = 128,
    use_cuda: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """
    Wrap AnnData in a PyTorch DataLoader via AnnLoader.
    """
    dataloader = AnnLoader(
        adata,
        batch_size=batch_size,
        shuffle=shuffle,
        convert=encoders,
        use_cuda=use_cuda,
    )
    return dataloader


def get_dataloaders(
    adata: AnnData,
    batch_size: int = 128,
    test_size: float = 0.2,
    val_size: float = 0.1,
    use_cuda: bool = True,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return train, validation, and test dataloaders.
    """
    adata_train, adata_val, adata_test = split_adata(
        adata, test_size=test_size, val_size=val_size, random_state=random_state
    )

    train_loader = build_dataloader(
        adata_train,
        None,
        batch_size,
        use_cuda,
        shuffle=True,
    )
    val_loader = build_dataloader(
        adata_val,
        None,
        batch_size,
        use_cuda,
        shuffle=False,
    )
    test_loader = build_dataloader(
        adata_test,
        None,
        batch_size,
        use_cuda,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
