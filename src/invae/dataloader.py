from typing import Any, Dict

import scanpy as sc
from anndata import AnnData
from anndata.experimental.pytorch import AnnLoader
from torch.utils.data import DataLoader


def load_pancreas_data(filename: str, backup_url: str) -> AnnData:
    # ) -> Tuple[AnnData, Dict[str, Dict[str, Any]]]:

    adata = sc.read(filename, backup_url=backup_url)
    # adata.X = adata.raw.X  # use raw counts
    # adata.obs["size_factors"] = adata.X.sum(1)

    # # encoders
    # encoder_study = OneHotEncoder(sparse=False, dtype=np.float32)
    # encoder_study.fit(adata.obs["study"].to_numpy()[:, None])

    # encoder_celltype = LabelEncoder()
    # encoder_celltype.fit(adata.obs["cell_type"])

    # encoders = {
    #     "obs": {
    #         "study": lambda s: encoder_study.transform(s.to_numpy()[:, None]),
    #         "cell_type": encoder_celltype.transform,
    #     }
    # }
    # return adata, encoders
    return adata


def build_dataloader(
    adata: AnnData,
    encoders: Dict[str, Dict[str, Any]],
    batch_size: int = 128,
    use_cuda: bool = True,
) -> DataLoader:
    dataloader = AnnLoader(
        adata,
        batch_size=batch_size,
        shuffle=True,
        # convert=encoders,
        use_cuda=use_cuda,
    )
    return dataloader
