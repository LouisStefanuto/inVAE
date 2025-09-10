from pathlib import Path

import scanpy as sc


def main() -> None:
    # 1. Load and explore
    adata = sc.read(
        Path(__file__).parent.parent.parent / "data/pancreas.h5ad",
        backup_url="https://www.dropbox.com/s/qj1jlm9w10wmt0u/pancreas.h5ad?dl=1",
    )

    print(adata.obs.columns)
    print(f"Shape: {adata.shape}")
    print(f"Cell types: {adata.obs.celltype.value_counts()}")

    # 2. Standard preprocessing of data
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    print(f"Shape: {adata.shape}")
    print(adata.X)


if __name__ == "__main__":
    main()
