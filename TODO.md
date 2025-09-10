# TODO

## Paper

- Main idea: split the latent representation into two parts, the invariant and the spurious embeddings.

Content

- pages 1-5: intro
- pages 5-15: results
- pages 16: discussion about room for improvement and interesting avenues for next papers
- pages 18-20: inVAE method
    - VAE + penalization loss to enforce independence between the latent variables
    - loss ELBO, KL div coefficient is linearly increased during training during warmup
- pages 20-xx: hyper param
    - model: MLP, 2 layers, hidden dim 128, Relu, batch norm + dropout 0.1
    - latent dim 30 and 5
    - prior is set to gaussian, with learnable variace and zero mean
    - batch size 256
    - the decoder models the observed data using a Negative Binomial distribution
- Post training analysis
    - Feature importance: train a RF in latents, then asses importance by using permutation (n=10)
    - Benchmarking
        - Silhouette Score
        - Adjusted Rand Index (ARI)
        - Normalized Mutual Information (NMI)
        - Batch Effect Metrics (kBET)
        - UMAP/t-SNE

## Technologies/Tools
