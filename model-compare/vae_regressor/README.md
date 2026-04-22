# VAE Regressor Baseline

The code lives in `model_compare`. This directory is kept as the artifact namespace:
trained runs are written to `artifacts/vae_regressor/runs`.

This is a lightweight generative-representation baseline: each frame is encoded by a VAE,
the latent sequence is summarized by a GRU, and the head predicts a Gaussian CSI distribution.

