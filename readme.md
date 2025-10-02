# Probablistic Time Series Forecasting using Transformers

This repository contains the code for probablistic time series prediction using transformers.

The code is intended to be pedagogic, modular and performant rather than completely production ready.

## Highlights
- **Causal decoder**, transformer architecture
- The model predicts **distribution heads** rather than point estimates for probablistic autoregression
- The distribution head can inherit arbitrary torch.distributions classes
- Uncertainty can be estimated using Monte Carlo rollouts
- Everything is provided as a Torch module before being wrapped in Lightning, so either can be used
- Basic time series dataset utilities

## Transformer Architecture
- Decoder only transformer
- RMSNorm pre-normalisation on FFN
- Rotary Positional Embeddings (RoPE)
- Accellerated scaled dot product attention using Torch implementation

## To do:
- SwiGLU FFN
- KV caching
- MC dropout

## Usage:

As of now, you don't need to install the repo as a package, you can run `workbench.py` directly to try out the model on some basic synthetic time series.
