# Probablistic Time Series Forecasting using Transformers

## Highlights
- Causal decoder transformer architecture
- Distribution head driven predictions, rather than point estimates
- The distribution heads can utilise arbitrary torch.distributions classes
- Uncertainty can be estimated using Monte Carlo rollouts
- Everything is provided as a Torch module before being wrapped in Lightning, so either can be used
- Basic time series dataset utilities

## Transformer Architecture
- Decoder only transformer
- RMSNorm pre-normalisation on FFN
- Rotary Positional Embeddings (RoPE)
- Accellerated scaled dot product attention using Torch Flash Attention implementation
- Key, value caching

## To do:
- SwiGLU FFN
- MC dropout

## Usage:

The `workbench.py` script can be run directly to try out the model on some basic synthetic time series.
