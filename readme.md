# Probablistic Time Series Forecasting using Transformers

## Highlights
- Based on the causal decoder transformer architecture
- Probablistic predictions which can utilise arbitrary torch.distributions classes
- Pure PyTorch and Lightning modules available
- Basic time series dataset utilities
- Modular enough that architectural features can be hyperparameters

## Transformer Architecture
- Decoder only (GPT-style)
- Rotary Positional Embeddings (RoPE)
- Accelerated scaled dot product attention via Torch Flash Attention implementation
- RMSNorm pre-normalisation on FFN
- Gated Linear Unit FFN (SwiGLU/GeGLU)
- Key value caching


## Model Details
### Instance Normalisation
To try and address input scaling, each input sequence is standardised via an instance normalisation layer. The model learns to make predictions in the normalised space of the context. An inverse denormalisation step is applied to the output samples to transform back to the context space. At inference, it would be correct to refit the normalisation layer on the context and the models own predictons (fully autoregressive). However, doing so would invalidate the KV cache which is very useful for inference performance. Therefore, is it assumed the initial context is representative enough and normalisation statistics are only gathered from the initial context.

### Dropout
Dropout is applied to the residual sum after the attention unit, residual sum after the FFN and to the embedding layer. Each dropout can be changed manually.

## To do:
- Progressive batching
- Quasirandom sampling
- Common random numbers epistemic estimator
- Combined uncertainty estimator
- MC dropout
- Scheduled sampling

## Requirements
- PyTorch
- Lightning
- Numpy
- DearPyGUI (for explorer)

## Usage:

The `workbench.py` script can be run directly to try out the model on some basic synthetic time series.

The `dpg_explorer.py` script uses the [DearPyGUI](https://github.com/hoffstadt/DearPyGui) library for interactive testing on tabular data such as .csv or .parquet files.

![](./media/explorer_screenshot.jpg)
