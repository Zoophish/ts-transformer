# Probablistic Time Series Forecasting using Transformers

## Highlights
- Causal decoder transformer architecture
- Distribution head driven predictions which can utilise arbitrary torch.distributions classes
- Pure PyTorch and Lightning modules available
- Basic time series dataset utilities

## Transformer Architecture
- Decoder only (GPT-style)
- RMSNorm pre-normalisation on FFN
- Rotary Positional Embeddings (RoPE)
- Accelerated scaled dot product attention via Torch Flash Attention implementation
- Key value caching

## To do:
- SwiGLU FFN
- MC dropout
- Instance normalisation
- Scheduled sampling

## Requirements
- PyTorch
- Lightning
- Numpy
- DearPyGUI (for explorer)

## Usage:

The `workbench.py` script can be run directly to try out the model on some basic synthetic time series.

An explorer app is available courtesy of the [DearPyGUI](https://github.com/hoffstadt/DearPyGui) library. Run `dpg_explorer.py` to open the GUI for testing the model on tabular data such as .csv or .parquet files.

![](./media/explorer_screenshot.jpg)
