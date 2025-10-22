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
To try and address input scaling, each input sequence is standardised via an instance normalisation layer. The model learns to make predictions in the normalised space of the context. An inverse denormalisation step is applied to the output samples to transform back to the context space. At inference, it would be correct to refit the normalisation layer on the context and the model's own predictons (i.e. fully autoregressive). However, doing so would invalidate the KV cache would significantly impact performance. Therefore, it is assumed that the normalisation statistics of the initial context are representative enough.

### Dropout
Dropout is applied to the residual sum after the attention unit, residual sum after the FFN, and to the embedding layer. Each dropout probability can be tuned manually or be entirely substituted with [Concrete Dropout](https://arxiv.org/abs/1705.07832). Concrete dropout attempts to optimise the dropout probabilties via backprop, but requires careful tuning of the regularisation parameters. If concrete dropout is enabled, the model resorts to using a slower custom scaled dot product attention implementation.

### Uncertainty
Predictions can be generated using an uncertainty decomposition method. [Monte Carlo Dropout](https://arxiv.org/abs/1506.02142) is employed due to its simplicity, but it must be noted that this is a pragmatic approximation rather than a true epistemological measurement. The separate aleatoric and epistemic components of uncertainty are estimated using a common (quasi)-random numbers method; i.e. the same points in the state space are sampled multiple times with different dropout masks, meaning the variation in the results are attributable to dropout.

Producing accurate epistemic uncertinaty estimates is difficult and requires careful calibration and validation. Ways of assessing the quality of the epistemic


## To do:
- MC epistemic regulariser
- Variational dropout
- Make the concrete transformer its own class?
- Scheduled sampling
- Adaptive minibatch selection
- Smarter tokenisation
- Ensembling (snapshot, weight inits, batch ensembling)

## Requirements
- PyTorch
- Lightning
- Numpy
- GluonTS (to use StudentT dist)
- DearPyGUI (for explorer)

## Usage:

The `workbench.py` script can be run directly to try out the model on some basic synthetic time series.

The `dpg_explorer.py` script uses the [DearPyGUI](https://github.com/hoffstadt/DearPyGui) library for interactive testing on tabular data such as .csv or .parquet files.

![](./media/explorer_screenshot.jpg)
