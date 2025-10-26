# Probablistic Time Series Forecasting using Transformers

## Highlights
- Based on the causal decoder transformer architecture
- Probablistic predictions which can utilise arbitrary torch.distributions classes
- "Drop-in" variational Baysian methods for epistemic uncertainty estimation
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
Predictions can be generated using an variational bayes and uncertainty decomposition method. The options available for variational bayes are [Monte Carlo Dropout](https://arxiv.org/abs/1506.02142) and [Bayes by Backprop](https://arxiv.org/abs/1505.05424).

Monte Carlo Dropout is a pragmatic approximation rather than a true bayesian method. There is effectively no performance penalty, since it uses existing dropout layers. Either standard dropout or concrete dropout can be used for this method.

Bayes by Backprop imposes actual distributions over the learnable parameters, which is a better variational bayes approximation. It uses more memory and is slower as it produces twice the number of parameters.

Aleatoric uncertainty is estimated by sampling the horizon space many times like a Monte Carlo simulation. The epistemic component of uncertainty is estimated by sampling across different model states; in other words, samples from the effective variational posterior of the model. The epistemic uncertainty can then be estimated as the variance in the outputs across different model states. Common Sobol' sequences are used to improve the estimate variance.


## To do:
- MC epistemic regulariser (M>1)
- Variational dropout
- Better param initialisation
- Stateful QRN and PRN generators with batching
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
