from typing import Type
import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal

from .causal_in import CausalINLayer
from .transformer import DecoderTransformer
from .distribution_head import DistributionHead


class ProbablisticTransformer(nn.Module):
    """
    Combines the decoder transformer and distribution head for probablistic
    autoregressive predictions.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        d_model: int,
        n_head: int,
        d_ff: int,
        dropout: float,
        n_layers: int,
        dist_cls: Type[Distribution] = Normal
    ):
        super().__init__()

        self.causal_inst_norm = CausalINLayer(
            n_feat=in_dim
        )
        self.model = DecoderTransformer(
            in_dim=in_dim,
            out_dim=d_ff,
            d_model=d_model,
            n_head=n_head,
            d_ff=d_ff,
            dropout=dropout,
            n_layers=n_layers
        )
        self.dist_head = DistributionHead(
            in_dim=d_ff,
            out_dim=out_dim,
            distr_cls=dist_cls
        )

    def reset_kv_cache(self):
        self.model.reset_kv_cache()

    def forward(
            self,
            x: torch.Tensor,
            pad_mask: torch.Tensor = None,
            targets: torch.Tensor = None
        ):
        """
        Predict the distribution of possible next steps.
        Returns a dict containing:
            - "dist": The distribution prediction.
            - "loss": The NLL loss, if targets are provided, else None.
        """
        # assume inference if no targets provided
        is_inference = targets is None
        # normalise x (done externally if using inference with kv caching)
        if not is_inference:
            x = self.causal_inst_norm(x, fit=True)
        # run transformer
        hidden_states = self.model(x, pad_mask, is_inference=is_inference)
        # normalise y to the statistics of x
        if targets is not None:
            targets = self.causal_inst_norm(targets, fit=False)
        # get distribution, and NLL loss if targets is set
        outputs = self.dist_head(hidden_states, targets)
        return outputs
    
    @torch.no_grad
    def generate(
        self,
        context: torch.Tensor,
        horizon_len: int,
        pad_mask: torch.Tensor = None
    ):
        n_batch, ctx_len, n_feat = context.shape
        y = torch.zeros((n_batch, horizon_len, n_feat), device=context.device)

        self.reset_kv_cache()

        X = self.causal_inst_norm(context, fit=True)
        for i in range(horizon_len):
            out = self.forward(X, pad_mask=pad_mask)
            sample = out['dist'].sample()[:, -1:, :]
            y[:, i:, :] = self.causal_inst_norm.denormalise(sample[:, :, :])
            X = sample
        
        return y
