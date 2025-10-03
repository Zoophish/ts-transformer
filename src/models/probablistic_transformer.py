from typing import Type
import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal

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

    def forward(
            self,
            x: torch.Tensor,
            pad_mask: torch.Tensor = None,
            targets: torch.Tensor = None,
            is_inference: bool = False
        ):
        """
        Predict the distribution of possible next steps.
        Returns a dict containing:
            - "dist": The distribution prediction.
            - "loss": The NLL loss, if targets are provided, else None.
        """
        hidden_states = self.model(x, pad_mask, is_inference=is_inference)
        outputs = self.dist_head(hidden_states, targets)
        return outputs
    
    @torch.no_grad
    def generate(
        self,
        context: torch.Tensor,
        horizon_len: int,
        num_samples: int,
        pad_mask: torch.Tensor = None
    ):
        ...
