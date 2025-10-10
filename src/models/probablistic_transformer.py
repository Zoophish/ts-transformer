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
        n_layers: int,
        dropout_embed: float,
        dropout_attn: float,
        dropout_residual: float,
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
            n_layers=n_layers,
            dropout_embed=dropout_embed,
            dropout_attn=dropout_attn,
            dropout_residual=dropout_residual,
        )
        self.dist_head = DistributionHead(
            in_dim=d_ff,
            out_dim=out_dim,
            distr_cls=dist_cls
        )

    def reset_kv_cache(self):
        self.model.reset_kv_cache()

    def toggle_dropout(self, state=False):
        """
        Reactivates dropout for inference. Must be called after calls to eval().
        """
        func_attr = 'train' if state else 'eval'
        getattr(self.model.embed_dropout, func_attr)()
        for decoder_block in self.model.decoder_blocks:
            decoder_block.attention.force_dropout = state
            getattr(decoder_block.dropout, func_attr)()

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
        pad_mask: torch.Tensor = None,
        use_mcd: bool = False
    ):
        batch_size, ctx_len, n_feat = context.shape
        y = torch.zeros((batch_size, horizon_len, n_feat), device=context.device)

        self.toggle_dropout(use_mcd)
        self.reset_kv_cache()

        X = self.causal_inst_norm(context, fit=True)
        for t in range(horizon_len):
            out = self.forward(X, pad_mask=pad_mask)
            sample = out['dist'].sample()[:, -1:, :]
            y[:, t:, :] = self.causal_inst_norm.denormalise(sample[:, :, :])
            X = sample
        
        return y
    

    @torch.no_grad
    def generate_mcd(
        self,
        context: torch.Tensor,
        horizon_len: int,
        pad_mask: torch.Tensor = None,
        mc_samples : int = 32,
        dropout_samples : int = 8,
        max_batch_size : int = 256,
        buffer_device = 'cpu'
    ):
        """
        Uses common random numbers method for estimating aleatoric and epistemic uncertainty
        separately using Monte Carlo rollouts and dropout masks.
        """
        mc_samples, ctx_len, n_feat = context.shape
        y = torch.zeros((mc_samples * dropout_samples, horizon_len, n_feat), device=buffer_device)

        self.toggle_dropout(True)
        self.reset_kv_cache()
        
        # calculate work sizes to fully utilise max_batch_size
        total_work_size = mc_samples * dropout_samples
        work_group_size = min(max_batch_size, total_work_size)
        n_work_groups = total_work_size // work_group_size
        remainder_group_size = total_work_size % work_group_size

        # deterministic samples in [0, 1] (Sobol has good MC properties)
        sampler = torch.quasirandom.SobolEngine(dimension=mc_samples, scramble=True, seed=0)

        dimension = horizon_len * n_feat
        sampler = torch.quasirandom.SobolEngine(dimension=dimension, scramble=True)
        uniform_samples = sampler.draw(mc_samples).to(context.device)
        uniform_samples = uniform_samples.reshape(mc_samples, horizon_len, n_feat)

        for i in range(n_work_groups):
            X = self.causal_inst_norm(context, fit=True)
            for t in range(horizon_len):
                out = self.forward(X, pad_mask=pad_mask)
                # draw sample from Sobol sequence
                uniform_sample = sampler.draw()
                # use inverse CDF method to sample distribution
                sample = out['dist'].icdf(uniform_sample)[:, -1:, :]
                y[:, t:, :] = self.causal_inst_norm.denormalise(sample[:, :, :])
                X = sample
            self.reset_kv_cache()
            sampler.reset()

        return y
