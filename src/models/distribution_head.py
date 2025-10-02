from typing import Type
import torch
import torch.nn as nn
from torch.distributions import Normal, Distribution, constraints
import torch.nn.functional as F


class NormalDistributionHead(nn.Module):
    """
    Maps hidden states to a torch normal distribution. Mainly for clarity
    since DistributionHead generalises over more distributions.
    """
    def __init__(
            self,
            in_dim : int,
            out_dim : int = 1,
            eps : float = 1e-6
        ):
        super().__init__()
        self.mu_layer = nn.Linear(in_dim, out_dim)
        self.sigma_layer = nn.Linear(in_dim, out_dim)
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, targets: torch.Tensor = None):
        mu = self.mu_layer(hidden_states)
        sigma = F.softplus(self.sigma_layer(hidden_states)) + self.eps
        pred_dist = Normal(loc=mu, scale=sigma)
        
        loss = None
        if targets is not None:
            loss = -pred_dist.log_prob(targets).mean()

        return {"dist": pred_dist, "loss": loss}


class DistributionHead(nn.Module):
    """
    A general-purpose distribution head that maps hidden states to the parameters
    of a specified torch.distributions class.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        distr_cls: Type[Distribution] = Normal,
        eps: float = 1e-6
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.distr_cls = distr_cls
        self.eps = eps

        # discover parameter names and constraints from the distribution class
        self.param_names = list(self.distr_cls.arg_constraints.keys())
        
        # create a projection layer for each parameter
        self.param_proj = nn.ModuleDict({
            name: nn.Linear(in_dim, out_dim) for name in self.param_names
        })

    def _apply_constraints(self, raw_params: dict) -> dict:
        """
        Applies transformations to the raw network outputs to satisfy
        the distribution's parameter constraints.
        """
        constrained_params = {}
        for name, value in raw_params.items():
            constraint = self.distr_cls.arg_constraints[name]
            if isinstance(constraint, constraints._PositiveDefinite):
                constrained_params[name] = F.softplus(value) + self.eps
            elif isinstance(constraint, constraints._GreaterThan):
                if constraint.lower_bound == 0:
                    constrained_params[name] = F.softplus(value) + self.eps
                else:
                    constrained_params[name] = constraint.lower_bound + F.softplus(value) + self.eps
            else:
                # unconstrained params
                constrained_params[name] = value
        return constrained_params

    def forward(self, hidden_states: torch.Tensor, targets: torch.Tensor = None):
        # project hidden states to raw parameter values
        raw_params = {
            name: self.param_proj[name](hidden_states)
            for name in self.param_names
        }
        # apply constraint transformations
        distr_args = self._apply_constraints(raw_params)
        
        # instantiate distribution
        try:
            pred_dist = self.distr_cls(**distr_args)
        except ValueError as e:
            print(f"Error instantiating {self.distr_cls.__name__} with args:")
            for name, val in distr_args.items():
                print(f"  {name}: shape={val.shape}, min={val.min().item()}, max={val.max().item()}")
            raise e
        
        loss = None
        if targets is not None:
            loss = -pred_dist.log_prob(targets).mean()

        return {"dist": pred_dist, "loss": loss}