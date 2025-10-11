import torch
import torch.nn as nn


class CausalINLayer(nn.Module):
    """
    Reversable Causal Instance Normalisation Layer.

    This layer can be used to normalise inputs along the sequence dimension of a
    time series, and subsequently denormalise predictions made in this space.

    In essence, allows y to be predicted in the standardised space of x.
    """
    def __init__(self, n_feat : int, eps : float = 1e-5, affine=True):
        super().__init__()
        self.n_feat = n_feat
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.n_feat))
            self.beta = nn.Parameter(torch.zeros(self.n_feat))

    def forward(self, x : torch.Tensor, fit : bool = True):
        """
        Assumes input shape is (B, L, C).
        """
        if fit:
            # create 'temporary' attributes, assuming mean and std will be kept for denormalisation
            # detach since these aren't learned and need to reuse them in denormalise
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            var = torch.var(x, dim=1, keepdim=True, unbiased=True).detach()
            self.std = torch.sqrt(var + self.eps*self.eps)

        if not hasattr(self, 'mean') or not hasattr(self, 'std'):
            raise RuntimeError("Requires call to forward with fit first.")
        
        # standardise the instance
        x = x - self.mean
        x = x / self.std
        # apply learnable scale and offset
        if self.affine:
            x = x * self.gamma
            x = x + self.beta
        return x
        
    def denormalise(self, x : torch.Tensor):
        if not hasattr(self, 'mean') or not hasattr(self, 'std'):
            raise RuntimeError("Requires call to forward (normalise) before denormalise.")
        
        if self.affine:
            x = x - self.beta
            x = x / (self.gamma + self.eps*self.eps)
        x = x * self.std
        x = x + self.mean
        return x
        