import torch
import torch.nn as nn


class Ensemble(nn.Module):
    def __init__(self, base : nn.Module, n : int = 1, **kwargs):
        super().__init__()
        self.base = base
        self.n = n

    