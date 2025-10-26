import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConcreteDropout(nn.Module):
    """
    Concrete Dropout layer based on https://arxiv.org/abs/1705.07832.

    Can be used as a drop-in replacement for nn.Dropout or StatefulDropout.

    Original paper adds a regularisation term to the loss to balance
    generalisation.
    - dropout_regularizer is analogous to 1 / (τN)
    - weight_regularizer is analogous to l^2 / (τN)
    """
    def __init__(
            self,
            weight_regularizer=1e-6,
            dropout_regularizer=1e-5,
            init_min=0.1,
            init_max=0.5,
            eps=1e-7
        ):
        super(ConcreteDropout, self).__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        init_min = math.log(init_min) - math.log(1. - init_min)
        init_max = math.log(init_max) - math.log(1. - init_max)
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.temp = 0.1
        self.eps = eps

        self.generator = None

    @property
    def p(self):
        return torch.sigmoid(self.p_logit).detach().item()

    def forward(self, x : torch.Tensor):
        if not self.training:
            return x
        
        p = torch.sigmoid(self.p_logit)

        unif_noise = torch.empty_like(x).uniform_(generator=self.generator)
        
        drop_prob = (
            torch.log(p + self.eps)
            - torch.log(1. - p + self.eps)
            + torch.log(unif_noise + self.eps)
            - torch.log(1. - unif_noise + self.eps)
        )
        
        drop_mask = torch.sigmoid(drop_prob / self.temp)
        random_tensor = 1. - drop_mask
        retain_prob = 1. - p

        return (x * random_tensor) / (retain_prob + self.eps)

    def regularisation(self):
        p = torch.sigmoid(self.p_logit)
        weight_reg = self.weight_regularizer * torch.sum(p)
        dropout_reg = self.dropout_regularizer * torch.sum(-p * torch.log(p + self.eps) - (1-p) * torch.log(1-p + self.eps))
        return weight_reg + dropout_reg
