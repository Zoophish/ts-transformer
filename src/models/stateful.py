import torch
import torch.nn as nn


class StatefulDropout(nn.Dropout):
    """
    An extension of regular dropout that enables stateful (deterministic)
    sampling using a random number generator.

    Behaves like regular dropout if generator=None.

    This is useful for variational bayesian methods where you want to sample
    the model in the same state multiple times.
    """
    def __init__(self, p = 0.5, inplace = False):
        super().__init__(p, inplace)
        self.generator = None

    def forward(
            self,
            x : torch.Tensor,
        ):
        if not self.training:
            return x
        elif self.generator is None:
            return nn.functional.dropout(x, self.p, True, self.inplace)
        else:
            keep_prob = 1 - self.p
            mask = torch.bernoulli(
                torch.full_like(x, fill_value=keep_prob),
                generator=self.generator
            )
            return x * mask / keep_prob
