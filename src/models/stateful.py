import torch
import torch.nn as nn


class StatefulModule(nn.Module):
    """
    A module class that references a generator for deterministic stochastic
    sampling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = None

    def set_generator(self, generator : torch.Generator):
        """Sets this module's generator and all stateful modules within it."""
        for child in self.modules():
            if isinstance(child, StatefulModule):
                child.generator = generator


class StatefulDropout(nn.Dropout, StatefulModule):
    """
    An extension of regular dropout that enables stateful (deterministic)
    sampling using a random number generator.

    Behaves like regular dropout if generator=None.

    This is useful for variational bayesian methods where you want to sample
    the model in the same state multiple times.
    """
    def __init__(self, p = 0.5, inplace = False):
        super().__init__(p, inplace)

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
