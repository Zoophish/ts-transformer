import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class VariationalLinear(nn.Module):
    """
    Variational linear layer following the "Bayes by Backprop"
    paper: https://arxiv.org/abs/1505.05424.

    The local reparameterisation trick from (https://arxiv.org/abs/1506.02557)
    rather than te global one used in BBB as this typically results in lower
    variance and better training stability.

    Intended to be drop-in replacement for nn.Linear layers.
    The posterior is fixed to a 'diagonal guassian', but the prior
    can be changed to other torch distributions.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool = False,
            use_global_reparam: bool = False,
            eps: float = 1e-6
        ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.use_global_reparam = use_global_reparam
        self.eps = eps
        self.freeze = False
        self.generator = None

        # the prior is the 'belief'/ideal distribution for weights
        # most priors encourage smaller posterior centers and prevent the 'width' collapsing
        # a normal distribution prior results in L2 regularisation
        self.w_prior = D.Normal(0, 1)
        self.b_prior = D.Normal(0, 1) if self.bias else None
        
        # the variational posterior distribution type is the same for the weights and biases
        # fix it as diagonal gaussian (diagonal meaning independent)
        # this is because the local reparam trick only works for gaussian for now
        self._posterior_dist = D.Normal
        
        # the weight and bias parameters are treated as guassian distributions
        self.w_mu = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.w_rho = nn.Parameter(torch.Tensor(out_dim, in_dim))
        
        if self.bias:
            self.b_mu = nn.Parameter(torch.Tensor(out_dim))
            self.b_rho = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('b_mu', None)
            self.register_parameter('b_rho', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_mu, a=5**0.5)
        nn.init.constant_(self.w_rho, -7)  # NOTE: I have found that these are extremely important parameters for overall performance
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = fan_in**-0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.b_mu, -bound, bound)
            nn.init.constant_(self.b_rho, -5)  # NOTE: here too

    def forward(self, x : torch.Tensor):
        if self.freeze:
            return F.linear(x, self.w_mu, self.b_mu)
        else:
            if self.use_global_reparam:
                # NOTE: use Sobol samples to get better MC properties
                # ensure positivity in the rho parameters
                w_sigma = F.softplus(self.w_rho)
                b_sigma = F.softplus(self.b_rho) if self.bias else None

                # instantiate the actual variational posterior distributions
                w_q = self._posterior_dist(loc=self.w_mu, scale=w_sigma)
                b_q = self._posterior_dist(loc=self.b_mu, scale=b_sigma) if self.bias else None

                if self.generator is not None:
                    assert False, "Reminder to implement generator sampling method here"

                # sample the posterior but keep it differentiable
                # sample fixed noise then transform it - the 'reparameterisation trick'
                w_sampled = w_q.rsample()
                b_sampled = b_q.rsample() if self.bias else None

                # the 'global' reparam trick applies the same sampled w and b to the entire batch
                return F.linear(x, w_sampled, b_sampled)
            else:
                # the 'local' reparam trick effectively results in the noise applied per batch item
                # this is done simply by applying the noise after the linear transform
                # if you tried to generate unique weights/biases before the transformation, you'd need
                # a prohibitive amount of memory
                
                # mean of the affine transformation (standard deterministic path)
                affine_mu = F.linear(x, self.w_mu, self.b_mu)
                # variance of the affine transformation
                w_sigma = F.softplus(self.w_rho)
                # variance of weights (scaling rule): Var(W*x) = x^2 * Var(W)
                w_var = F.linear(x*x, w_sigma*w_sigma)

                if self.bias:
                    b_sigma = F.softplus(self.b_rho)
                    # variance from bias is independent of input x
                    b_var = b_sigma*b_sigma
                    # total var is sum of variances from weights and bias
                    affine_var = w_var + b_var
                else:
                    affine_var = w_var

                affine_std = torch.sqrt(affine_var + self.eps)
                # sample gaussian noise for the *entire batch* rather than per weight/bias
                epsilon = torch.empty_like(affine_std).normal_(generator=self.generator)
                return affine_mu + affine_std * epsilon

    def kl_cost(self):
        # instantiate the distributions again
        w_sigma = F.softplus(self.w_rho)
        b_sigma = F.softplus(self.b_rho) if self.bias else None
        w_q = self._posterior_dist(loc=self.w_mu, scale=w_sigma)
        b_q = self._posterior_dist(loc=self.b_mu, scale=b_sigma) if self.bias else None

        # KL divergence 'cost'
        w_kl = torch.sum(D.kl_divergence(w_q, self.w_prior))
        b_kl = torch.sum(D.kl_divergence(b_q, self.b_prior)) if self.bias else 0
        return w_kl + b_kl
