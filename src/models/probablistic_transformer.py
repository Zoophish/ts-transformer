from typing import Type
import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal

from .inst_norm import CausalINLayer
from .transformer import DecoderTransformer
from .distribution_head import DistributionHead
from .stateful import StatefulDropout
from .concrete_dropout import ConcreteDropout
from .variational_linear import VariationalLinear


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
        dist_cls: Type[Distribution] = Normal,
        use_conc_dropout: bool = False,
        weight_reg: float = 1e-6,
        dropout_reg: float = 1e-4,
        use_var_bayes: bool = False,
        use_stateful_dropout: bool = False
    ):
        super().__init__()

        self.causal_inst_norm = CausalINLayer(
            n_feat=in_dim
        )
        is_stateful = use_conc_dropout or use_var_bayes or use_stateful_dropout
        attn_imp = 'internal' if is_stateful else 'torch'
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
            attn_imp=attn_imp
        )
        self.dist_head = DistributionHead(
            in_dim=d_ff,
            out_dim=out_dim,
            distr_cls=dist_cls
        )

        if use_conc_dropout:
            self.conc_dropout_modules = []
            self._drop_in_concrete_dropout(self, weight_reg, dropout_reg)
        elif use_stateful_dropout:
            self._drop_in_stateful_dropout(self)

        if use_var_bayes:
            self.bayes_units = []
            self._drop_in_variational_bayes(self)

        self.stateful_modules = []
        self._register_stateful_modules(self)
            
    def reset_kv_cache(self):
        self.model.reset_kv_cache()

    def toggle_dropout(self, state=False):
        func_attr = 'train' if state else 'eval'
        getattr(self.model.embed_dropout, func_attr)()
        for decoder_block in self.model.decoder_blocks:
            decoder_block.attention.force_dropout = state
            getattr(decoder_block.attention.weight_dropout_layer, func_attr)()
            getattr(decoder_block.dropout, func_attr)()

    def set_generator(self, generator : torch.Generator):
        for stateful_mod in self.stateful_modules:
            stateful_mod.generator = generator

    def _register_stateful_modules(self, module : nn.Module):
        stateful_mod_types = {
            StatefulDropout,
            ConcreteDropout,
            VariationalLinear
        }
        for child in module.children():
            if type(child) in stateful_mod_types:
                self.stateful_modules.append(child)
            else:
                self._register_stateful_modules(child)

    def _drop_in_stateful_dropout(self, module : nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.Dropout):
                new_module = StatefulDropout(child.p)
                setattr(module, name, new_module)
            else:
                self._drop_in_stateful_dropout(child)

    def _drop_in_concrete_dropout(
            self,
            module : nn.Module,
            weight_reg : float,
            dropout_reg : float
        ):
        for name, child in module.named_children():
            if isinstance(child, nn.Dropout):
                new_module = ConcreteDropout(weight_reg, dropout_reg)
                setattr(module, name, new_module)
                self.conc_dropout_modules.append(new_module)
            else:
                self._drop_in_concrete_dropout(child, weight_reg, dropout_reg)
    
    def _drop_in_variational_bayes(self, module : nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                new_module = VariationalLinear(
                    in_dim=child.in_features,
                    out_dim=child.out_features,
                    bias=True if child.bias is not None else False
                )
                setattr(module, name, new_module)
                self.bayes_units.append(new_module)
            else:
                self._drop_in_variational_bayes(child)

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
        output_dict = self.dist_head(hidden_states, targets)
        return output_dict
    
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
        model_state_samples : int = 8,
        max_batch_size : int = 256,
        buffer_device = 'cpu'
    ):
        """
        Generates predictions with decomposed aleatoric and epistemic uncertainty using MC rollouts
        with common random numbers, and under different model states (variational bayes sampling).

        By keeping the random numbers fixed for each of the mc_samples paths across all dropout_samples
        worlds, the difference observed is almost entirely attributable to the change in the dropout mask,
        not the randomness of the sampling path. This is the basis for estimating the epistemic uncertainty.
        """
        in_batch_size, ctx_len, n_feat = context.shape
        if in_batch_size != 1:
            raise ValueError("Batch dimension of context should be of size 1.")

        # calculate work size
        total_rollouts = mc_samples * model_state_samples

        # deterministic samples in [0, 1] (Sobol has good MC properties)
        dimension = horizon_len * n_feat
        sampler = torch.quasirandom.SobolEngine(dimension, scramble=True)
        # draw mc_samples from Sobol (common across each mc_samples chunk)
        uniform_samples = sampler.draw(mc_samples).to(context.device)
        uniform_samples = uniform_samples.view(mc_samples, horizon_len, n_feat)

        self.toggle_dropout(True)

        y_buff = torch.zeros((total_rollouts, horizon_len, n_feat), device=buffer_device)
        # fit instance norm layer once
        norm_context = self.causal_inst_norm(context.clone(), fit=True)

        # NOTE: force this for now to make stateful model easier to implement on batches
        max_batch_size = mc_samples
        generator_seed = 0
        generator = torch.Generator(context.device)
        self.set_generator(generator)

        # perform full autoregressive rollout per sub batch
        for start_idx in range(0, total_rollouts, max_batch_size):
            end_idx = min(start_idx + max_batch_size, total_rollouts)
            current_batch_size = end_idx - start_idx

            # prepare the sub batch
            sub_batch_context = norm_context.expand(current_batch_size, -1, -1)
            x_t = sub_batch_context

            # perform rollout over sub batch
            self.reset_kv_cache()
            for t in range(horizon_len):
                # ensure the model state is teh same for all timesteps
                generator.manual_seed(generator_seed)

                out = self.forward(x_t)
                # get samples for sub batch and step (wrap indexing to tile the 0th dim)
                wrap_indices = torch.arange(start_idx, end_idx) % mc_samples
                samples_t = uniform_samples[wrap_indices, t:t+1, :]

                # use inverse CDF method to sample distribution
                norm_next_x_t = out['dist'].icdf(samples_t)[:, -1:, :]

                # store result in output buffer
                next_x_t = self.causal_inst_norm.denormalise(norm_next_x_t)
                y_buff[start_idx:end_idx, t:t+1, :] = next_x_t.to(buffer_device)

                # next step
                x_t = norm_next_x_t
            
            # set the model state
            generator_seed += 1

        y_buff = y_buff.reshape(model_state_samples, mc_samples, horizon_len, -1)
        # determine variance across common Sobol samples
        mc_means = y_buff.mean(dim=1)
        epistemic_var = mc_means.var(dim=0)

        var_per_model = y_buff.var(dim=0)
        # since each monte carlo rollout uses the *same samples*, but different state,
        # the 'average' aleatoric prediction is the average over each state
        mean_aleatoric_rollout = y_buff.mean(dim=0)
        # we now have the 'average' mc_samples output, so we get quantiles from this
        # notice that this is the same as the mean of both dim 0 and 1
        aleatoric_mean = mean_aleatoric_rollout.mean(dim=0)
        aleatoric_var = mean_aleatoric_rollout.var(dim=0)

        # total var = aleatoric var + epsitemic
        # we want to plot total quantiles (tangled aleatoric and epistemic)
        # what we want to show is that for each timestep quantile, how much of that region
        # is uncertain due to epistemic vs aleatoric


        # y_buff = y_buff.reshape(model_state_samples * mc_samples, horizon_len, -1)

        # total_var = y_buff.var(dim=0)
        # epistemic_ratio = epistemic_var / total_var

        self.toggle_dropout(False)
        return y_buff #, epistemic_var, aleatoric_var
