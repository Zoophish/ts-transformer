from typing import Type

import torch
from torch.distributions import Distribution, Normal

from scipy.stats import spearmanr

import pytorch_lightning as L

from ..models.transformer import DecoderTransformer
from ..models.probablistic_transformer import ProbablisticTransformer


class ProbablisticTransformerLightning(L.LightningModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        d_model: int,
        n_head: int,
        d_ff: int,
        n_layers: int,
        learning_rate: float,
        l2_lambda: float,
        dist_cls: Type[Distribution] = Normal,
        dropout_embed: float = 0.0,
        dropout_attn: float = 0.0,
        dropout_residual: float = 0.0,
        # concrete dropout params
        use_conc_dropout: bool = False,
        weight_reg: float = 1e-6,
        dropout_reg: float = 1e-4,
        # variational Bayes by backprop
        use_var_bayes: bool = False,
        kl_beta: float = 1e-4,
        use_stateful_dropout: bool = True
    ):
        super().__init__()

        self.save_hyperparameters()
        self.kl_beta = kl_beta

        self.model = ProbablisticTransformer(
            in_dim=in_dim,
            out_dim=out_dim,
            d_model=d_model,
            n_head=n_head,
            d_ff=d_ff,
            dropout_embed=dropout_embed,
            dropout_attn=dropout_attn,
            dropout_residual=dropout_residual,
            n_layers=n_layers,
            dist_cls=dist_cls,
            use_conc_dropout=use_conc_dropout,
            weight_reg=weight_reg,
            dropout_reg=dropout_reg,
            use_var_bayes=use_var_bayes,
            use_stateful_dropout=use_stateful_dropout
        )

    def reset_kv_cache(self):
        self.model.reset_kv_cache()

    def forward(
            self, x : torch.tensor,
            pad_mask : torch.Tensor = None,
            targets : torch.Tensor = None,
        ):
        return self.model(x, pad_mask, targets)
    
    def generate(
        self,
        context: torch.Tensor,
        horizon_len: int,
        pad_mask: torch.Tensor = None,
        use_mcd: bool = False
    ):
        return self.model.generate(
            context,
            horizon_len,
            pad_mask,
            use_mcd
        )
    
    def generate_bayes(
        self,
        context: torch.Tensor,
        horizon_len: int,
        pad_mask: torch.Tensor = None,
        mc_samples : int = 32,
        model_state_samples : int = 8,
        scramble_seed : int = 0,
        max_batch_size : int = 256,
        buffer_device = 'cpu'
    ):
        return self.model.generate_bayes(
            context=context,
            horizon_len=horizon_len,
            pad_mask=pad_mask,
            mc_samples=mc_samples,
            model_state_samples=model_state_samples,
            scramble_seed=scramble_seed,
            max_batch_size=max_batch_size,
            buffer_device=buffer_device
        )
    
    def compute_loss(self, batch):
        window_batch, pad_mask = batch
        # single step predictions
        X_batch = window_batch[:, :-1, ...]
        y_batch = window_batch[:, 1:, ...]
        pad_mask = pad_mask[:, :-1]
        # feed the model targets so it generates the NLL loss
        loss = self.model(X_batch, pad_mask, targets=y_batch)['loss']

        # add concrete dropout regularisation if enabled
        if hasattr(self.model, 'conc_dropout_modules'):
            for conc_dropout in self.model.conc_dropout_modules:
                loss += conc_dropout.regularisation()

        # add kl cost term if bayes is enabled
        if hasattr(self.model, 'bayes_units'):
            # NOTE: can technically batch this
            for bayes_unit in self.model.bayes_units:
                loss += self.kl_beta * bayes_unit.kl_cost()

        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("kl_beta", self.kl_beta, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.l2_lambda
        )
        return optimiser

    def uncertainty_calibration(model, test_loader):
        all_epistemic_vars = torch.zeros(len(test_loader))
        all_prediction_errors = torch.zeros(len(test_loader))
        model = model.to('cuda')

        for i, (context, y_true) in enumerate(test_loader):
            context, y_true = context.to('cuda'), y_true.to('cuda')
            y_buff, epistemic_var, _ = model.generate_mcd(
                context,
                horizon_len=1,
                mc_samples=8,
                model_state_samples=8
            )
            
            pred_mean = y_buff.mean(dim=0)

            avg_epistemic_var = epistemic_var.mean().cpu().item()
            all_epistemic_vars[i] = avg_epistemic_var

            error = torch.mean(torch.abs(pred_mean.cpu() - y_true.cpu()))
            all_prediction_errors[i] = error.item()
            print(f"{i} / {len(test_loader)}")

        correlation, p_value = spearmanr(all_epistemic_vars, all_prediction_errors)
        
        print(f"Spearman Rank Correlation between Epistemic Signal and MAE:")
        print(f"Rho: {correlation:.4f}")
        print(f"P-value: {p_value:.4f}")

        print(f"Epistemic Signal Statistics:")
        print(f"Mean {all_epistemic_vars.mean().item()}")
        print(f"Min {all_epistemic_vars.min().item()}")
        print(f"Max {all_epistemic_vars.max().item()}")
        
        return correlation
