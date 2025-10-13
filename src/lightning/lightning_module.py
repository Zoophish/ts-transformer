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
        dropout_embed: float,
        dropout_attn: float,
        dropout_residual: float,
        n_layers: int,
        learning_rate: float,
        l2_lambda: float,
        dist_cls: Type[Distribution] = Normal,
        use_conc_dropout: bool = False
    ):
        super().__init__()

        self.save_hyperparameters()

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
            use_conc_dropout=use_conc_dropout
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
    
    def generate_mcd(
        self,
        context: torch.Tensor,
        horizon_len: int,
        pad_mask: torch.Tensor = None,
        mc_samples : int = 32,
        crn_samples : int = 8,
        max_batch_size : int = 256,
        buffer_device = 'cpu'
    ):
        return self.model.generate_mcd(
            context=context,
            horizon_len=horizon_len,
            pad_mask=pad_mask,
            mc_samples=mc_samples,
            crn_samples=crn_samples,
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

        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
        all_epistemic_ratios = torch.zeros(len(test_loader))
        all_prediction_errors = torch.zeros(len(test_loader))
        model = model.to('cuda')

        for i, (context, y_true) in enumerate(test_loader):
            context, y_true = context.to('cuda'), y_true.to('cuda')
            y_buff, epistemic_ratio = model.generate_mcd(
                context,
                horizon_len=1,
                mc_samples=8,
                crn_samples=8
            )
            
            pred_mean = y_buff.mean(dim=0)

            avg_epistemic_ratio = epistemic_ratio.mean().cpu().item()
            all_epistemic_ratios[i] = avg_epistemic_ratio

            error = torch.mean(torch.abs(pred_mean.cpu() - y_true.cpu()))
            all_prediction_errors[i] = error.item()
            print(f"{i} / {len(test_loader)}")

        correlation, p_value = spearmanr(all_epistemic_ratios, all_prediction_errors)
        
        print(f"Spearman Rank Correlation between Epistemic Signal and MAE:")
        print(f"Rho: {correlation:.4f}")
        print(f"P-value: {p_value:.4f}")

        print(f"Epistemic Signal Statistics:")
        print(f"Mean {all_epistemic_ratios.mean().item()}")
        print(f"Min {all_epistemic_ratios.min().item()}")
        print(f"Max {all_epistemic_ratios.min().item()}")
        
        return correlation
