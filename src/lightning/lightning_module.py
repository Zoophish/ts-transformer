from typing import Type

import torch
from torch.distributions import Distribution, Normal

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
            dist_cls=dist_cls
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
    
    def compute_loss(self, batch):
        window_batch, pad_mask = batch
        # single step predictions
        X_batch = window_batch[:, :-1, ...]
        y_batch = window_batch[:, 1:, ...]
        pad_mask = pad_mask[:, :-1]
        # feed the model targets so it generates the NLL loss 
        out = self.model(X_batch, pad_mask, targets=y_batch)
        return out['loss']
    
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
