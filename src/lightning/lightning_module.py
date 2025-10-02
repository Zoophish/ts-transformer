from typing import Type

import torch
from torch.distributions import Distribution, Normal

import pytorch_lightning as L

from ..models.transformer import DecoderTransformer
from ..models.probablistic_transformer import ProbablisticTransformer


class ProbablisticTSTransformer(L.LightningModule):
    """
    Lightning wrapper for the DecoderTransformer.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        d_model: int,
        n_head: int,
        d_ff: int,
        dropout: float,
        n_layers: int,
        learning_rate: float,
        l2_lambda: float
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = DecoderTransformer(
            in_dim=in_dim,
            out_dim=out_dim,
            d_model=d_model,
            n_head=n_head,
            d_ff=d_ff,
            dropout=dropout,
            n_layers=n_layers
        )

    def forward(self, x : torch.tensor, pad_mask=None):
        return self.model(x, pad_mask)

    def _compute_nll_loss(self, batch):
        window_batch, pad_mask = batch

        # single step predictions
        X_batch = window_batch[:, :-1, ...]
        y_batch = window_batch[:, 1:, ...]
        pad_mask = pad_mask[:, :-1]

        out = self.model(X_batch, pad_mask)

        # output normal distribution
        mu = out[:, :, 0]
        log_var = out[:, :, 1]
        sigma2 = torch.clamp(torch.exp(log_var), min=1e-6)

        # negative log likelihood
        nll = 0.5*log_var + (y_batch[:, :, 0] - mu)**2 / (2*sigma2)
        return nll.mean()
    
    def training_step(self, batch, batch_idx):
        loss = self._compute_nll_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_nll_loss(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.l2_lambda
        )
        return optimiser


class ProbablisticTransformerLightning(L.LightningModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        d_model: int,
        n_head: int,
        d_ff: int,
        dropout: float,
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
            dropout=dropout,
            n_layers=n_layers,
            dist_cls=dist_cls
        )

    def forward(
            self, x : torch.tensor,
            pad_mask : torch.Tensor = None,
            targets : torch.Tensor = None
        ):
        return self.model(x, pad_mask, targets)
    
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