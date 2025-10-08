"""
Workbench is intended for quick ad hoc experiments using the lightning modules.
"""

import sys
import os
import torch
import torch.distributions.constraints as constraints
import numpy as np
import pandas as pd
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    base_path = os.path.dirname(os.path.abspath(__file__))
    try:
        sys.path.append(base_path)
        from src.lightning.lightning_module import ProbablisticTransformerLightning
        from src.dataset.synthetic import sinusoidal, brownian_series
        from src.dataset.torch_datasets import IntervalDataset, collate_pad
        from src.dataset.utils import partition
    except ImportError:
        raise RuntimeError("Could not import local modules.")

    BATCH_SIZE = 64
    TRAIN_RATIO = 0.8
    MAX_EPOCHS = -1
    PATIENCE = 16
    MIN_LOSS_DELTA = 0.025

    LEARNING_RATE = 2e-4
    L2_LAMBDA = 1e-4

    TRAIN_CXT_SIZE = 256
    TEST_CXT_SIZE = 256
    MODE = {'train', 'plot'}

    series_len = 1024 * 20
    time_series = sinusoidal(series_len, 1, 0.04, 0, 0.1) + 1
    df = pd.DataFrame(time_series)
    df.to_parquet('sinusoidal.parquet')

    # plt.plot(time_series); plt.show()

    train_ts, val_ts = partition(time_series, TRAIN_RATIO)

    train_dataset = IntervalDataset(train_ts, interval_length=TRAIN_CXT_SIZE)
    val_dataset = IntervalDataset(val_ts, interval_length=TEST_CXT_SIZE)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_pad)

    model = ProbablisticTransformerLightning(
        in_dim=1,
        out_dim=1,
        d_model=256,
        n_head=4,
        d_ff=512,
        dropout=0.3,
        n_layers=4,
        learning_rate=LEARNING_RATE,
        l2_lambda=L2_LAMBDA,
        dist_cls=torch.distributions.StudentT
    )

    # manually limit the T distribution's degrees of freedom
    model.model.dist_head.const_overrides['df'] = constraints.greater_than(lower_bound=2.1)

    early_stopper = EarlyStopping(
        monitor="val_loss",
        min_delta=MIN_LOSS_DELTA,
        patience=PATIENCE,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=base_path + r'/checkpoints',
        filename='best',
        save_top_k=1,
        mode="min",
        enable_version_counter=False
    )

    if 'train' in MODE:
        trainer = L.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="auto",
            devices=1,
            callbacks=[early_stopper, checkpoint_callback],
            gradient_clip_val=1.0
        )
        trainer.fit(model, train_loader, val_loader)


    if 'plot' in MODE:
        model = ProbablisticTransformerLightning.load_from_checkpoint(base_path + r'/checkpoints/best.ckpt')

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        context_len = 256
        horizon_len = 256
        mc_samples = 16
        n_feat = 1
        X = torch.tensor(val_ts[0:context_len], dtype=torch.float32, device=device)
        X = X.repeat(mc_samples, 1).unsqueeze(-1)
        y_true = val_ts[context_len:context_len + horizon_len]

        y = model.generate(X, horizon_len).cpu()

        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        q_values = np.quantile(y[:, :, 0], q=quantiles, axis=0)
        t = np.arange(horizon_len)
        plt.fill_between(t, q_values[0], q_values[4], color='red', alpha=0.3, label='90% CI')
        plt.fill_between(t, q_values[1], q_values[3], color='red', alpha=0.5, label='50% CI (IQR)')
        plt.plot(t, q_values[2], color='red', lw=1, label='Median (50th)')

        plt.plot(t, y_true, label='Target')
        plt.legend()
        plt.show()
