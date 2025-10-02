"""
Workbench is intended for quick ad hoc experiments using the lightning modules.
"""

import sys
import os
import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


if __name__ == '__main__':
    base_path = os.path.dirname(os.path.abspath(__file__))
    try:
        sys.path.append(base_path)
        from src.lightning.lightning_module import ProbablisticTransformerLightning
        from src.dataset.synthetic import sinusoidal
        from src.dataset.torch_datasets import IntervalDataset, collate_pad
        from src.dataset.utils import partition
    except ImportError:
        raise RuntimeError("Could not import local modules.")

    BATCH_SIZE = 64
    TRAIN_RATIO = 0.8
    MAX_EPOCHS = 100
    PATIENCE = 12
    MIN_LOSS_DELTA = 0.05

    LEARNING_RATE = 2e-4
    L2_LAMBDA = 0.0001

    TRAIN_CXT_SIZE = 128
    TEST_CXT_SIZE = 128

    synthetic_series_len = 1024 * 10
    time_series = sinusoidal(synthetic_series_len, 1, 0.04, 0, 0.1) + 1

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
        dropout=0.1,
        n_layers=3,
        learning_rate=LEARNING_RATE,
        l2_lambda=L2_LAMBDA,
        dist_cls=torch.distributions.StudentT
    )

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
        save_top_k=1,
        mode="min"
    )

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[early_stopper, checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)


    # ---- TEST PLOT ----
    best_model_path = checkpoint_callback.best_model_path
    model = ProbablisticTransformerLightning.load_from_checkpoint(best_model_path)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    context_len = 128
    horizon_len = 256
    X = torch.tensor(val_ts[0:context_len], dtype=torch.float32, device=device)
    y_true = val_ts[context_len:context_len + horizon_len]

    y = []
    with torch.no_grad():
        for i in range(horizon_len):
            out = model(X.unsqueeze(0).unsqueeze(-1))
            sample = out['dist'].sample()[0, -1, 0]
            y.append(sample.cpu())
            X = torch.cat([X, sample.unsqueeze(0)])

    plt.plot(y_true, label='Target')
    plt.plot(y, label='Prediction')
    plt.legend()
    plt.show()
