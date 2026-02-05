from dataclasses import dataclass

import numpy as np
import torch
import polars as pl


def partition(iterable, ratio, split_gap=0):
    """
    Split the iterable into two proportionally to ratio. Optionally ensure
    an aboslute gap ahead of the split point to prevent window overlapping.
    
    Args:
        iterable (Iterable): The iterable to split
        ratio (float): The split ratio
        split_gap (int): The index gap between the two splits
    
    Returns:
        (Iterable, Iterable): The split segments
    """
    return (
        iterable[:int(len(iterable)*ratio) - 1],
        iterable[int(len(iterable)*ratio) - 1 + split_gap:]
    )


@dataclass
class TSBatch:
    """
    Container for time series tensors, since special treatment is required
    for the time encodings.
    """
   # [batch, seq, n_feat]
    channels: torch.Tensor
    
    # [batch, seq, time_feat]
    time_encoding: torch.Tensor

    def to(self, device):        
        return TSBatch(
            channels=self.channels.to(device),
            time_encoding=self.time_encoding.to(device)
        )
    
    def __len__(self):
        return self.channels.size(1)
        
    def __getitem__(self, key):
        if isinstance(key, slice):
            return TSBatch(
                channels=self.channels[:, key, :],
                time_encoding=self.time_encoding[:, key, :]
            )


class TimeSeriesDataFrameProcessor:
    """
    Maps raw tabular (Polars) data to embeddable numpy arrays. Generates
    discrete time encodings from the timestamp column.
    """

    def __init__(
            self,
            target_cols : set[str],
            timestamp_col : str,
            time_encodings : set[str] = {'minute-of-hour', 'hour-of-day', 'day-of-week', 'day-of-month', 'month-of-year'}
        ):
        self.target_cols = target_cols
        self.timestamp_col = timestamp_col
        self.time_features = time_encodings

    def process(self, df : pl.DataFrame):
        missing = [c for c in self.target_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing target columns: {missing}")

        time_exprs = []
        for tfeat in self.time_features:
            match tfeat:
                case 'minute-of-hour':
                    expr = pl.col(self.time_col).dt.minute()
                case 'hour-of-day':
                    expr = pl.col(self.time_col).dt.hour()
                case 'day-of-month':
                    expr = pl.col(self.time_col).dt.day() - 1
                case 'day-of-week':
                    expr = pl.col(self.time_col).dt.weekday() - 1
                case 'month-of-year':
                    expr = pl.col(self.time_col).dt.month() - 1
            time_exprs.append(expr.alias(tfeat))

        tdf = df.select(time_exprs)

        real_data = df.select(self.target_cols).to_numpy().astype(np.float32)
        time_data = tdf.select(self.time_features).to_numpy().astype(np.int32)

        real_data = torch.tensor(real_data, dtype=torch.float32)
        time_data = torch.tensor(real_data, dtype=torch.int32)

        return TSBatch(real_data.unsqueeze(0), time_data.unsqueeze(0))
