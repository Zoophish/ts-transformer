import torch
import random


class ContextHorizonDataset(torch.utils.data.Dataset):
    """
    Generates context and horizon window pairs.

    Sample mode:
        - 'context-horizon-coverage' Ensures context and horizon window pairs have complete
            coverage (excluding remainder)
        - 'horizon-coverage' Ensures the horizons have complete coverage (exlucding remainder)
        - 'random' Randomly positions the context-horizon pair. 
    """
    def __init__(
            self,
            time_series,
            context_length,
            horizon_length,
            sample_mode='random',
            **kwargs
        ):
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.sample_mode = sample_mode
        self.n_samples = kwargs.get('n_samples')
        self.time_series = time_series
        self.indices = self._create_indices()
        self.remainder = 0

    def _create_indices(self):
        indices = []
        match self.sample_mode:
            case 'context-horizon-coverage':
                i = self.context_length
                while i <= len(self.time_series) - self.horizon_length:
                    indices.append(i)
                    i += self.context_length + self.horizon_length
                self.remainder = len(self.time_series) - 1 - i
            case 'horizon-coverage':
                i = self.context_length
                while i <= len(self.time_series) - self.horizon_length:
                    indices.append(i)
                    i += self.horizon_length
                self.remainder = len(self.time_series) - 1 - i
            case 'random':
                indices = [random.randint(
                    self.context_length,
                    len(self.time_series) - self.horizon_length
                ) for _ in range(self.n_samples)]
                self.remainder = self.horizon_length
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        horizon_start_idx = self.indices[idx]
        context_start_idx = horizon_start_idx - self.context_length
        x = torch.tensor(
            self.time_series[context_start_idx:horizon_start_idx],
            dtype=torch.float32
        )
        y = torch.tensor(
            self.time_series[horizon_start_idx:horizon_start_idx + self.horizon_length],
            dtype=torch.float32
        )
        if x.dim() == y.dim() == 1:
            x = x.unsqueeze(-1)
            y = y.unsqueeze(-1)
        return x, y


class IntervalDataset(torch.utils.data.Dataset):
    """
    Generates single intervals over the time series.
    """
    def __init__(self, time_series, interval_length : int):
        super().__init__()
        self.interval_length = interval_length
        self.time_series = time_series

    def __len__(self):
        return len(self.time_series) // self.interval_length
    
    def __getitem__(self, idx):
        start_idx = idx * self.interval_length
        end_idx = start_idx + self.interval_length
        x = torch.tensor(
            self.time_series[start_idx:end_idx],
            dtype=torch.float32
        )
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return x


class RandomIntervalDataset(torch.utils.data.Dataset):
    """
    Randomly samples intervals of variable length and position.
    """
    def __init__(self, time_series, min_steps, max_steps, n_samples):
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.n_samples = n_samples
        self.time_series = time_series

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        n_steps = random.randint(self.min_steps, self.max_steps)
        start_idx = random.randint(0, len(self.time_series) - n_steps)
        end_idx = start_idx + n_steps
        x = torch.tensor(
            self.time_series[start_idx:end_idx],
            dtype=torch.float32
        )
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return x


def collate_pad(batch_items):
    """
    Collates a list of variable length sequences into a
    batch tensor and generates the respective padding mask.
    """
    # _, feat_dim = batch_items[0].shape
    # batch_size = len(batch_items)
    lengths = [len(seq) for seq in batch_items]
    # padded_batch = torch.zeros(batch_size, max(lengths), feat_dim)
    padded_batch = torch.nn.utils.rnn.pad_sequence(batch_items, batch_first=True)
    mask = torch.ones(padded_batch.shape[0], padded_batch.shape[1])
    for i, length in enumerate(lengths):
        padded_batch[i, :length, ...] = batch_items[i]
        mask[i, :length] = 0.0
    return padded_batch, mask.bool()
