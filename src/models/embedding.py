import math

import torch
import torch.nn as nn


class FixedEmbedding(nn.Module):
    """
    "Attention Is All You Need" style positional encodings.
    """

    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TimeSeriesEmbeddingBlock(nn.Module):
    """
    Embeds the input into d_model.

    The temporal embeddings are summed with the projected feature embeddings.
    """
    def __init__(
            self,
            d_model : int,
            d_feat : int,
            feature_config : dict,
            fixed_temporal_emb : bool = False
        ):
        super().__init__()

        self.d_model = d_model

        n_real = len(feature_config['real_features'])

        self.feature_embedding = nn.Conv1d(
            in_channels=n_real, 
            out_channels=n_real * d_feat, 
            kernel_size=1, 
            groups=n_real
        )
        self.input_projection = nn.Linear(n_real * d_feat, d_model)
        self.temporal_embeddings = nn.ModuleDict()
        
        Embedding = FixedEmbedding if fixed_temporal_emb else nn.Embedding
        for name, cardinality in feature_config['temporal_features']:
            self.temporal_embeddings[name] = Embedding(cardinality, d_model)

    def forward(self, x_real : torch.Tensor, x_time : torch.Tensor) -> torch.Tensor:
        # x_real: [Batch, Seq, n_real]
        # Conv1d expects [Batch, Channels, Seq]
        x = x_real.permute(0, 2, 1) # -> [Batch, n_real, Seq]
        
        x = self.feature_embedding(x)
        
        # Swap back -> [Batch, Seq, n_real * d_feat]
        x = x.permute(0, 2, 1)
        
        x = self.input_projection(x) # -> [Batch, Seq, d_model]

        # add temporal embeddings
        for i, name in enumerate(self.temporal_embeddings):
            x = x + self.temporal_embeddings[name](x_time[i])
                
        return x
