import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEncoding(nn.Module):
    """
    Method from "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    def __init__(self, dim, precompute_seq_len=512, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base

        # precompute inverse of theta vector Θ = {θi = 10000−2(i−1)/d, i ∈ [1, 2, ..., d/2]}
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(seq_len=precompute_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        # generate indices
        t = torch.arange(seq_len).type_as(self.inv_freq)
        # calculate the arguments for sin and cos: m * theta_i
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # concatenate along the last dimension to get pairs of (m*theta_i, m*theta_i)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_cached = emb.cos()
        sin_cached = emb.sin()
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        self.max_seq_len_cached = seq_len

    def forward(self, x : torch.Tensor, start_pos=0):
        batch_size, n_head, seq_len, dim = x.shape
        # support a start pos so kv caching can be used
        if seq_len + start_pos > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len + start_pos)
        # add singleton dimensions for batch and attn heads
        cos = self.cos_cached[None, None, start_pos:seq_len+start_pos, ...]
        sin = self.sin_cached[None, None, start_pos:seq_len+start_pos, ...]

        # computationally efficient form of rotary matrix multiplication
        x_half_1 = x[..., 0::2]  # get every even indexed element
        x_half_2 = x[..., 1::2]  # ... and every odd indexed
        # x_rotated = torch.cat((-x_half_2, x_half_1), dim=-1)
        x_rotated = torch.stack([-x_half_2, x_half_1], dim=-1).flatten(start_dim=-2)
        return x * cos + x_rotated * sin


class Attention(nn.Module):
    """
    Multihead attention mechanism (with RoPE).
    
    The faster torch implementation of scaled dot product attention is available.
    """
    def __init__(
            self,
            dim : int,
            n_head : int,
            weight_dropout : float = 0.0,
            imp : str ='torch'
        ):
        super().__init__()

        if dim % n_head != 0:
            raise ValueError("dim must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.weight_dropout = weight_dropout
        self.force_dropout = False
        self.imp = imp
        
        # query, key, value, output matrices
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)
        
        # weight dropout module for internal implementation only
        self.weight_dropout_layer = nn.Dropout(weight_dropout)

        self.rope = RotaryPositionalEncoding(self.head_dim)
        self.kv_cache = None

    @property
    def _weight_dropout(self):
        """
        If the torch implementation of scaled dot product attn is used, can't use
        a Dropout module, so this is a workaround.
        """
        return self.weight_dropout if self.training or self.force_dropout else 0.0

    # @torch.compile
    def scaled_dot_product_attention(self, q, k, v, mask, dropout_p = 0.0):
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if mask is not None:
            mask = mask.unsqueeze(1)  # extend the mask across attn head dim
            scores = scores.masked_fill(mask, float('-inf'))  # note, true -> -inf
        attention_weights = self.weight_dropout_layer(F.softmax(scores, dim=-1))
        return torch.matmul(attention_weights, v)

    def forward(self, x : torch.Tensor, mask=None, use_kv_cache = False):
        batch_size, seq_len, n_feat = x.shape
        # generate query, key and value for every token in seq
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        # separate each head and transpose for correct batching
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        # at inference, need to provide offset
        start_pos = 0
        if self.kv_cache is not None:
            start_pos = self.kv_cache[0].shape[2]
        # apply RoPE
        q = self.rope(q, start_pos)
        k = self.rope(k, start_pos)
        # at inference k,v,q are of length 1; the new q must look at the old and new k, v
        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        if use_kv_cache:
            self.kv_cache = k, v
        # dot product attention
        match self.imp:
            case 'internal':
                output = self.scaled_dot_product_attention(q, k, v, mask, self._weight_dropout)
            case 'torch':
                mask = ~mask.unsqueeze(1) if not use_kv_cache else None
                output = F.scaled_dot_product_attention(q, k, v, mask, self._weight_dropout)

        # transpose back to original dims, ensure contiguity before creating view
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # apply output layer
        output = self.W_o(output)
        return output


class MLP(nn.Module):
    """
    Can be used inplace of GLU in EncoderLayer. Here for completeness.
    """
    def __init__(self, d_model : int, d_ff : int, activation = F.silu):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.activation = activation

    def forward(self, x : torch.Tensor):
        return self.fc2(self.activation(self.fc1(x)))


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) which sort of does dynamic intra-token information flow.
    """
    def __init__(self, d_model : int, d_ff : int, activation = F.silu):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.activation = activation

    def forward(self, x : torch.Tensor):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated_gate = self.activation(gate)
        gated_output = activated_gate * up
        return self.down_proj(gated_output)


class EncoderLayer(nn.Module):
    def __init__(
            self,
            d_model : int,
            n_head : int,
            d_ff : int,
            dropout_attn : float,
            dropout_residual : float,
            attn_imp = 'torch'
        ):
        super().__init__()
        self.attention = Attention(
            d_model,
            n_head,
            dropout_attn,
            imp=attn_imp
        )
        self.ffn = GatedLinearUnit(d_model, d_ff)
        self.layer_norm1 = nn.RMSNorm(d_model)
        self.layer_norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_residual)

    def reset_kv_cache(self):
        self.attention.kv_cache = None

    def forward(self, x : torch.Tensor, mask=None, use_kv_cache=False):
        # pre-layer normalisation
        normed_x = self.layer_norm1(x)
        attn = self.attention(normed_x, mask, use_kv_cache)
        x = x + self.dropout(attn)

        normed_x = self.layer_norm2(x)
        ff_out = self.ffn(normed_x)
        x = x + self.dropout(ff_out)
        return x


class DecoderTransformer(nn.Module):
    def __init__(
            self,
            in_dim : int,
            out_dim : int,
            d_model : int,
            n_head : int,
            d_ff : int,
            n_layers : int,
            dropout_attn : float,
            dropout_residual : float,
            dropout_embed : float,
            attn_imp = 'torch'
        ):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(in_dim, d_model)
        self.embed_dropout = nn.Dropout(dropout_embed)
        self.decoder_blocks = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout_attn, dropout_residual, attn_imp)
            for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, out_dim)

    def _generate_causal_mask(self, sz: int, device) -> torch.Tensor:
        """
        The causal mask will mean that at any output position, the attention heads
        can only attend to inputs preceding it. This is a training trick which is more
        efficient than showing the model input/target pairs, but only for single
        step predictions.
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask
    
    def reset_kv_cache(self):
        for decoder_block in self.decoder_blocks:
            decoder_block.reset_kv_cache()
    
    def forward(self, x : torch.Tensor, mask=None, is_inference=False):
        seq_len = x.size(1)

        if not is_inference:
            # the causal mask extends across the batch dim
            causal_mask = self._generate_causal_mask(seq_len, x.device).unsqueeze(0)
            if mask is None:
                mask = causal_mask
            else:
                # combine causal mask with mask (extended across the key dim)
                mask = mask.unsqueeze(1) | causal_mask

        # project the input (tokens) to the embedding dimension
        x = self.embed_dropout(self.input_projection(x))

        # process each decoder block
        decoder_out = x
        for decoder_block in self.decoder_blocks:
            decoder_out = decoder_block(decoder_out, mask, use_kv_cache=is_inference)
        # apply output layer
        out = self.fc(decoder_out)
        return out
