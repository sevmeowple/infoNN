import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional

class MultiHeadAttention(pl.LightningModule):
    def __init__(
        self,
        q_dim:      int,
        kv_dim:     int,
        qk_out_dim: Optional[int] = None,
        v_out_dim:  Optional[int] = None,
        output_dim: Optional[int] = None,
        heads:      int = 1,
        dropout:    float = 0.0
        ):
        super().__init__()

        if qk_out_dim is None:
            qk_out_dim = q_dim
        if v_out_dim is None:
            v_out_dim  = qk_out_dim
        if output_dim is None:
            output_dim = v_out_dim

        self.heads       = heads
        self.qk_head_dim = qk_out_dim // heads
        self.v_head_dim  = v_out_dim // heads

        self.qeury = nn.Linear(q_dim, qk_out_dim)
        self.key   = nn.Linear(kv_dim, qk_out_dim)
        self.value = nn.Linear(kv_dim, v_out_dim)

        self.projection = nn.Linear(v_out_dim, output_dim)
        self.dropout    = nn.Dropout(dropout)
    
    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
        ):
        batch = x_q.shape[0]
        query_len, key_len, value_len = x_q.shape[1], x_k.shape[1], x_v.shape[1]

        queries = self.qeury(x_q)
        keys    = self.key(x_k)
        values  = self.value(x_v)

        queries = queries.reshape(batch, query_len, self.heads, self.qk_head_dim)
        keys    = keys.reshape(batch, key_len, self.heads, self.qk_head_dim)
        values  = values.reshape(batch, value_len, self.heads, self.v_head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if attention_mask is not None:
            energy = energy.masked_fill(attention_mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.qk_head_dim ** 0.5), dim=3)
        attention = self.dropout(attention)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            batch, query_len, self.heads * self.v_head_dim
        )
        out = self.projection(out)
        return out

class Query_Gen_transformer(pl.LightningModule):
    def __init__(self, input_dim, dim, max_batch=64, num_filters=128, hidden_dim=512, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.max_batch = max_batch

        self.mlp_x = nn.Sequential(
            nn.Linear(self.input_dim // 2, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        self.mlp_y = nn.Sequential(
            nn.Linear(self.input_dim // 2, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.attention = MultiHeadAttention(
            q_dim=self.dim,
            kv_dim=self.hidden_dim,
            heads=8,
            dropout=dropout
        )

        self.query = nn.Parameter(torch.randn(1, dim, dim))
        self.norm_q = nn.LayerNorm(self.dim)
        self.norm_k = nn.LayerNorm(self.hidden_dim)
        self.norm_v = nn.LayerNorm(self.hidden_dim)
        
    def forward(self, input):
        batch_size = input.shape[0]
        query = self.query.repeat(batch_size, 1, 1)
        X = input[:, :, 0].unsqueeze(-1)
        Y = input[:, :, 1].unsqueeze(-1)

        X_long = self.mlp_x(X)
        Y_long = self.mlp_y(Y)

        Q = self.norm_q(query)
        K = self.norm_k(X_long)
        V = self.norm_v(Y_long)
        attention = self.attention(Q, K, V)
        return attention

def PositionalEmbedding(input):
    input = input.squeeze()
    batch_size = input.shape[0]
    input_dim = input.shape[1]
    device = input.device
    positional_embedding = torch.zeros(batch_size, input_dim, device=device)
    position = torch.arange(0, batch_size, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, input_dim, 2, device=device).float() * (-torch.log(torch.tensor(10000.0, device=device)) / input_dim))
    positional_embedding[:, 0::2] = torch.sin(position * div_term)
    positional_embedding[:, 1::2] = torch.cos(position * div_term)
    return (input + positional_embedding).unsqueeze(0)

class Query_Gen_transformer_PE(pl.LightningModule):
    def __init__(self, input_dim, dim, num_filters=128, hidden_dim=512, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.mlp_x = nn.Sequential(
            nn.Linear(self.input_dim // 2, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        self.mlp_y = nn.Sequential(
            nn.Linear(self.input_dim // 2, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.attention = MultiHeadAttention(
            q_dim=self.dim,
            kv_dim=self.hidden_dim,
            heads=8,
            dropout=dropout
        )

        self.query = nn.Parameter(torch.randn(1, dim, dim))
        self.norm_q = nn.LayerNorm(self.dim)
        self.norm_k = nn.LayerNorm(self.hidden_dim)
        self.norm_v = nn.LayerNorm(self.hidden_dim)
        
    def forward(self, input):
        input = PositionalEmbedding(input)
        X = input[:, :, 0]
        Y = input[:, :, 1]

        X_long = self.mlp_x(X)
        Y_long = self.mlp_y(Y)

        Q = self.norm_q(self.query)
        K = self.norm_k(X_long)
        V = self.norm_v(Y_long)
        attention = self.attention(Q, K, V)
        return attention