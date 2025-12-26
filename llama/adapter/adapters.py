"""
@author: mayouneng
@email: youneng.ma@qq.com
@time: 2024/3/28 15:46
@DESC: 

"""
import torch.nn as nn
import torch
import math


class LoraLayer(nn.Module):
    def __init__(self, in_dim, out_dim, r=4, alpha=32, drop_out=0.1, apply_c_mask=False):
        super().__init__()
        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)
        self.apply_c_mask = apply_c_mask
        self.dropout = drop_out
        self.alpha = alpha
        if self.dropout > 0:
            self.dropout = nn.Dropout(p=drop_out)
        else:
            self.dropout = lambda x: x
        nn.init.zeros_(self.A.weight)
        nn.init.kaiming_uniform_(self.B.weight, a=math.sqrt(5))

    def forward(self, x, c_mask=None):
        x = self.A(x)
        x = self.alpha * self.B(x)
        x = self.dropout(x)

        if self.apply_c_mask and c_mask is not None:
            x = x*c_mask
        return x


class AdapterLayer(nn.Module):
    def __init__(self,
                 in_dim =None,
                 r=None,
                 dropout=0.0,
                 adapter_scalar="1.0",
                 adapter_layernorm_option=None,
                 apply_c_mask=False):
        super().__init__()
        self.n_embd = in_dim
        self.down_size = r
        self.adapter_layernorm_option = adapter_layernorm_option
        self.apply_c_mask = apply_c_mask

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout

        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None, c_mask=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
        
        # not needed in the inference stage
        if self.apply_c_mask and c_mask is not None:
            up = up*c_mask

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, pre_seq_len, n_heads, kv_channels, num_layers, hidden_size=96, prefix_projection=False):
        super().__init__()
        self.prefix_projection = prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            kv_size = num_layers * kv_channels * n_heads * 2
            self.embedding = torch.nn.Embedding(pre_seq_len, kv_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(kv_size, hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_size, kv_size)
            )
        else:
            self.embedding = torch.nn.Embedding(pre_seq_len,
                                                num_layers * kv_channels * n_heads * 2)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
