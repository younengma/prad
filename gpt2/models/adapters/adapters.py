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
    def __init__(self, in_dim, out_dim, r=4, alpha=32, drop_out=0.1, apply_c_masks=False):
        super().__init__()
        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)
        self.apply_c_masks = apply_c_masks
        self.dropout = drop_out
        self.alpha = alpha
        if self.dropout > 0:
            self.dropout = nn.Dropout(p=drop_out)
        else:
            self.dropout = lambda x: x
        nn.init.zeros_(self.A.weight)
        nn.init.kaiming_uniform_(self.B.weight, a=math.sqrt(5))

    def forward(self, x, c_masks=None):
        x = self.A(x)
        x = self.alpha * self.B(x)
        x = self.dropout(x)

        if self.apply_c_masks and c_masks is not None:
            x = x*c_masks
        return x


class AdapterLayer(nn.Module):
    def __init__(self,
                 in_dim =None,
                 r=None,
                 dropout=0.0,
                 adapter_scalar="1.0",
                 adapter_layernorm_option=None,
                 apply_c_masks=False):
        super().__init__()
        self.n_embd = in_dim
        self.down_size = r
        self.adapter_layernorm_option = adapter_layernorm_option
        self.apply_c_masks = apply_c_masks

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

    def forward(self, x, add_residual=True, residual=None, c_masks=None):
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
        if self.apply_c_masks and c_masks is not None:
            up = up*c_masks

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


class DoRALayer(nn.Module):
    # https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch
    """
    DoRA is an evolution of LoRA which decomposes the original pre-trained weight matrix
    into a magnitude vector and direction matrix, upon which LoRA is applied to the direction matrix.
    The low-rank matrices and the magnitude vector are then jointly trained/fintuned.
    """
    def __init__(self, linear, in_dim, out_dim, r=4, alpha=4, drop_out=0.1, apply_c_masks=False):
        super().__init__()
        self.linear = linear
        self.lora = LoraLayer(in_dim, out_dim, r, alpha, drop_out, apply_c_masks)
        self.magnitude = nn.Parameter(self.linear.weight.norm(p=2, dim=0, keepdim=True), requires_grad=True)
        self.linear.weight.requires_grad = False
        self.apply_c_masks = apply_c_masks

    def post_init(self):
        # need to call after the base model has been initialized, because the initial magitude value depends on the base model
        self.magnitude = nn.Parameter(self.linear.weight.norm(p=2, dim=0, keepdim=True), requires_grad=True)
    

    def forward(self, x, c_masks=None):
        # 原来的层的结果
        # 引入adapter的结果
        # y0 = self.linear(x)
        # y_p = self.lora(x, c_masks=c_masks)
        # lora_w = (self.lora.A.weight.T @ self.lora.B.weight.T) * self.lora.alpha
        # merged_weights = self.linear.weight + lora_w
        # 对于lora 只有prompt部分经过dora层。这里需要改变一下才行。先获取prompt的部分。prompt经过dora。再获取其它的部分，只经过linear层。可以通过mask来实现。
        if self.apply_c_masks and c_masks is not None:
           merged_magnitude = (self.lora.A.weight.T @ self.lora.B.weight.T * self.lora.alpha + self.linear.weight).norm(p=2, dim=0, keepdim=True) # used to norm the magnitude
           yp = (self.linear(x)+self.lora(x))*self.magnitude/merged_magnitude*c_masks # 保留prompt的部分，其它设置为零
           yd = self.linear(x)*(1-c_masks)  # 保留非prompt的部分，targets的部分设置为零
           y = yp + yd
        else:
            merged_magnitude = (self.lora.A.weight.T @ self.lora.B.weight.T * self.lora.alpha + self.linear.weight).norm(p=2, dim=0, keepdim=True) # used to norm the magnitude
            y = (self.linear(x)+self.lora(x))*self.magnitude/merged_magnitude
        return y

