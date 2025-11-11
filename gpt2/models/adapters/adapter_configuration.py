"""Implements the adapters and other parameter-efficient finetuning methods' configurations."""

from dataclasses import dataclass


@dataclass
class AdapterConfig(object):

    attn_mode = "adapter"  # none, adapter, lora, prefix
    attn_option = "sequential"  # sequential, parallel, concat(only for prefix)
    attn_r = 30  # attn bottleneck dim
    attn_adapter_scalar = 1  # attn adapter scaler

    ffn_mode = "lora"     # none, adapter, lora
    ffn_option = "parallel"   # sequential
    ffn_adapter_layernorm_option = "none"
    ffn_adapter_scalar = 4
    ffn_r = 512  # ffn bottleneck dim

    lora_alpha = 32
    lora_dropout = 0.1
    lora_init = "lora"

    apply_prompt_tuning = True   # whether add soft prompts in the input
    prompt_length = 16
    prefix_length = 32   # prefix length if we use prefix tuning

    apply_c_masks = True


# ----- MAM adapter -----
# attn_mode="prefix"
# attn_option="concat"
# attn_composition="add"
# attn_bn=30  # attn bottleneck dim
#
# ffn_mode="adapter"
# ffn_option="parallel"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="lora"
# ffn_adapter_scalar="4"
# ffn_bn=512 # ffn bottleneck dim

# ----- prefix tuning baseline -----
# attn_mode="prefix"
# attn_option="concat"
# attn_composition="add"
# attn_bn=200  # attn bottleneck dim

# ffn_mode="none"
# ffn_option="parallel"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="lora"
# ffn_adapter_scalar="4"
# ffn_bn=512 # ffn bottleneck dim

# ----- Houlsby Adapter -----
# attn_mode="adapter"
# attn_option="sequential"
# attn_composition="add"
# attn_bn=200  # attn bottleneck dim

# ffn_mode="adapter"
# ffn_option="sequential"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="bert"
# ffn_adapter_scalar="1"
# ffn_bn=200 # ffn bottleneck dim


# ----- FFN Scaled Parallel Adapter -----
# attn_mode="none"
# attn_option="parallel"
# attn_composition="add"
# attn_bn=200  # attn bottleneck dim

# ffn_mode="adapter"
# ffn_option="parallel"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="lora"
# ffn_adapter_scalar="4"
# ffn_bn=512 # ffn bottleneck dim

# ----- Prompt Tuning -----
# attn_mode="prompt_tuning"
# attn_option="parallel"
# attn_composition="add"
# attn_bn=200  # attn bottleneck dim

# ffn_mode="none"
# ffn_option="parallel"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="lora"
# ffn_adapter_scalar="4"
# ffn_bn=512 # ffn bottleneck dim

# ----- bitfit -----
# attn_mode="bitfit"
# attn_option="parallel"
# attn_composition="add"
# attn_bn=200  # attn bottleneck dim

# ffn_mode="none"
# ffn_option="parallel"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="lora"
# ffn_adapter_scalar="4"
# ffn_bn=512 # ffn bottleneck dim

# ----- lora -----
# attn_mode="lora"
# attn_option="none"
# attn_composition="add"
# attn_bn=16

# # set ffn_mode to be 'lora' to use
# # lora at ffn as well

# ffn_mode="none"
# ffn_option="none"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="bert"
# ffn_adapter_scalar="1"
# ffn_bn=16

# lora_alpha=32
# lora_dropout=0.1
# lora_init="lora"



