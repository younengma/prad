"""
@author: mayouneng
@email: youneng.ma@qq.com
@time: 2024/3/28 10:36
@DESC: 

"""
from models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from models.gpt2.configuration_gpt2 import GPT2Config
from transformers import AutoModel, AutoConfig, AutoTokenizer

MODEL_CLASS = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer, GPT2Config),
    "auto": (AutoModel, AutoConfig, AutoTokenizer)
}
