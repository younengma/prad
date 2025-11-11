"""
@author: mayouneng
@email: youneng.ma@qq.com
@time: 2024/3/28 13:52
@DESC: 

"""
from log_util import get_logger
import nltk
from transformers.utils import is_offline_mode
from filelock import FileLock
import os
import json
import re
import time
import datetime


log = get_logger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def get_time_tag():
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return time_str

def get_tag(path):
    tag = re.findall(f'output/.*/(.*?)/seed', str(path))
    if len(tag) > 0:
        tag = tag[0]
    else:
        tag = None
    seed = re.findall(f'/seed([0-9]+)/', str(path))
    if len(seed) > 0:
       seed = seed[0]
    else:
       seed = 0
    return tag, seed       

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def alter_model_after_init(model):
    for key, param in model.named_parameters():
        if any([v in key for v in {'prefix', "adapter", "lora", "dora",  "transformer.wte.weight"}]):
            param.requires_grad = True
            log.info(f"trainable:{key}, num:{param.numel()}")
        else:
            param.requires_grad = False


def parameter_count(model, fix_pretrained_token_embed=True, new_token_number=0):
    v1 = 0
    num = 0
    for k, p in model.named_parameters():
        # if "lm_head" in k:
        #     print("===========", k, p.requires_grad)
        num += p.numel()
        if p.requires_grad:
            # only count the newly added token embedding
            if "transformer.wte.weight" in k and fix_pretrained_token_embed:
                v1 += new_token_number*p.size(-1)
            else:
                v1 += p.numel()

    log.info(
        f"========================== tol params:{num}, params requires_grad:{v1} with ratio of {'%.3f' % (v1 / num * 100)}% ================")
    return v1


def dataset2json(dataset, save_dir="./"):
    for key in dataset.keys():
        with open(os.path.join(save_dir, f'{key}.json'), 'w', encoding='utf-8') as f:
            for sample in dataset[key]:
                f.write(json.dumps(sample, ensure_ascii=False)+"\n")
