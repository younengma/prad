# 用于finetune de2en 数据的脚本。训练翻译任务。
import os
import sys
from typing import List

import deepspeed

import fire
import torch
import transformers
from datasets import load_dataset
from model.custom_llama import LlamaForCausalLM
from model.configuration_llama import LlamaConfig
from adapter.adapter_configuration import AdapterConfig
from pathlib import Path
from data_collator import DataCollator
from util import parameter_count

from transformers import LlamaTokenizer

from utils.prompter import Prompter
from util import alter_model_after_init, save_adapter

import json
import os
from data_utils import *


def train(
    # model/data params
    base_model: str = "/data/oceanus_ctr/j-mayouneng-jk/llama-7b",  # the only required argument
    adapter_config_file = "configs/adapter0.json",  # experiment file
    # training hyperparams
    batch_size: int = 4,
    micro_batch_size: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 2e-4,
    cutoff_len: int = 256,
    val_set_size: int = 4000,
    seed: int = 42,
    eval_steps: int = 500,
    local_rank: int = 0,
    task: str = "",
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    # gpu_num = torch.cuda.device_count()
    # micro_batch_size = batch_size//gpu_num
    torch.manual_seed(seed)
    method = Path(adapter_config_file).stem
    run_name = f"seed{seed}_lr{int(learning_rate*1e5)}_epochs{num_epochs}"
    output_dir = os.path.join("output", task, method, run_name)

    save_steps = eval_steps

    os.makedirs(output_dir, exist_ok=True)
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training model on {task} with params:\n"
            f"base_model: {base_model}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"output_dir: {output_dir}\n"
        )
    
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    config = LlamaConfig.from_pretrained(base_model)
    with open(adapter_config_file, "r", encoding='utf-8') as f:
        adaper_config_dict = json.load(f)

    adapter_config = AdapterConfig()
    for key, value in adaper_config_dict.items():
        if hasattr(adapter_config, key):
            setattr(adapter_config, key, value)
    # print(adapter_config.__dict__)
    model = LlamaForCausalLM.from_pretrained(base_model, adapter_config=adapter_config, torch_dtype=torch.bfloat16, device_map=device_map)
    # 下面的换为model=model.from_pretrained， 返回的参数就不一样了
    alter_model_after_init(model)
    print(model)
    parameter_count(model)

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )

    tokenizer.padding_side = "left"  # Allow batched inference

    too_long_sample = 0

    def tokenize(sample):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast

        prompt, target = sample[source_column], sample[target_column]
        # 前面4个id对应"#INPUT":, 后面5个id对应 "#OUTPUT":
        prompt = f"#INPUT: {prompt} #OUTPUT: "   # 加入一个特殊的分隔符号
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        
        if len(prompt_ids) > cutoff_len + 5:
            prompt_ids = prompt_ids[:cutoff_len] + prompt_ids[-5:]  # keep first and last format tokens
            # print("too long, prossed as: ", tokenizer.decode(prompt_ids)) 

        prompt_ids = [tokenizer.bos_token_id] + prompt_ids

        target_cutoff_len = cutoff_len

        if task == "xsum":
            target_cutoff_len = 128

        target_ids = tokenizer.encode(target, add_special_tokens=False)[:target_cutoff_len] + [tokenizer.eos_token_id]
        
        result = dict()
        # prompt mask
        c_mask = len(prompt_ids) * [1]
        input_ids = prompt_ids + target_ids

        result['c_mask'] = c_mask + [0]*(len(input_ids)-len(c_mask))
        result['input_ids'] = input_ids

        result["labels"] = result["input_ids"].copy()

        if not train_on_inputs:
            result['labels'] = [-100]*len(prompt_ids) + input_ids[len(prompt_ids):]

        assert len(result['c_mask']) == len(result['input_ids'])

        return result

    def generate_and_tokenize_prompt(sample):
        tokenized_full_prompt = tokenize(sample)
        return tokenized_full_prompt

    if task == "mt":
        data, source_column, target_column = get_mt_dataset()
    elif task == "xsum":
        data, source_column, target_column = get_xsum_dataset()
    elif task == "mnli":
        data, source_column, target_column = get_mnli_dataset()
    elif task == "sst2":
        data, source_column, target_column = get_sst2_dataset()
    else:
        raise Exception("task not supported yet")

    train_data = data["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt, num_proc=4)
    val_set_size = min(len(data["validation"]), val_set_size)
    val_data = data["validation"].select(range(val_set_size)).map(generate_and_tokenize_prompt, num_proc=4)

    # 打印一个处理的结果瞧一瞧
    print("sample:", train_data[0])

    print("input_text:", tokenizer.decode(train_data[0]["input_ids"]))

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_safetensors=False,
            warmup_steps=200,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            seed=seed,
            bf16=True,
            logging_steps=50,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if val_set_size > 0 else None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="tensorboard",
        #    deepspeed="ds_zero2_no_offload.json",
            run_name=Path(adapter_config_file).stem,
        ),
        data_collator=DataCollator(tokenizer)
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if local_rank == 0:
        trainer.save_model()
        trainer.save_state()

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
