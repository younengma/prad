"""
@author: mayouneng
@email: youneng.ma@qq.com
@time: 2024/5/23 9:40
@DESC: 

"""
# !/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# 采用batch的形式来进行推理
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import datasets
from utils import alter_model_after_init, parameter_count
import torch
import shutil
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    # DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed
)
from train_utils import DataCollatorForSeq2Seq
from models.gpt2.custom_gpt2 import GPT2LMHeadModel
from data_utils import get_compute_metrics_and_dataset
from pathlib import Path
from preprocessor import get_preprocess_funcs
from arguments import *
import json
from utils import get_time_tag
from tqdm import tqdm


logger = logging.getLogger(__name__)


# to import to train
def evaluate(data_args, save_path, data_collator, predict_dataset, model, tokenizer, batch_size=16):
    print("***start to predict with generate****")
    if "prefix" in str(save_path) or "mam" in str(save_path):
        batch_size = 1  # prefix tuning do not support batch inference yet
    eval_data_loader = DataLoader(predict_dataset, batch_size, collate_fn=data_collator)
    save_path = Path(save_path)
    tokenizer.padding_side = "left"
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    # start to evaluate
    output_text = []
    ref_text = []
    prev = None  # to identify the refs in the same group
    gen_kwargs = {
        "num_beams": 3,  # data_args.num_beams,
        "do_sample": False,
    #    "no_repeat_ngram_size": data_args.no_repeat_ngram_size,
        "max_new_tokens": data_args.val_max_target_length,
        "eos_token_id": tokenizer.encode('<|end|>')#tokenizer.all_special_ids
    }
    print(tokenizer.special_tokens_map)
    print("gen_kwargs:", gen_kwargs)
    generate_file = save_path / f'bs{data_args.num_beams}_detail_generated.json'
    prev = ""
    with open(generate_file, 'w', encoding='utf-8') as f:
        for i, batched_data in tqdm(enumerate(eval_data_loader)):
            input_ids = batched_data['input_ids']
            bsize = input_ids.shape[0]
            # print(batched_data)
            sources = [tokenizer.decode(batched_data['input_ids'][i], skip_special_tokens=False) for i in range(bsize)]
            targets = [tokenizer.decode([_ for _ in batched_data['labels'][i] if _ != -100], skip_special_tokens=True) for i in range(bsize)]

            # we need left padding for batch infer
            batched_data = tokenizer(sources, padding=True, return_tensors="pt")
            input_ids = batched_data['input_ids']
            batched_data['attention_mask'] = (input_ids != tokenizer.pad_token_id).float()
            batched_data['c_masks'] = batched_data['attention_mask'].clone().unsqueeze(-1)
            # print(batched_data)
            # 可进行整理，同样地source不用推理两遍哟， 对于mt和xsum 一个source只有一个reference
            # 一个source对应有多个references
            with torch.no_grad():
                batched_data = batched_data.to(model.device)
                outputs = model.generate(**batched_data, **gen_kwargs)
                preds = outputs[:, input_ids.shape[1]:]
                out_texts = tokenizer.batch_decode(preds, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True)
                out_texts = [out_text.replace('\n', '') for out_text in out_texts]

            for source, target, out_text in zip(sources, targets, out_texts):
              source = source.replace("<|end|>", "")
              if prev != source:  
                # print(f'\n<source: {source}\n>>label: {target}\n>>>pred: {out_text}')
                  f.write(
                    json.dumps({'source': source, 'target': target, 'pred': out_text}, ensure_ascii=False) + "\n")
                  ref_text.append("")
                  output_text.append(out_text.replace('\n', ''))

              ref_text.append(target)
              prev = source


            if i % 10 == 0:
               print(f'processing: {i}/{len(eval_data_loader)}')
            
    sys_file = save_path / f"bs{data_args.num_beams}_bi_output.txt"
    with open(sys_file, 'w', encoding='utf-8') as f:
        for line in output_text:
            f.write(line)
            f.write('\n')

    ref_text = ref_text[1:]
    ref_file = save_path / f"ref_bi.txt"
    with open(ref_file, 'w', encoding='utf-8') as f:
        for line in ref_text:
            f.write(line)
            f.write('\n')
    print("predict file saved at:", sys_file)
    print("ref file saved at:", ref_file)
    print(f"detail generated file saved at :{generate_file}")



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, AdapterArguments))
    args_json = ['configs/base.json',
                 "configs/prefix_adapter_c.json",
                 "configs/all.json",
                 "configs/web_nlg.json",
                 "configs/e2e.json"][-1]
    if len(sys.argv) == 1:
        # for local testing purpose
        model_args, data_args, training_args, adapter_config = parser.parse_json_file(json_file=args_json)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args_json = os.path.abspath(sys.argv[1])
        model_args, data_args, training_args, adapter_config = parser.parse_json_file(json_file=args_json)
    elif len(sys.argv) == 3 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args_json = os.path.abspath(sys.argv[1])
        model_args, data_args, training_args, adapter_config = parser.parse_json_file(json_file=args_json)
        training_args.seed = int(sys.argv[2])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_summarization", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    if model_args.eval_checkpoint is None:
        model_args.eval_checkpoint = f'output/{data_args.task_name}/{Path(args_json).stem}/seed{training_args.seed}'
    # Set seed before initializing model.
    set_seed(training_args.seed)
    config = AutoConfig.from_pretrained(
        model_args.eval_checkpoint,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.eval_checkpoint,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    training_args.output_dir = model_args.eval_checkpoint
    model = GPT2LMHeadModel(config=config, adapter_config=adapter_config)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    logger.info(f"original vocab_size: {embedding_size}, new vocab_size: {len(tokenizer)}")
    model = model.from_pretrained(
        model_args.eval_checkpoint,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        adapter_config=adapter_config,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=False
    ).cuda()
    # alter_model_after_init(model)
    # parameter_count(model, True, 1)
    # use a separate path for evaluation
    logger.info("model loaded.")

    raw_datasets, compute_metrics = get_compute_metrics_and_dataset(task=data_args.task_name, tokenizer=tokenizer)

    if hasattr(model.config, 'n_positions'):
        assert data_args.max_source_length + data_args.max_target_length <= model.config.n_positions, "sequence too long"

    column_names = raw_datasets["test"].column_names

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    _, preprocess_function = get_preprocess_funcs(data_args, tokenizer, model, 0)

    predict_dataset = raw_datasets["test"]
    if data_args.max_predict_samples is not None:
        max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
        predict_dataset = predict_dataset.select(range(max_predict_samples))
    with training_args.main_process_first(desc="prediction dataset map pre-processing"):
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    # only when prompt is turned on will append the prompt related data
    if adapter_config.apply_prompt_tuning and adapter_config.prompt_length > 0:
        prompt_length = adapter_config.prompt_length
    else:
        prompt_length = 0

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        prompt_start_index=len(tokenizer) - prompt_length,
        prompt_length=prompt_length
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    save_path = Path(model_args.eval_checkpoint) / 'eval'
    os.makedirs(save_path, exist_ok=True)

    evaluate(data_args, save_path, data_collator, predict_dataset, model, tokenizer)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
