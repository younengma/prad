#!/usr/bin/env python
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
重跑一下mt的实验。尤其是lora的实验。

"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import warnings


import datasets
from utils import alter_model_after_init, parameter_count
import torch
import shutil
import time

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
from train_utils import DataCollatorForSeq2Seq, Seq2SeqTrainer
from models.gpt2.custom_gpt2 import GPT2LMHeadModel
from transformers.trainer_utils import get_last_checkpoint
from pathlib import Path
from data_utils import get_compute_metrics_and_dataset
from arguments import *
from preprocessor import get_preprocess_funcs
from eval import evaluate

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.39.0")


logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, AdapterArguments))
    args_json = ["configs/e2e/adapter.json"][-1]
    if len(sys.argv) == 1:
        # for local testing purpose
        model_args, data_args, training_args, adapter_config = parser.parse_json_file(json_file=args_json)
        training_args.output_dir = os.path.join(training_args.output_dir, data_args.task_name, Path(args_json).stem)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args_json = os.path.abspath(sys.argv[1])
        model_args, data_args, training_args, adapter_config = parser.parse_json_file(json_file=args_json)
        training_args.output_dir = os.path.join(training_args.output_dir, data_args.task_name, Path(sys.argv[1]).stem)
    elif len(sys.argv) == 3 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args_json = os.path.abspath(sys.argv[1])
        model_args, data_args, training_args, adapter_config = parser.parse_json_file(json_file=args_json)
        training_args.output_dir = os.path.join(training_args.output_dir, data_args.task_name, Path(sys.argv[1]).stem)
        # replace the seed
        training_args.seed = int(sys.argv[2])
    elif len(sys.argv) == 4:
        args_json = os.path.abspath(sys.argv[1])
        model_args, data_args, training_args, adapter_config = parser.parse_json_file(json_file=args_json)
        training_args.output_dir = os.path.join(training_args.output_dir, data_args.task_name, Path(sys.argv[1]).stem)
        # replace the seed
        training_args.seed = int(sys.argv[2])
        training_args.learning_rate = float(sys.argv[-1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(sys.argv)
    training_args.output_dir = os.path.join(training_args.output_dir, f"seed{training_args.seed}lr{int(training_args.learning_rate*1e5)}bsize{training_args.per_device_train_batch_size}")
    training_args.logging_dir = os.path.join(training_args.output_dir, "runs")
    logger.info(f"output_dir is: {training_args.output_dir}")
    eval_dir = os.path.join(training_args.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

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

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if adapter_config.full_tune:
        adapter_config = None

    model = GPT2LMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        adapter_config=adapter_config,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        device_map="cuda"
    )

    origin_vocab_size = model.get_input_embeddings().weight.shape[0]
    if adapter_config is not None:
        if not(adapter_config.apply_prompt_tuning and adapter_config.prompt_length > 0):
            adapter_config.apply_prompt_tuning = False
            adapter_config.prompt_length = 0
        prompt_length = adapter_config.prompt_length
        fix_pretrained_token_embed = True
        logger.info("===========peft tuning.==========")
    else:
        prompt_length = 0
        fix_pretrained_token_embed = False
        logger.info("===========full finetuning.=======")

    preprocess_function, preprocess_function_predict = get_preprocess_funcs(data_args, tokenizer,
                                                                            model, prompt_length)
    new_tokens_number = len(tokenizer) - origin_vocab_size
    prompt_start_index = len(tokenizer) - prompt_length
    logger.info(f"original vocab_size: {origin_vocab_size}, new vocab_size: {len(tokenizer)}")
    
    # setting the trainable parameters for the peft methods
    if adapter_config is not None:
        # dora 需要在初始化以后重新初始化一下magtude
        if adapter_config.ffn_mode == "dora":
            for m in model.transformer.h:
                m = m.mlp
                if hasattr(m, 'f1_dora'):
                    m.f1_dora.post_init()
                if hasattr(m, 'f2_dora'):
                    m.f2_dora.post_init()
        alter_model_after_init(model)



    parameter_count(model, fix_pretrained_token_embed, new_tokens_number)

    # use a separate path for evaluation
    if not training_args.do_train and model_args.eval_checkpoint is not None and training_args.do_predict:
        logger.info(
            f"====== separate model for evaluation, load evaluation model from:{model_args.eval_checkpoint}======")
        model = model.from_pretrained(model_args.eval_checkpoint, config=config, adapter_config=adapter_config)
    time.sleep(3)
    logger.info("model loaded.")
    raw_datasets, compute_metrics = get_compute_metrics_and_dataset(task=data_args.task_name, tokenizer=tokenizer)
    logger.info('dataset loaded. start to load dataset')
    if hasattr(model.config, 'n_positions'):
        assert data_args.max_source_length + data_args.max_target_length <= model.config.n_positions, "sequence too long"

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if training_args.do_train:
        logger.info("start to train...")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        print('input:', tokenizer.decode(train_dataset[0]['input_ids']))

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=eval_dataset.column_names,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_predict,
                batched=True,
                remove_columns=predict_dataset.column_names,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        prompt_start_index=prompt_start_index,
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

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        vocab_size_original=origin_vocab_size,
        fix_pretrained_token_embed=fix_pretrained_token_embed,   
        max_tokens_per_batch=data_args.max_tokens_per_batch,
    )
    print(f"=======model dtype:{model.dtype}========")
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_ds_name, eval_ds in eval_dataset.items():
                dataset_metrics = trainer.evaluate(eval_dataset=eval_ds,
                                                   metric_key_prefix=f"eval_{eval_ds_name}",
                                                   max_new_tokens=data_args.val_max_target_length)
                metrics.update(dataset_metrics)
        else:
            metrics = trainer.evaluate(eval_dataset,
                                       metric_key_prefix="eval",
                                       max_new_tokens=data_args.val_max_target_length)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        evaluate(data_args, eval_dir, data_collator, predict_dataset, model, tokenizer)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
