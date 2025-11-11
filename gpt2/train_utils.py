from dataclasses import dataclass
import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers import Seq2SeqTrainer as BaseSeq2SeqTrainer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import trainer
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import json
from utils import postprocess_text
import copy
import collections
from transformers.file_utils import is_datasets_available
from transformers.trainer_pt_utils import LengthGroupedSampler, DistributedSampler, RandomSampler, DistributedLengthGroupedSampler, DistributedSamplerWithLoop, IterableDatasetShard
from transformers.training_args import ParallelMode, TrainingArguments
from log_util import get_logger

logger = get_logger(__name__)


_is_torch_generator_available = False
_is_native_amp_available = False

if is_datasets_available():
    import datasets


@dataclass
class DataCollatorForSeq2Seq:
    """
    # adapted from transformers DataCollatorForSeq2Seq
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    prompt_start_index: int = -1
    prompt_length: int = 0

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # temporarily pop the c_mask if we have it
        c_masks = [feature.pop("c_masks") for feature in features] if "c_masks" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # processing the context masks, mask 0, unmasked 1
        if c_masks is not None:
            c_max_length = len(features['input_ids'][0])
            c_masks = [c_mask + [0] * (c_max_length - len(c_mask)) for c_mask in c_masks]
            features['c_masks'] = torch.Tensor(c_masks).float().unsqueeze(-1)

        # handler prompt input ids
        if self.prompt_length > 0:
            input_ids = features['input_ids']
            bsize = input_ids.size(0)
            # prepend prompt to the input_ids
            prompt_ids = torch.arange(start=self.prompt_start_index,
                                      end=len(self.tokenizer),
                                      dtype=input_ids.dtype).unsqueeze(0).expand(bsize, -1)
            features['input_ids'] = torch.cat([prompt_ids, input_ids], dim=-1)
            if "token_type_ids" in features:
                features['token_type_ids'] = torch.cat([prompt_ids, features['token_type_ids']], dim=-1)

            if "labels" in features:
                prompt_labels = torch.full_like(prompt_ids, -100, dtype=features['labels'].dtype)
                features['labels'] = torch.cat([prompt_labels, features['labels']], dim=-1)

            if 'c_masks' in features:
                features['c_masks'] = torch.cat([
                    torch.ones_like(prompt_ids, dtype=features['c_masks'].dtype).unsqueeze(-1),
                    features['c_masks']
                ], dim=1)

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        # we do not need attention mask
        if "attention_mask" in features:
            features.pop("attention_mask")

        return features


class Seq2SeqTrainer(BaseSeq2SeqTrainer):
    def __init__(
            self,
            model: Union["PreTrainedModel", nn.Module] = None,
            args: "TrainingArguments" = None,
            data_collator: Optional["DataCollator"] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional["PreTrainedTokenizerBase"] = None,
            model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
            compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
            callbacks: Optional[List["TrainerCallback"]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            vocab_size_original: Optional[int] = 0,
            fix_pretrained_token_embed: Optional[int] = True,
            max_tokens_per_batch: Optional[int] = 0,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.max_tokens_per_batch = max_tokens_per_batch
        # Override self.model.generation_config if a GenerationConfig is specified in args.
        # Priority: args.generation_config > model.generation_config > default GenerationConfig.
        if self.args.generation_config is not None:
            gen_config = self.load_generation_config(self.args.generation_config)
            self.model.generation_config = gen_config
        self.vocab_size_original = vocab_size_original
        self.fix_pretrained_token_embed = fix_pretrained_token_embed
        self.eval_dir = os.path.join(self.args.output_dir, 'eval')
        os.makedirs(self.eval_dir, exist_ok=True)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        loss = super().training_step(model, inputs)
        # update only newly added token embeddings
        if (model.transformer.wte.weight.grad is not None
             and self.fix_pretrained_token_embed):
             model.transformer.wte.weight.grad[:self.vocab_size_original, :] = 0
        return loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=self.args.seed,
                )

        else:
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    return RandomSampler(self.train_dataset, generator=generator)
                return RandomSampler(self.train_dataset)
            elif (
                    self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                    and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        # import pdb; pdb.set_trace()
        if isinstance(train_dataset, torch.utils.data.dataset.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        if self.max_tokens_per_batch == 0:
            train_sampler = self._get_train_sampler()

            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            return DataLoader(
                train_dataset,
                collate_fn=self.data_collator,
                batch_sampler=train_dataset.make_dynamic_sampler(
                    self.max_tokens_per_batch, distributed=(self.args.world_size > 1)),
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        compute_loss=True,
        **gen_kwargs,
    ) -> Dict[str, float]:
        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        self._gen_kwargs = gen_kwargs
        result = dict()
        result_metric = dict()
        # calculate the loss on the evaluation
        eval_data_loader1 = DataLoader(eval_dataset, self.args.per_device_eval_batch_size,
                                       collate_fn=self.data_collator)
        self.model.eval()
        loss = 0
        for i, batched_data in enumerate(eval_data_loader1):
            with torch.no_grad():
                outputs = self.model(**batched_data.to(self.model.device))
                loss += outputs.loss.item()
        result["eval_loss"] = loss/(i+1)
        self.model.train()

        result["number_samples"] = len(eval_dataset)
        result['step'] = self.state.global_step
        result.update(result_metric)
        # metric_save
        report_file = os.path.join(self.eval_dir, f'all_eval_result_of_history_ckpts.json')
        with open(report_file, "a", encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result

    def predict(
        self,
        test_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
    ) -> Dict[str, float]:

        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        gen_kwargs["eos_token_id"] = (
            gen_kwargs["eos_token_id"] if gen_kwargs.get("eos_token_id") is not None else self.tokenizer.eos_token_id
        )
        self._gen_kwargs = gen_kwargs
        logger.info(f"gen_kwargs: {gen_kwargs}")
        bsize = 1
        eval_data_loader = DataLoader(test_dataset, bsize, collate_fn=self.data_collator)
        total = len(eval_data_loader)
        decoded_labels = []
        decoded_preds = []
        save_text_file = os.path.join(self.eval_dir, f'{metric_key_prefix}_generated.jsonl')
        prev = ""
        prev_pred = ""
        self.model.eval()
        with open(save_text_file, "w", encoding="utf-8") as f:
            for i, batched_data in enumerate(eval_data_loader):
                label = [l for l in batched_data.pop("labels")[0].tolist() if l != -100]
                decoded_label = self.tokenizer.decode(label, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                decoded_label = decoded_label.replace('\n', "")
                input_ids = batched_data['input_ids'][0]
                cur = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                if prev == cur:
                    decoded_pred = prev_pred
                    logger.info(f'skip: {cur[:40]}, already processed')
                else:
                    batched_data = batched_data.to(self.model.device)
                    outputs = self.model.generate(**batched_data, **gen_kwargs)
                    pred = [v for v in outputs[0][len(input_ids):].tolist()]
                    decoded_pred = self.tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    decoded_pred = decoded_pred.replace('\n', "")
                    logger.info(f'\n<label: {decoded_label}\n>pred:{decoded_pred}')

                decoded_labels.append(decoded_label)
                decoded_preds.append(decoded_pred)
                prev = cur
                prev_pred = decoded_pred
                if i % 100 == 0:
                    logger.info(f"processing {i}/{total}")

                f.write(json.dumps({"label": decoded_label, "pred": decoded_pred, "input": cur}, ensure_ascii=False) + "\n")
        result = self.compute_metrics((decoded_preds, decoded_labels))
        result = {f"{metric_key_prefix}_{k}": round(v, 4) for k, v in result.items() if type(v) != str}
        result[f"{metric_key_prefix}_number_samples"] = len(test_dataset)
        logger.info(f"generated report saved at: {save_text_file}")
        logger.info(f"eval result: {result}")
        return result
