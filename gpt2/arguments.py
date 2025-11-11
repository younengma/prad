"""
@author: mayouneng
@email: youneng.ma@qq.com
@time: 2024/5/11 11:26
@DESC: 

"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    eval_checkpoint: str = field(
        metadata={"help": "Path to a checkpoint path for evaluation from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class AdapterArguments:

    full_tune: bool = field(
        default=False,
        metadata={
            "help": (
                "whether use full finetune, if is True, all model weights will be tuned,  other adapter configs will be ignored"
            )
        },
    )
    attn_mode: str = field(default='prefix',
                           metadata={"help": "adapter used in the attention layer, eg: none, adapter, lora, prefix"}
                           )
    attn_option: str = field(default='parallel',
                             metadata={"help": "adapter options eg: sequential, parallel, concat(only for prefix)"}
                             )
    attn_r: Optional[int] = field(default=32,
                                  metadata={"help": "attn bottleneck dim"}
                                  )
    attn_adapter_scalar: Optional[int] = field(default=1,
                                               metadata={"help": "attn adapter scaler"}
                                               )
    ffn_mode: str = field(default='adapter',
                          metadata={"help": "adapter used in the attention layer, eg: none, adapter, lora"}
                          )
    ffn_option: str = field(default='parallel',
                            metadata={"help": "adapter options eg: sequential, parallel, concat(only for prefix)"}
                            )
    ffn_adapter_layernorm_option: str = field(default='none',
                                              metadata={
                                                  "help": "place to apply the layernorm in the adatper: none, in, out"}
                                              )
    ffn_adapter_scalar: Optional[int] = field(default=4,
                                              metadata={"help": "ffn adapter scaler"}
                                              )
    ffn_r: Optional[int] = field(default=32,
                                 metadata={"help": "ffn adapeter bottleneck dim"}
                                 )

    lora_alpha: Optional[int] = field(default=32,
                                      metadata={"help": "lora alpha when lora are used"}
                                      )
    lora_dropout: Optional[float] = field(default=0.1,
                                          metadata={"help": "lora dropout when lora are used"}
                                          )
    apply_prompt_tuning: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to add virtual input tokens"
            )
        },
    )
    prefix_projection: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to add virtual input tokens"
            )
        },
    )
    prompt_length: Optional[int] = field(default=32,
                                         metadata={"help": "how many virtual tokens to add when if apply prompt tuning"}
                                         )

    prefix_length: Optional[int] = field(default=32,
                                         metadata={"help": "prefix length if use prefix tuning"}
                                         )

    apply_c_masks: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to add mask only to the prompt. If apply_c_masks is True,  "
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    task_name: str = field(default=None, metadata={"help": "task name"})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    source_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    target_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=924,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    max_target_length: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    max_tokens_per_batch: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "if > 0, will use dynamic batching"
            )
        },
    )

    no_repeat_ngram_size: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "if > 0, will use dynamic batching"
            )
        },
    )

    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`. "
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    ignore_token_type_ids: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id. "
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if (
                self.dataset_name is None
                and self.train_file is None
                and self.validation_file is None
                and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training, validation, or test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
