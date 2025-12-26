from datasets import load_dataset
from pathlib import Path


def get_mnli_dataset():
    data_dir = Path(r'../gpt2/data/mnli-m')
    data_files = {'train': data_dir / 'train.json',
                  'validation': data_dir / 'validation.json'}
    dataset = load_dataset("json", data_files={k: str(v) for k, v in
                                               data_files.items()})
    return dataset, "source", "target"


def get_sst2_dataset():
    data_dir = Path(r'../gpt2/data/sst2')
    data_files = {'train': data_dir / 'train.json',
                  'validation': data_dir / 'validation.json'}
    dataset = load_dataset("json", data_files={k: str(v) for k, v in
                                               data_files.items()})
    return dataset, "text", "sentiment"


def get_mt_dataset():
    data_dir = Path(r'../gpt2/data/iwslt14-de-en')
    data_files = {'train': data_dir / 'train.json',
                  'validation': data_dir / 'validation.json',
                  'test': data_dir / 'test.json'}
    dataset = load_dataset("json", data_files={k: str(v) for k, v in
                                               data_files.items()})
    return dataset, "de", "en"


def get_xsum_dataset():
    # max_src_len: 35312 max_target_len: 96
    data_dir = Path(r'../gpt2/data/xsum')
    data_files = {'train': data_dir / 'train.json',
                  'validation': data_dir / 'validation.json',
                  'test': data_dir / 'test.json'}
    dataset = load_dataset("json", data_files={k: str(v) for k, v in
                                               data_files.items()}
                           )
    return dataset, "document", "summary"






