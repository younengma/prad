"""
@author: mayouneng
@email: youneng.ma@qq.com
@DESC: 

"""
from datasets import load_dataset
from pathlib import Path
import numpy as np
from utils import postprocess_text
from metrics.bleu import compute_bleu
from metrics.rouge import compute_rouge
from metrics.nist_mt import compute_nist
from metrics.meteor import compute_meteor
import numpy as np



def get_xsum_dataset():
    # max_src_len: 35312 max_target_len: 96
    data_dir = Path(r'data/xsum')
    data_files = {'train': data_dir / 'train.json',
                  'validation': data_dir / 'validation.json',
                  'test': data_dir / 'test.json'}
    dataset = load_dataset("json", data_files={k: str(v) for k, v in
                                               data_files.items()}
                           )
    return dataset

def get_sst2_dataset():
    data_dir = Path(r'data/sst2')
    data_files = {'train': data_dir / 'train.json',
                  'validation': data_dir / 'validation.json',
                  'test': data_dir / 'validation.json'} # use validation as test
    dataset = load_dataset("json", data_files={k: str(v) for k, v in
                                               data_files.items()}
                           )
    return dataset

def get_mnli_dataset():
    # max_src_len: 35312 max_target_len: 96
    data_dir = Path(r'data/mnli-m')
    data_files = {'train': data_dir / 'train.json',
                  'validation': data_dir / 'validation.json',
                  'test': data_dir / 'validation.json'}
    dataset = load_dataset("json", data_files={k: str(v) for k, v in
                                               data_files.items()}
                           )
    return dataset




def get_longest_sample_inds(dataset, src_col, target_col, num):
    lens = [len(v[src_col]) + len(v[target_col]) for v in dataset]
    inds = np.argsort(lens)
    print(dataset[int(inds[-1])])
    return inds[-1*num:]


def get_max_seq_length(dataset, tokenizer,  src_col, target_col, N=5000, num_workers=4):

    dataset = dataset.select(get_longest_sample_inds(dataset, src_col, target_col, N))

    def process_func(examples):
        src_lens = []
        target_lens = []
        for i in range(len(examples[src_col])):
            if examples[src_col][i] and examples[target_col][i]:
                src_lens.append(len(tokenizer.tokenize(examples[src_col][i])))
                target_lens.append(len(tokenizer.tokenize(examples[target_col][i])))
        return {
            "src_len": src_lens,
            "target_len": target_lens
        }
    dataset = dataset.map(
        process_func,
        batched=True,
        num_proc=num_workers,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
    )

    max_src_len = 0
    max_target_len = 0
    for sample in dataset:
        if sample["src_len"] > max_src_len:
            max_src_len = sample['src_len']
        if sample['target_len'] > max_target_len:
            max_target_len = sample['target_len']

    print(f"max_src_len:{max_src_len}   max_target_len:{max_target_len}")
    return max_src_len, max_target_len


def get_webnlg_dataset():
    # src: 188, target: 105
    data_dir = Path(r'data/web_nlg')
    data_files = {'train': data_dir / 'train.json',
                  'validation': data_dir / 'validation.json',
                  'test': data_dir / 'test.json'}
    dataset = load_dataset("json", data_files={k: str(v) for k, v in
                                               data_files.items()}
                           )
    return dataset


def get_sum_dataset():
    data_dir = Path(r'data/cnn_daily')
    data_files = {'train': data_dir / 'train.json',
                  'validation': data_dir / 'validation.json',
                  'test': data_dir / 'test.json'}
    dataset = load_dataset("json", data_files={k: str(v) for k, v in
                                               data_files.items()}
                           )
    return dataset


def get_ro2en_dataset():
    data_dir = Path(r'data/en_ro')
    data_files = {'train': data_dir / 'train.json',
                  'validation': data_dir / 'validation.json',
                  'test': data_dir / 'test.json'}
    dataset = load_dataset("json", data_files={k: str(v) for k, v in
                                               data_files.items()})
    return dataset


def get_mt_dataset():
    data_dir = Path(r'data/iwslt14-de-en')
    data_files = {'train': data_dir / 'train.json',
                  'validation': data_dir / 'validation.json',
                  'test': data_dir / 'test.json'}
    dataset = load_dataset("json", data_files={k: str(v) for k, v in
                                               data_files.items()})
    return dataset


def get_e2e_dataset():
    # src 50, target: 76
    data_dir = Path(r'data/e2e')
    data_files = {'train': data_dir / 'train.json',
                  'validation': data_dir / 'validation.json',
                  'test': data_dir / 'test.json'}
    dataset = load_dataset("json", data_files={k: str(v) for k, v in
                                               data_files.items()})

    # BLEU: 0.5737
    # NIST: 9.5549
    # METEOR: 0.4020
    # ROUGE_L: 0.7151
    # CIDEr: 3.6612   # https://github.com/vrama91/cider/tree/master
    return dataset


def get_compute_metrics(task, tokenizer):
    def compute_rouge1(eval_preds):
        decoded_preds, decoded_labels = eval_preds
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = compute_rouge(predictions=decoded_preds, references=decoded_labels)
        result = {k: round(v, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(tokenizer.encode(pred) != tokenizer.pad_token_id) for pred in decoded_preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    def compute_bleu1(eval_preds):
        decoded_preds, decoded_labels = eval_preds
        result = compute_bleu(predictions=decoded_preds, references=decoded_labels, smooth=True)
        result['bleu'] = round(result['bleu'], 4)
        result["precisions"] = str(result["precisions"])
        return result

    def e2e_metric_compute(eval_preds):
        decoded_preds, decoded_labels = eval_preds
        result = {
            'bleu': compute_bleu1(eval_preds)['bleu'],
            'rougeL': compute_rouge1(eval_preds)['rougeL'],
            'meteor': compute_meteor(decoded_preds, decoded_labels),
            'nist': compute_nist(decoded_preds, decoded_labels)
        }
        return result

    if task == "xsum":
        return compute_rouge1
    elif task == "en2ro":
        return compute_bleu1
    elif task == "e2e":
        return e2e_metric_compute
    elif task == "e2e_s":
        return e2e_metric_compute 
    elif task == "web_nlg":
        return compute_bleu1
    else:
        return compute_bleu1


def get_compute_metrics_and_dataset(task, tokenizer):
    if task == "xsum":
        dataset = get_xsum_dataset()
    elif task == "ro2en":
        dataset = get_ro2en_dataset()
    elif task == "e2e":
        dataset = get_e2e_dataset()
    elif task == "e2e_s":
        dataset = get_e2e_s_dataset()
    elif task == "web_nlg":
        dataset = get_webnlg_dataset()
    elif task == "sum":
        dataset = get_sum_dataset()
    elif task == 'mt':
        dataset = get_mt_dataset()
    elif task == 'sst2':
        dataset = get_sst2_dataset()
    elif task in {"mnli", "mnli1"}:
        dataset = get_mnli_dataset()
    else:
        raise ValueError("unsupported task name")
    compute_metrics = get_compute_metrics(task, tokenizer)
    return dataset, compute_metrics


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("G:\\nlp\\pretrained_model\\gpt2")
    if False:
        print("webnlg")
        dataset = get_webnlg_dataset()['train']
        src_col = "triple"
        target_col = "text"
        get_max_seq_length(dataset, tokenizer, src_col, target_col)

        print("e2e")
        dataset = get_e2e_dataset()['train']
        src_col = "meaning_representation"
        target_col = "human_reference"
        get_max_seq_length(dataset, tokenizer, src_col, target_col)

    print("xsum")
    dataset = get_xsum_dataset()['train']
    src_col = "document"
    target_col = "summary"
    get_max_seq_length(dataset, tokenizer, src_col, target_col)







