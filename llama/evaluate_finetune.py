import torch
import json
from transformers import LlamaTokenizer, HfArgumentParser
from model.custom_llama import LlamaForCausalLM
from adapter.adapter_configuration import AdapterConfig
from data_utils import *
import time
from tqdm import tqdm
import os
import argparse


def prepare_cutoff_inputs(prompts, cutoff_len):
    
    input_ids = []
    attention_mask = []
    
    max_length = cutoff_len + 8
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        # the last five is the response split
        if len(prompt_ids) > cutoff_len + 5:
            prompt_ids = prompt_ids[:cutoff_len] + prompt_ids[-5:]  # keep first and last format tokens
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids
        _len = len(prompt_ids)
        _input_ids = [tokenizer.pad_token_id]*(max_length-_len) + prompt_ids
        input_ids.append(_input_ids)
        attention_mask.append([0]*(max_length-_len) + [1]*_len)
    
    input_tokens = {
          "input_ids": torch.LongTensor(input_ids),
          "attention_mask": torch.Tensor(attention_mask)
        }
    # left pad the input_ids
    return input_tokens


def pet_gen(prompts0, max_new_tokens=256, num_beams=1, cutoff_len=512):
    if type(prompts0) == str:
        prompts0 = [prompts0]
    prompts = [f"#INPUT: {prompt} #OUTPUT: " for prompt in prompts0]
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    input_ids = input_tokens['input_ids']
    seq_len = input_ids.shape[1]
    if seq_len > cutoff_len + 5:
       # 需要进行截断处理
       input_tokens = prepare_cutoff_inputs(prompts, cutoff_len)

    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')
    
    if adapter_config and adapter_config.apply_c_mask:
        input_tokens['c_mask'] = input_tokens['attention_mask'].unsqueeze(-1).to(model.device).to(model.dtype)
    
    # print(input_tokens)
    outputs = model.generate(**input_tokens, max_new_tokens=max_new_tokens,
                             do_sample=False, num_beams=num_beams,
                             eos_token_id=tokenizer.eos_token_id)
    # 获取答案
    out_text = tokenizer.batch_decode(outputs[:, input_tokens["input_ids"].shape[1]:], skip_special_tokens=True)
    return out_text


def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts


def get_answer(answer):
    return answer.strip()


def batch_infer(prompts):
#    batch_size = 2
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        b_answers = pet_gen(batch_input, max_new_tokens=256)
        answers.extend(b_answers)
      #  print(b_answers)
    answers = [get_answer(answer) for answer in answers]
    return answers


def main():
    output_filename = os.path.join(save_dir,  '%s.json' % (model_tag))
    if "test" in dataset:
       test_dataset = dataset['test'] #.select(range(100))
       print("evaluate on test data")
    elif "validation" in dataset:
        test_dataset = dataset['validation']
        print("evaluate on validation data")
    else:
        raise Exception("no data for evaluation")
    start_time = time.time()

    records = []
    for i in range(len(test_dataset)):
        # get prompt and make sure it fits
        sample = test_dataset[i]
        prompt = sample[source_column]
        label = sample[target_column].strip()
        records.append({'prompt': prompt, 'answer': label})
    # print(prompt)
    pred_answers = batch_infer([record['prompt'] for record in records])
    gold_answers = [record['answer'] for record in records]
    run_results = {'pred_answers': pred_answers, 'gold_answers': gold_answers}

    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))
    print("evaluation done, file saved at:", output_filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--task', type=str, default="mt")
    args = parser.parse_args()
    batch_size = 16
    # output/mt/adapter0/seed42_lr20_epochs2/checkpoint-5000
    model_name_or_path = args.model_name_or_path
    adapter_config_file = f"configs/{args.model_type}.json"
    tokenizer_path = "/data/oceanus_ctr/j-mayouneng-jk/llama-7b"
 
    items = model_name_or_path.split("/")

    if args.task == "mt":
        dataset, source_column, target_column = get_mt_dataset()
    elif args.task == "xsum":
        dataset, source_column, target_column = get_xsum_dataset()
    elif args.task == "mnli":
        dataset, source_column, target_column = get_mnli_dataset()
    elif args.task == "sst2":
        dataset, source_column, target_column = get_sst2_dataset()
    else:
        raise Exception("task not handled:" + str(args.task))

    save_dir = os.path.join("results", args.task)
    os.makedirs(save_dir, exist_ok=True)

    model_tag = "#".join(model_name_or_path.split("/")[-3:])
    model_tag = model_tag.strip("#")

    print("start to load model from path:", model_name_or_path)

    with open(adapter_config_file, 'r', encoding='utf-8') as f:
        adapter_config_dict = json.load(f)

    adapter_config = AdapterConfig()
    for key, value in adapter_config_dict.items():
        if hasattr(adapter_config, key):
            setattr(adapter_config, key, value)
            print(key, value)
    print(f'peft for inference type: {args.model_type}')


    model = LlamaForCausalLM.from_pretrained(model_name_or_path,
                                             adapter_config=adapter_config,
                                             torch_dtype=torch.bfloat16, device_map="auto",
                                             ignore_mismatched_sizes=True,
                                             low_cpu_mem_usage=True)
    model.eval()

    # tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=False,
        padding_side="left",
    )
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    print("model loaded.")

    main()




