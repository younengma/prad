"""
@author: mayouneng
@email: youneng.ma@qq.com
@time: 2024/5/11 11:11
@DESC: 

"""
import torch
from arguments import DataTrainingArguments
from log_util import get_logger

logger = get_logger(__name__)

SPECIAL_TOKENS = {
    "web_nlg": ['<|subject|>', '<|property|>', '<|object|>', '<|triple_sep|>'],
    "e2e": [
          "<|name|>", "<|eatType|>", "<|familyFriendly|>", "<|priceRange|>", "<|food|>", "<|near|>", "<|area|>", "<|customerRating|>"],
      # "smd":["<bos_smd>", "<eos_smd>", "<user>", "<system>", "<KB>"]
                  }
# ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
#                          'additional_special_tokens': ('<speaker1>', '<speaker2>', '<persona>',)}
ATTR_TO_SPECIAL_TOKEN = {
    'pad_token': '<|end|>', 'eos_token': '<|end|>',
    'additional_special_tokens': [
        '<|subject|>', '<|property|>', '<|object|>', '<|triple_sep|>',  # wen_nlg
        '<|start|>', '<|sep|>', "<|source|>", "<target>",  # common generation
        "<|name|>", "<|eatType|>", "<|familyFriendly|>", "<|priceRange|>", "<|food|>", "<|near|>", "<|area|>" , "<|customerRating|>" # e2e
    ]}

task_column_mapping = {
    "e2e": ("review_body", "review_title"),
    "web_nlg": ("description", "abstract"),
}


def add_special_tokens_(model, tokenizer, prompt_length=0):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    # add task embedding
    num_added_tokens = tokenizer.add_special_tokens(
        ATTR_TO_SPECIAL_TOKEN)  # returns 0 and doesn't add if they are already there
    # add soft prompt if required.
    if prompt_length > 0:
        num_added_tokens += tokenizer.add_special_tokens({'additional_special_tokens': [f"<|p{i}|>"
                                                                                        for i in range(prompt_length)]})
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"newly added token number: {num_added_tokens}")


def get_preprocess_funcs(data_args: DataTrainingArguments, tokenizer, model, prompt_length=0):
    add_special_tokens_(model, tokenizer, prompt_length)

    # Get the column names for input/target.
    dataset_columns = task_column_mapping.get(data_args.dataset_name, None)
    source_column = data_args.source_column
    target_column = data_args.target_column

    if source_column is None:
        source_column = dataset_columns[0] if dataset_columns is not None else "source"

    if target_column is None:
        target_column = dataset_columns[1] if dataset_columns is not None else "target"

    # list of special tokens
    # 1. common
    start_id = tokenizer.encode('<|start|>')
    source_id = tokenizer.encode("<|source|>")
    sep_id = tokenizer.encode("<|sep|>")
    target_id = tokenizer.encode("<target>")
    end_id = tokenizer.encode('<|end|>')

    # 2. task specific
    subject_id, property_id, object_id, triple_sep_id = (tokenizer.encode(v) for v in SPECIAL_TOKENS['web_nlg'])
    e2e_token_dict = {_: tokenizer.encode(_) for _ in SPECIAL_TOKENS['e2e']}

    def preprocess_webnlg(examples):
        input_ids = []
        labels = []
        c_masks = []
        token_type_ids = []
        for i in range(len(examples[source_column])):
            # remove pairs where at least one record is None
            if examples[source_column][i] and examples[target_column][i]:
                input_triples = examples[source_column][i]
                # 1. process the source
                i_source_ids = [] + start_id
                i_token_type_ids = [] + start_id
                for triple in input_triples:
                    # '<|subject|>', '<|property|>', '<|object|>'
                    sub, rel, obj = (_.strip() for _ in triple.split("|"))
                    sub_ids = tokenizer.encode(sub)
                    property_ids = tokenizer.encode(rel)
                    obj_ids = tokenizer.encode(obj)
                    i_source_ids += subject_id + sub_ids + \
                                    property_id + property_ids + \
                                    object_id + obj_ids + triple_sep_id
                    i_token_type_ids += subject_id + subject_id * len(sub_ids) \
                                        + property_id + property_id * len(property_ids) \
                                        + object_id + object_id * len(obj_ids) + triple_sep_id
                # remove the last triple_sep_id
                i_source_ids = i_source_ids[:-1]
                i_token_type_ids = i_token_type_ids[:-1]

                c_masks.append(torch.ones((len(i_source_ids) + 1), dtype=torch.long))
                # 2. process the target
                summary_ids = tokenizer.encode(examples[target_column][i])
                i_token_type_ids += sep_id + target_id * len(summary_ids) + end_id
                i_input_ids = i_source_ids + sep_id + summary_ids + end_id
                i_len = len(i_source_ids) + 1  # also ignore the sep for loss
                i_labels = i_len * [-100] + i_input_ids[i_len:]
                input_ids.append(i_input_ids)
                labels.append(i_labels)
                token_type_ids.append(i_token_type_ids)

        # currently we do not use padding to the max_length, each is an example
        if data_args.ignore_token_type_ids:
            model_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "c_masks": c_masks,
            }
        else:
            model_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "c_masks": c_masks,
                "token_type_ids": token_type_ids
            }
        return model_inputs

    def preprocess_webnlg_predict(examples):
        # encoder-decoder 架构的输入和 decoder-only的输入输出形式是不一样的
        # eval/test的时候，注意只能看到condition的部分，不能像训练的时候采用teacher-force的方式来进行evaluate了。
        input_ids = []
        labels = []
        c_masks = []
        token_type_ids = []
        for i in range(len(examples[source_column])):
            if examples[source_column][i] and examples[target_column][i]:
                input_triples = examples[source_column][i]
                # 1. process the source
                i_source_ids = [] + start_id
                i_token_type_ids = [] + start_id
                for triple in input_triples:
                    # '<|subject|>', '<|property|>', '<|object|>'
                    sub, rel, obj = (_.strip() for _ in triple.split("|"))
                    sub_ids = tokenizer.encode(sub)
                    property_ids = tokenizer.encode(rel)
                    obj_ids = tokenizer.encode(obj)
                    i_source_ids += subject_id + sub_ids + \
                                    property_id + property_ids + \
                                    object_id + obj_ids + triple_sep_id
                    i_token_type_ids += subject_id + subject_id * len(sub_ids) \
                                        + property_id + property_id * len(property_ids) \
                                        + object_id + object_id * len(obj_ids) + triple_sep_id
                # remove the last triple sep_id
                i_source_ids = i_source_ids[:-1]
                i_token_type_ids = i_token_type_ids[:-1]

                c_masks.append(torch.ones((len(i_source_ids) + 1), dtype=torch.long))
                # 2. preprocess the target
                i_labels = tokenizer.encode(examples[target_column][i]) + end_id
                i_input_ids = i_source_ids + sep_id
                i_token_type_ids += sep_id

                input_ids.append(i_input_ids)
                labels.append(i_labels)
                token_type_ids.append(i_token_type_ids)

        if data_args.ignore_token_type_ids:
            model_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "c_masks": c_masks,
            }
        else:
            model_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "c_masks": c_masks,
                "token_type_ids": token_type_ids

            }
        return model_inputs

    def preprocess_e2e(examples):
        input_ids = []
        labels = []
        c_masks = []
        token_type_ids = []
        for i in range(len(examples[source_column])):
            # remove pairs where at least one record is None
            if examples[source_column][i] and examples[target_column][i]:
                mean_repr = examples[source_column][i]
                items = mean_repr.split(', ')
                # 1. process the source
                i_source_ids = [] + start_id
                i_token_type_ids = [] + start_id
                for item in items:
                    key, value = item.strip().split('[')
                    if key == "customer rating":
                        key = "customerRating"
                    key = f'<|{key}|>'
                    value = value.replace(']', "")
                    key_id = e2e_token_dict[key]
                    value_id = tokenizer.encode(value)
                    i_source_ids += key_id + value_id
                    i_token_type_ids += key_id + key_id*len(value_id)

                c_masks.append(torch.ones((len(i_source_ids) + 1), dtype=torch.long))
                # 2. process the target
                summary_ids = tokenizer.encode(examples[target_column][i])
                i_token_type_ids += sep_id + target_id * len(summary_ids) + end_id
                i_input_ids = i_source_ids + sep_id + summary_ids + end_id
                i_len = len(i_source_ids) + 1  # also ignore the sep for loss
                i_labels = i_len * [-100] + i_input_ids[i_len:]
                input_ids.append(i_input_ids)
                labels.append(i_labels)
                token_type_ids.append(i_token_type_ids)

        # currently we do not use padding to the max_length, each is an example
        if  data_args.ignore_token_type_ids:
            model_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "c_masks": c_masks,
            }
        else:
            model_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "c_masks": c_masks,
                "token_type_ids": token_type_ids
            }
        return model_inputs

    def preprocess_e2e_predict(examples):
        # encoder-decoder 架构的输入和 decoder-only的输入输出形式是不一样的
        # eval/test的时候，注意只能看到condition的部分，不能像训练的时候采用teacher-force的方式来进行evaluate了。
        input_ids = []
        labels = []
        c_masks = []
        token_type_ids = []
        for i in range(len(examples[source_column])):
            if examples[source_column][i] and examples[target_column][i]:
                mean_repr = examples[source_column][i]
                items = mean_repr.split(', ')
                # 1. process the source
                i_source_ids = [] + start_id
                i_token_type_ids = [] + start_id
                for item in items:
                    key, value = item.strip().split('[')
                    if key == "customer rating":
                        key = "customerRating"
                    key = f'<|{key}|>'
                    value = value.replace(']', "")
                    key_id = e2e_token_dict[key]
                    value_id = tokenizer.encode(value)
                    i_source_ids += key_id + value_id
                    i_token_type_ids += key_id + key_id * len(value_id)

                c_masks.append(torch.ones((len(i_source_ids) + 1), dtype=torch.long))
                # 2. preprocess the target
                i_labels = tokenizer.encode(examples[target_column][i]) + end_id
                i_input_ids = i_source_ids + sep_id
                i_token_type_ids += sep_id

                input_ids.append(i_input_ids)
                labels.append(i_labels)
                token_type_ids.append(i_token_type_ids)

        if data_args.ignore_token_type_ids:
            model_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "c_masks": c_masks,
            }
        else:
            model_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "c_masks": c_masks,
                "token_type_ids": token_type_ids

            }
        return model_inputs


    def preprocess_mnli(examples):
        input_ids = []
        labels = []
        c_masks = []
        token_type_ids = []
        for i in range(len(examples[source_column])):
            # remove pairs where at least one record is None
            if examples[source_column][i] and examples[target_column][i]:
                # 1. process the source
                source = examples[source_column][i]
                source = source.lstrip('[hypothesis]:')
                hypothesis, premise = (_.strip() for _ in source.split("[premise]:"))
                h_ids = tokenizer.encode(hypothesis)
                p_ids = tokenizer.encode(premise)
                # use subject_id and object_id as special tokens for the premise and hypothesis
                i_source_ids = (start_id + subject_id + h_ids + object_id + p_ids)[:data_args.max_source_length] + sep_id
                i_token_type_ids = (start_id + subject_id*(len(h_ids) + 1) + object_id*(len(p_ids) + 1))[:data_args.max_source_length] + sep_id
                c_masks.append(torch.ones(len(i_source_ids), dtype=torch.long))

                # 2. process the target
                target_ids = tokenizer.encode(examples[target_column][i])[:data_args.max_target_length]
                i_token_type_ids += target_id * len(target_ids) + end_id
                i_input_ids = i_source_ids + target_ids + end_id

                i_len = len(i_source_ids)
                i_labels = i_len * [-100] + i_input_ids[i_len:]
                input_ids.append(i_input_ids)
                labels.append(i_labels)
                token_type_ids.append(i_token_type_ids)
        # currently we do not use padding to the max_length, each is an example
        if data_args.ignore_token_type_ids:
            model_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "c_masks": c_masks,
            }
        else:
            model_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "c_masks": c_masks,
                "token_type_ids": token_type_ids
            }
        return model_inputs

    def preprocess_mnli_predict(examples):
        input_ids = []
        labels = []
        c_masks = []
        token_type_ids = []
        for i in range(len(examples[source_column])):
            if examples[source_column][i] and examples[target_column][i]:
                i_labels = tokenizer.encode(examples[target_column][i]) + end_id

                source = examples[source_column][i]
                source = source.lstrip('[hypothesis]:')
                hypothesis, premise = (_.strip() for _ in source.split("[premise]:"))
                h_ids = tokenizer.encode(hypothesis)
                p_ids = tokenizer.encode(premise)

                i_input_ids = start_id + subject_id + h_ids + object_id + p_ids + sep_id
                i_token_type_ids = start_id + subject_id*(len(h_ids) + 1) + object_id*(len(p_ids) + 1) + sep_id
                c_masks.append(torch.ones((len(i_input_ids)), dtype=torch.long))
                input_ids.append(i_input_ids)
                labels.append(i_labels)
                token_type_ids.append(i_token_type_ids)

        if data_args.ignore_token_type_ids:
            model_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "c_masks": c_masks,
            }
        else:
            model_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "c_masks": c_masks,
                "token_type_ids": token_type_ids

            }
        return model_inputs


    def preprocess(examples):
        input_ids = []
        labels = []
        c_masks = []
        token_type_ids = []
        for i in range(len(examples[source_column])):
            # remove pairs where at least one record is None
            if examples[source_column][i] and examples[target_column][i]:
                # 1. process the source
                i_source_ids = start_id + tokenizer.encode(examples[source_column][i])[:data_args.max_source_length] + sep_id
                i_token_type_ids = start_id + (len(i_source_ids) - 2)*source_id + sep_id
                c_masks.append(torch.ones(len(i_source_ids), dtype=torch.long))

                # 2. process the target
                target_ids = tokenizer.encode(examples[target_column][i])[:data_args.max_target_length]
                i_token_type_ids += target_id * len(target_ids) + end_id
                i_input_ids = i_source_ids + target_ids + end_id

                i_len = len(i_source_ids)
                i_labels = i_len * [-100] + i_input_ids[i_len:]
                input_ids.append(i_input_ids)
                labels.append(i_labels)
                token_type_ids.append(i_token_type_ids)

        # currently we do not use padding to the max_length, each is an example
        if data_args.ignore_token_type_ids:
            model_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "c_masks": c_masks,
            }
        else:
            model_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "c_masks": c_masks,
                "token_type_ids": token_type_ids
            }
        return model_inputs

    def preprocess_predict(examples):
        input_ids = []
        labels = []
        c_masks = []
        token_type_ids = []
        for i in range(len(examples[source_column])):
            if examples[source_column][i] and examples[target_column][i]:
                i_labels = tokenizer.encode(examples[target_column][i]) + end_id
                i_input_ids = start_id + tokenizer.encode(examples[source_column][i])[:data_args.max_source_length] + sep_id
                i_token_type_ids = start_id + (len(i_input_ids) - 2) * source_id + sep_id
                c_masks.append(torch.ones((len(i_input_ids)), dtype=torch.long))

                input_ids.append(i_input_ids)
                labels.append(i_labels)
                token_type_ids.append(i_token_type_ids)

        if data_args.ignore_token_type_ids:
            model_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "c_masks": c_masks,
            }
        else:
            model_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "c_masks": c_masks,
                "token_type_ids": token_type_ids

            }
        return model_inputs

    if data_args.task_name == "web_nlg":
        return preprocess_webnlg, preprocess_webnlg_predict
    elif data_args.task_name == "e2e":
        return preprocess_e2e, preprocess_e2e_predict
    elif data_args.task_name == "mnli":
        return preprocess_mnli, preprocess_mnli_predict
    else:
        return preprocess, preprocess_predict
