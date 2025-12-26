import numpy as np
import torch


class DataCollator(object):
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        lengths = [len(instance["input_ids"]) for instance in batch]
        batch_max_len = max(lengths)
        if self.pad_to_multiple_of > 1:
            batch_max_len = int(np.ceil(batch_max_len/self.pad_to_multiple_of)*self.pad_to_multiple_of)

        input_ids = []
        labels = []
        attention_mask = []
        c_mask = []
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0
        for instance in batch:
            _input_ids = instance["input_ids"]
            pad_tokens = (batch_max_len - len(_input_ids))*[pad_token_id]  # use <unk> for padding
            _pad_len = len(pad_tokens)
            # ignore the pad token for loss
            if self.tokenizer.padding_side == "left":
                _atten_mask = pad_tokens + [1] * len(_input_ids)
                _input_ids = pad_tokens + _input_ids
                _labels = [-100]*_pad_len + instance['labels']
                _c_mask = [0]*len(pad_tokens) + instance['c_mask']
            else:
                _atten_mask = [1] * len(_input_ids) + pad_tokens
                _input_ids = _input_ids + pad_tokens
                _labels = instance['labels'] + [-100]*_pad_len
                _c_mask = instance['c_mask'] + [0]*len(pad_tokens)

            input_ids.append(_input_ids)
            labels.append(_labels)
            attention_mask.append(_atten_mask)
            c_mask.append(_c_mask)

        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = torch.LongTensor(attention_mask)
        c_mask = torch.FloatTensor(c_mask).unsqueeze(-1)
        model_inputs = {
            "input_ids": input_ids,
            "labels": labels,
            "c_mask": c_mask,
            "attention_mask": attention_mask
        }
        return model_inputs







