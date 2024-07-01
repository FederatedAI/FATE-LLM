from fate.ml.nn.dataset.base import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from typing import List, Dict, Union, Literal
import logging
from jinja2 import Template
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


class InputOutputDataset(Dataset):

    def __init__(self, 
                tokenizer_path,
                input_template: str,
                output_template: str,
                max_input_length: int = 256, 
                max_target_length: int = 256,
                load_from: Literal['jsonl', 'hf_load_from_disk', 'hf_load_dataset'] = 'hf_load_from_disk',
                split_key: str = None
                ):

        super().__init__()
        self.tokenizer = None
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.dataset = None
        self.load_from = load_from
        self.input_template = Template(input_template)
        self.output_template = Template(output_template)
        self.split_key = split_key
        self.max_seq_length = max_input_length + max_target_length + 1

    def load(self, path):
        if self.load_from == 'hf_load_from_disk':
            import datasets
            self.dataset = [i for i in datasets.load_from_disk(path)]
            if self.split_key is not None:
                self.dataset = self.dataset[self.split_key]
        elif self.load_from == 'jsonl':
            import json
            with open(path, 'r') as f:
                json_lines = f.read().split('\n')
            self.dataset = []
            for i in json_lines:
                try:
                    self.dataset.append(json.loads(i))
                except:
                    print('skip line')
        elif self.load_from == 'hf_load_dataset':
            from datasets import load_dataset
            self.dataset = load_dataset(path)
            if self.split_key is not None:
                self.dataset = self.dataset[self.split_key]
        else:
            raise ValueError('unknown load format')

        if not isinstance(self.dataset, list) or not isinstance(self.dataset[0], dict):
            logger.warn('loaded dataset is expected to be a list of dict')

    def get_raw_dataset(self):
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def get_str_item(self, i) -> dict:

        data_item = self.dataset[i]
        in_ = self.input_template.render(**data_item)
        out_ = self.output_template.render(**data_item)
        return {
            'input': in_,
            'output': out_
        }

    def _process_item(self, data_item):

        a_ids = self.tokenizer.encode(text=data_item['input'], add_special_tokens=True, truncation=True,
                                      max_length=self.max_input_length)
        b_ids = self.tokenizer.encode(text=data_item['output'], add_special_tokens=False, truncation=True,
                                      max_length=self.max_target_length)

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]

        pad_len = self.max_seq_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.tokenizer.pad_token_id] * pad_len
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": input_ids,
            "labels": labels
        }

    def get_tokenized_item(self, i) -> dict:   

        str_item = self.get_str_item(i)
        ret_dict = self._process_item(str_item)
        return ret_dict

    def __getitem__(self, i) -> dict:
        item = self.get_tokenized_item(i)
        return item
