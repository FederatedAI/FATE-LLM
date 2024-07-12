import os.path

from fate.ml.nn.dataset.base import Dataset
from typing import List, Dict, Union, Literal
import logging
from jinja2 import Template
from transformers import AutoTokenizer
from ruamel import yaml


logger = logging.getLogger(__name__)


class FlexDataset(Dataset):

    def __init__(self,
                tokenizer_path,
                input_template: str,
                output_template: str,
                max_input_length: int = 256,
                max_target_length: int = 256,
                load_from: Literal['jsonl', 'hf_load_from_disk', 'hf_load_dataset'] = 'hf_load_from_disk',
                split_key: str = None,
                config: Union[dict, str] = None
                ):

        super().__init__()
        self.tokenizer = None
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.dataset = None
        self.data_dict = None
        self.load_from = load_from
        self.input_template = Template(input_template)
        self.output_template = Template(output_template)
        self.split_key = split_key
        self.max_seq_length = max_input_length + max_target_length + 1
        self.label_key = None
        self.text_key = None
        self.augment_format = None
        self.filter_format = None
        self.generate_format = None
        self.random_state = None
        self.sub_domain = None
        self.label_list = None
        self.config = config
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.load(f)

    def parse_config(self, config):
        self.label_key = config.get("label_key", None)
        self.text_key = config.get("text_key", None)
        self.augment_format = config.get("augment_format", None)
        self.filter_format = config.get("filter_format", None)
        self.generate_format = config.get("generate_format", None)
        self.sub_domain = config.get("sub_domain", None)
        self.random_state = config.get("random_state", None)
        self.label_list = config.get("label_list", None)

    def sample_data(self, sample_n=5, stratified=True):
        from sklearn.model_selection import StratifiedShuffleSplit
        if stratified:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_n, random_state=self.random_state)
            _, test_index = sss.split(self.data_dict["inputs"], self.data_dict["labels"])
            sampled_text = [self.data_dict['inputs'][i] for i in test_index]
            sampled_label = [self.data_dict['kabel'][i] for i in test_index]
            sampled_data = {"text": sampled_text, "label": sampled_label}
        else:
            from sklearn.utils import resample
            choices = resample(list(range(len(self))),
                               replace=False,
                               n_samples=sample_n,
                               random_state=self.random_state)
            sampled_data = [self.get_item_dict(i) for i in choices]
            sampled_data = FlexDataset.group_data_list(sampled_data, self.text_key, self.label_key)
        return sampled_data

    def generate_text(self):
        from lm_eval.utils import apply_template
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.label_list is None:
            self.label_list = list(set(self.data_dict['labels']))
        ids_dict = {}
        for label in self.label_list:
            prompt = apply_template(self.generate_format,
                                    {"sub_domain": self.sub_domain,
                                     "label": label})
            prompt_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids']
            ids_dict[label] = prompt_ids
        return ids_dict

    def prepare_few_shot(self, shot_num=5):

        pass

    def augment_data(self):
        pass

    def cluster_data(self, data):
        pass

    @staticmethod
    def group_data_list(data_list, text_key, label_key):
        inputs = [entry[text_key] for entry in data_list]
        labels = [entry[label_key] for entry in data_list]
        data_dict = {"inputs": inputs, "labels": labels}
        return data_dict

    def load(self, path):
        if self.load_from == 'hf_load_from_disk':
            import datasets
            self.dataset = datasets.load_from_disk(path)
            if self.split_key is not None:
                self.dataset = self.dataset[self.split_key]
            self.dataset = [i for i in self.dataset]
        elif self.load_from == 'jsonl':
            import json
            if path.endswith('.jsonl'):
                with open(path, 'r') as f:
                    json_lines = f.read().split('\n')
                self.dataset = []
                for i in json_lines:
                    try:
                        self.dataset.append(json.loads(i))
                    except:
                        print('skip line')
        elif self.load_from =='json':
            import json

            if path.endswith('.json'):
                with open(path, 'r') as f:
                    self.dataset = json.load(f)
                    self.data_dict = FlexDataset.group_data_list(self.dataset, self.text_key, self.label_key)
            elif os.path.isdir(path):
                data_dict = {}
                # Check if base_dir has any subdirectories
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if subdirs:
                    if self.sub_domain is None:
                        raise ValueError(f"sub_domain is required for loading data from directory")
                    subdir = [d if d == self.sub_domain for d in subdirs][0]
                    subdir_path = os.path.join(path, subdir)
                    for file_name in os.listdir(subdir_path):
                        if file_name.endswith(".json"):
                            if not self.split_key or (self.split_key and file_name.startswith(self.split_key)):
                                file_path = os.path.join(subdir_path, file_name)
                                with open(file_path, 'r') as f:
                                    file_content = json.load(f)
                                    data_dict= FlexDataset.group_data_list(file_content,
                                                                           self.text_key,
                                                                           self.label_key)

                    self.data_dict = data_dict
                else:
                    for file_name in os.listdir(path):
                        if file_name.endswith(".json"):
                            # only 1 effective file will be kept
                            if not self.split_key or (self.split_key and file_name.startswith(self.split_key)):
                                file_path = os.path.join(path, file_name)
                                with open(file_path, 'r') as f:
                                    file_content = json.load(f)
                                    self.dataset = file_content

        elif self.load_from == 'hf_load_dataset':
            from datasets import load_dataset
            self.dataset = load_dataset(path)
            if self.split_key is not None:
                self.dataset = self.dataset[self.split_key]
            self.dataset = [i for i in self.dataset]
        else:
            raise ValueError('unknown load format')

        if not isinstance(self.dataset, list) or not isinstance(self.dataset[0], dict):
            logger.warn('loaded dataset is expected to be a list of dict')

    def tokenize_function(self, query):
        tokenizer = self.tokenizer

        msg = [
            {"role": "system", "content": "You are a helpful assistant. "},
            {"role": "user", "content": query}
        ]
        encodeds = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)

        return encodeds

    def get_raw_dataset(self):
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def get_item(self, i):
        return self.dataset[i]

    def get_item_dict(self, i):
        return {"text": self.data_dict["text"][i],
                "label": self.data_dict["label"][i]}

    def __getitem__(self, i) -> dict:
        item = self.get_item(i)
        return item
