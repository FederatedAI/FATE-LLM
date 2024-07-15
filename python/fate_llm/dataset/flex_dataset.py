import os.path
from datasets import load_dataset
import re

from fate.ml.nn.dataset.base import Dataset
from typing import Union, Literal
import logging
from jinja2 import Template
from transformers import AutoTokenizer
from ruamel import yaml


logger = logging.getLogger(__name__)


def regex_replace(string, pattern, repl, count: int = 0):
    """
    adopted from lm-evaluation-harness/lm-eval/utils.py for offline use
    Parameters
    ----------
    string
    pattern
    repl
    count

    Returns
    -------

    """
    return re.sub(pattern, repl, string, count=count)


def apply_template(template, data):
    """
    adopted from lm-evaluation-harness/lm-eval/utils.py for offline use
    Parameters
    ----------
    template
    data

    Returns
    -------

    """
    return Template(template).render(data)


def tokenize_flex_dataset(raw_datasets, tokenizer, sub_domain, tokenize_format, data_part="train", save_path=None):
    tokenizer.pad_token = tokenizer.eos_token
    column_names = raw_datasets[data_part].column_names

    def tokenize_function(examples):
        texts = examples[column_names[0]]
        labels = examples[column_names[1]]
        input_ids_list = []
        attention_masks_list = []
        labels_list = []
        for text, label in zip(texts, labels):
            prefix = apply_template(tokenize_format,
                                    {"sub_domain": sub_domain,
                                     "label": label})
            entry = prefix + text
            tokenized_ret = tokenizer(entry, return_tensors="pt")
            input_ids = tokenized_ret["input_ids"]
            labels = input_ids.clone()
            attention_masks = tokenized_ret["attention_mask"]

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_masks_list.append(attention_masks)

        return dict(input_ids=input_ids_list, attention_mask=attention_masks_list, labels=labels_list)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    if save_path is not None:
        tokenized_datasets.save_to_disk(save_path)

    return tokenized_datasets


class FlexDataset(Dataset):

    def __init__(self,
                 tokenizer_path,
                 dataset_name: str,
                 load_from: Literal['jsonl', 'hf_load_from_disk', 'hf_load_dataset', 'json'] = 'json',
                 data_part: str = None,
                 config: Union[dict, str] = None,
                 need_preprocess: bool = False
                 ):

        super().__init__()
        self.tokenizer = None
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        self.dataset_name = dataset_name
        self.load_from = load_from
        self.data_part = data_part
        self.dataset = None
        self.ds = None
        self.label_key = None
        self.text_key = None
        self.augment_format = None
        self.filter_format = None
        self.few_shot_format = None
        self.tokenize_format = None
        self.random_state = None
        self.sub_domain = None
        self.label_list = None
        self.need_preprocess = need_preprocess
        self.config = config
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        self.parse_config()

    def parse_config(self, config=None):
        if config is None:
            config = self.config
        self.label_key = config.get("label_key", None)
        self.text_key = config.get("text_key", None)
        self.augment_format = config.get("augment_format", None)
        self.filter_format = config.get("filter_format", None)
        self.tokenize_format = config.get("tokenize_format", None)
        self.sub_domain = config.get("sub_domain", None)
        self.random_state = config.get("random_state", None)
        self.label_list = config.get("label_list", None)
        self.few_shot_format = config.get("few_shot_format", None)

    def sample_data(self, dataset, sample_n=5, stratified=True):
        from sklearn.model_selection import StratifiedShuffleSplit
        if stratified:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_n, random_state=self.random_state)
            _, test_index = sss.split(dataset[self.text_key], dataset[self.label_key])
            sampled_text = [dataset[self.text_key][i] for i in test_index]
            sampled_label = [dataset[self.label_key][i] for i in test_index]
        else:
            from sklearn.utils import resample
            choices = resample(list(range(len(self))),
                               replace=False,
                               n_samples=sample_n,
                               random_state=self.random_state)
            sampled_text = [dataset[self.text_key][i] for i in choices]
            sampled_label = [dataset[self.label_key][i] for i in choices]
        sampled_data = {self.text_key: sampled_text, self.label_key: sampled_label}
        return sampled_data

    def prepare_few_shot(self, shot_num=5):
        dataset = self.dataset
        if self.data_part:
            dataset = self.dataset[self.data_part]
        sampled_data = self.sample_data(dataset=dataset, sample_n=shot_num)
        # apply template
        few_shot_data = []
        for text, label in zip(sampled_data[self.text_key], sampled_data[self.label_key]):
            few_shot_data.append(apply_template(self.few_shot_format,
                                                {"text": text,
                                                 "label": label}))
        return few_shot_data

    def prepare_augment(self, few_shot_samples):
        data = []
        for i, sample in enumerate(few_shot_samples):
            query = self.augment_format + '\n' + sample
            encodeds = self.query_tokenize_function(query)
            data.append(encodeds)
        return data

    def regex_filter(self, data):
        pass

    def parse_augmented_data(self, sample_list):
        for i, sample in sample_list:
            data_list = sample.split('\n\n')


    def filter_cluster_data(self, clustered_data):
        pass

    @staticmethod
    def group_data_list(data_list, text_key, label_key):
        inputs = [entry[text_key] for entry in data_list]
        labels = [entry[label_key] for entry in data_list]
        data_dict = {text_key: inputs, label_key: labels}
        return data_dict

    def load(self, path):
        print(f"loading data from {path}, data_part: {self.data_part}")
        local_data = load_dataset('json', data_files={self.data_part: path})
        self.dataset = local_data
        if not self.need_preprocess:
            self.ds = local_data
        else:
            tokenized_ds = tokenize_flex_dataset(
                raw_datasets=local_data,
                tokenizer=self.tokenizer,
                sub_domain=self.sub_domain,
                tokenize_format=self.tokenize_format
            )
            self.ds = tokenized_ds[self.data_part]

    def query_tokenize_function(self, query):
        tokenizer = self.tokenizer

        msg = [
            {"role": "system", "content": "You are a helpful assistant. "},
            {"role": "user", "content": query}
        ]
        encoded = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)

        return encoded

    def get_raw_dataset(self):
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def get_item(self, i):
        return self.dataset[i]

    def get_item_dict(self, i):
        return {"text": self.dataset[self.text_key][i],
                "label": self.dataset[self.label_key][i]}

    def __getitem__(self, i) -> dict:
        return self.ds[i]
