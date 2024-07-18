#
#  Copyright 2024 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import re
from datasets import load_dataset
from fastchat.model import get_conversation_template
from jinja2 import Template
from ruamel import yaml
from transformers import AutoTokenizer
from typing import Union, Literal

from fate.ml.nn.dataset.base import Dataset

logger = logging.getLogger(__name__)


"""
Implementation of FDKT augmentation process, adopted from https://arxiv.org/abs/2405.14212
"""


def jinja_to_regex(template, placeholders):
    regex_template = re.escape(template)
    for placeholder in placeholders:
        regex_template = regex_template.replace(re.escape(f'{{{{{placeholder}}}}}'), f'(?P<{placeholder}>.*?)')
    pattern = re.compile(regex_template)

    return pattern


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


def tokenize_flex_dataset(raw_datasets, tokenizer, sub_domain, tokenize_format, text_key, label_key, data_part="train",
                          save_path=None, max_prompt_len=256):
    tokenizer.pad_token = tokenizer.eos_token
    column_names = raw_datasets[data_part].column_names

    def tokenize_function(examples):
        texts = tokenizer(examples[text_key])

        label_processed = [apply_template(tokenize_format,{"sub_domain": sub_domain,"label": label})
                           for label in examples[label_key]]
        labels = tokenizer(label_processed)
        input_ids = [i2 + i1 for i1, i2 in zip(texts['input_ids'], labels['input_ids'])]
        attention_mask = [i2 + i1 for i1, i2 in zip(texts['attention_mask'], labels['attention_mask'])]

        """
        cut off max prompt length
        """
        input_ids = [t[: max_prompt_len] for t in input_ids]
        attention_mask = [t[: max_prompt_len] for t in attention_mask]

        out = {"input_ids": input_ids,
               "attention_mask": attention_mask,
               "labels": input_ids}
        return out

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
                 load_from: Literal['json'] = 'json',
                 data_part: str = None,
                 config: Union[dict, str] = None,
                 need_preprocess: bool = True,
                 random_state: int = None,
                 max_prompt_len: int = 256,
                 select_num: int = None,
                 few_shot_num_per_label: int = None
                 ):

        super().__init__()
        self.tokenizer = None
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        self.dataset_name = dataset_name
        self.load_from = load_from
        self.data_part = data_part
        self.random_state = random_state
        self.need_preprocess = need_preprocess
        self.max_prompt_len = max_prompt_len
        self.select_num = select_num
        self.dataset = None
        self.ds = None
        self.label_key = None
        self.text_key = None
        self.augment_format = None
        self.filter_format = None
        self.few_shot_format = None
        self.tokenize_format = None
        self.sub_domain = None
        self.label_list = None
        self.text_with_label_format = None
        self.few_shot_num_per_label = few_shot_num_per_label
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
        self.label_list = config.get("label_list", None)
        self.few_shot_format = config.get("few_shot_format", None)
        self.text_with_label_format = config.get("text_with_label_format", None)
        if self.few_shot_num_per_label is None:
            self.few_shot_num_per_label = config.get("few_shot_num_per_label", 2)

    def get_generate_prompt(self, tokenize=True, return_tensors="pt"):
        prompt_list = [apply_template(self.tokenize_format,
                                      {"sub_domain": self.sub_domain,
                                       "label": label}) for label in self.label_list]
        if tokenize:
            tokenized_prompts = self.tokenizer(prompt_list, return_tensors=return_tensors)
            prompt_list = tokenized_prompts['input_ids']

        return {label: prompt for label, prompt in zip(self.label_list, prompt_list)}

    @staticmethod
    def construct_prompt_list(samples_dict, num_shot_per_label, prompt_num, format_template, random_state=None):
        from sklearn.utils import resample
        from collections import deque

        label_samples = {label: deque(resample(samples,
                                               replace=False,
                                               n_samples=len(samples))) for label, samples in samples_dict.items()}
        def get_samples_for_label(label):
            samples = []
            while len(samples) < num_shot_per_label:
                remaining_needed = num_shot_per_label - len(samples)
                if len(label_samples[label]) < remaining_needed:
                    batch_samples = list(label_samples[label])
                    samples.extend(batch_samples)
                    # reset to allow repetition
                    label_samples[label] = deque(resample(samples_dict[label],
                                                          replace=False,
                                                          n_samples=len(samples_dict[label])))
                else:
                    batch_samples = [label_samples[label].popleft() for _ in range(remaining_needed)]
                    samples.extend(batch_samples)
            return samples

        result = []
        for _ in range(prompt_num):
            prompt = ''
            for label in samples_dict.keys():
                samples = get_samples_for_label(label)
                for text in samples:
                    prompt += apply_template(format_template, {"text": text, "label": label})
            result.append(prompt)
        return result

    @staticmethod
    def group_text_label_list(text_list, label_list):
        group_data = [{"text": text, "label": label} for text, label in zip(text_list, label_list)]
        return group_data

    def prepare_few_shot(self, text_list, label_list, aug_prompt_num):
        from collections import defaultdict
        data_dict = defaultdict(list)
        for text, label in zip(text_list, label_list):
            # in case extra labels are present, ignore
            if label in self.label_list:
                data_dict[label].append(text)
        few_shot_list = FlexDataset.construct_prompt_list(samples_dict=data_dict,
                                                          num_shot_per_label=self.few_shot_num_per_label,
                                                          prompt_num=aug_prompt_num,
                                                          format_template=self.few_shot_format,
                                                          random_state=self.random_state)

        return few_shot_list

    def prepare_augment(self, text_list, label_list, aug_prompt_num):
        few_shot_samples = self.prepare_few_shot(text_list, label_list, aug_prompt_num)
        result = []
        instruction = apply_template(self.augment_format, {"sub_domain": self.sub_domain})
        for i, sample in enumerate(few_shot_samples):
            query =  instruction + '\n' + sample
            formatted_query = self.apply_chat_template(query)
            result.append(formatted_query)
        return result

    def abstract_from_augmented(self, sample_list):
        regex_pattern = jinja_to_regex(self.augment_format, ["label", "text"])
        res = {'inputs': [], 'labels': []}
        for sample in sample_list:
            data_list = sample.split('\n\n')
            for entry in data_list:
                match = regex_pattern.match(entry)
                if match:
                    if isinstance(self.label_list[0], int):
                        label = int(match.groupdict().get('label'))
                    elif isinstance(self.label_list[0], float):
                        label = float(match.groupdict().get('label'))
                    else:
                        label = match.groupdict().get('label')
                    text = match.groupdict().get('text').strip('</s>')
                    if label and text:
                        res['inputs'].append(text)
                        res['labels'].append(label)
        return res

    def prepare_query_to_filter_clustered(self, clustered_sentences_list, clustered_labels_list):
        prompt_list = []
        for clustered_sentences, clustered_labels in zip(clustered_sentences_list, clustered_labels_list):
            text_with_label = ''
            for i in range(len(clustered_sentences)):
                formatted_entry = apply_template(self.text_with_label_format, {"i": i,
                                                                               "text": clustered_sentences[i],
                                                                               "label": clustered_labels[i]})
                text_with_label += formatted_entry
            cluster_query = apply_template(self.filter_format, {"text_with_label": text_with_label})
            prompt_list.append(self.apply_chat_template(cluster_query))
        return prompt_list

    def parse_clustered_response(self, clustered_sentence, clustered_labels, response_list):
        """
        Parse the response from the clustering model and filter the data per cluster.
        :param clustered_sentence: nested list of clustered sentences
        :param clustered_labels: nested list of clustered labels
        :param response_list: list of responses from the clustering model
        """
        def parse_response(response):
            pattern = r'The eligible samples:\s*((?:\b\d+\b[\s.,]*)+)'
            matches = re.search(pattern, response, re.MULTILINE)
            if matches:
                digits = [int(i) for i in re.findall(r'\b\d+\b', matches.group())]
            else:
                digits = []
            return list(set(digits))

        filtered_text_list = []
        filtered_label_list = []
        for i in range(len(clustered_sentence)):
            parsed_response = parse_response(response_list[i])
            for idx in parsed_response:
                if idx < len(clustered_sentence[i]):
                    filtered_label_list.append(clustered_labels[i][idx])
                    filtered_text_list.append(clustered_sentence[i][idx])
        return filtered_text_list, filtered_label_list

    @staticmethod
    def group_data_list(data_list, text_key, label_key):
        inputs = [entry[text_key] for entry in data_list]
        labels = [entry[label_key] for entry in data_list]
        data_dict = {text_key: inputs, label_key: labels}
        return data_dict

    def load(self, path):
        local_data = load_dataset('json', data_files={self.data_part: path})
        self.dataset = local_data
        if not self.need_preprocess:
            self.ds = local_data
        else:
            tokenized_ds = tokenize_flex_dataset(
                raw_datasets=local_data,
                tokenizer=self.tokenizer,
                sub_domain=self.sub_domain,
                tokenize_format=self.tokenize_format,
                text_key=self.text_key,
                label_key=self.label_key
            )
            self.ds = tokenized_ds[self.data_part]

        if self.select_num is not None:
            self.ds = self.ds.select(range(self.select_num))

    def apply_chat_template(self, query):
        tokenizer = self.tokenizer

        if "llama-3" in self.tokenizer_path.lower():
            msg = [
                {"role": "system", "content": "You are a helpful assistant. "},
                {"role": "user", "content": query}
            ]
            prompt = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        else:
            conv = get_conversation_template(self.tokenizer_path)
            conv.append_message(conv.roles[0], query)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

        return prompt

    def get_raw_dataset(self):
        return self.dataset

    def __len__(self):
        return len(self.ds)

    def get_item(self, i):
        return self.dataset[self.data_part][i]

    def get_item_dict(self, i):
        return {"text": self.dataset[self.data_part][self.text_key][i],
                "label": self.dataset[self.data_part][self.label_key][i]}

    def __getitem__(self, i) -> dict:
        return self.ds[i]
