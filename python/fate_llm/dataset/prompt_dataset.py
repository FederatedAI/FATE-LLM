#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
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
import copy
import json

import datasets
import torch
from fate.ml.nn.dataset.base import Dataset
from ..data.tokenizers.cust_tokenizer import get_tokenizer


PROMPT_TEMPLATE = "{prompt}"


class PromptDataset(Dataset):
    def __init__(self,
                 text_max_length=512,
                 tokenizer_name_or_path=None,
                 trust_remote_code=False,
                 padding=False,
                 padding_side='left',
                 pad_token=None,
                 pad_token_id=None,
                 bos_token_id=None,
                 eos_token_id=None,
                 add_eos_token=True,
                 prompt_template=None,
                 add_special_tokens=False,
                 prompt_column="content",
                 response_column="summary",
                 max_prompt_length=256,
                 file_type="jsonl",
                 num_proc=4,
                 ):

        super(PromptDataset, self).__init__()
        self.tokenizer = None
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.padding = padding
        self.add_special_tokens = add_special_tokens
        self.max_prompt_length = max_prompt_length
        self.text_max_length = text_max_length

        self.tokenizer = get_tokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path,
            trust_remote_code=trust_remote_code,
            pad_token=pad_token,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            padding_side=padding_side,
            add_eos_token=add_eos_token,
        )

        self.prompt_template = prompt_template if prompt_template else PROMPT_TEMPLATE
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.file_type = file_type
        self.num_proc = num_proc
        self._data = None

    def load(self, file_path):
        if "jsonl" in self.file_type:
            prompts = []
            responses = []
            with open(file_path, "r") as fin:
                for line in fin:
                    line = json.loads(line)
                    prompts.append(line[self.prompt_column])
                    responses.append(line[self.response_column])

            ds = datasets.Dataset.from_dict({self.prompt_column: prompts, self.response_column: responses})
        else:
            ds = datasets.load_from_disk(file_path)

        self._data = ds.map(
            self._process_data,
            fn_kwargs={"tokenizer": self.tokenizer,
                       "prompt_template": self.prompt_template,
                       "prompt_column": self.prompt_column,
                       "response_column": self.response_column,
                       "max_prompt_length": self.max_prompt_length,
                       "max_length": self.text_max_length
                       },
            batched=True,
            remove_columns=ds.column_names,
            num_proc=self.num_proc,
        )

        max_length = None
        for d in self._data:
            if max_length is None:
                max_length = len(d["input_ids"])
            else:
                max_length = max(max_length, len(d["input_ids"]))

        self._data = self._data.map(
            self._pad_to_max_length,
            batched=True,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "max_length": max_length
            },
            num_proc=self.num_proc
        )

    @staticmethod
    def _process_data(examples, tokenizer, prompt_template, prompt_column,
                      response_column, max_prompt_length, max_length):
        prompts = examples[prompt_column]
        responses = examples[response_column]

        processed_data = dict()
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        for _prompt, _response in zip(prompts, responses):
            if isinstance(_response, list):
                _response = _response[0]
            _prompt = prompt_template.format_map(dict(prompt=_prompt))
            prompt_encoded = tokenizer(_prompt)
            if len(prompt_encoded['input_ids']) > 0 and prompt_encoded['input_ids'][-1] in tokenizer.all_special_ids:
                prompt_encoded['input_ids'] = prompt_encoded['input_ids'][:-1]
                prompt_encoded['attention_mask'] = prompt_encoded['attention_mask'][:-1]

            target_encoded = tokenizer(_response)
            if len(target_encoded['input_ids']) > 0 and target_encoded['input_ids'][-1] in tokenizer.all_special_ids:
                target_encoded['input_ids'] = target_encoded['input_ids'][:-1]
                target_encoded['attention_mask'] = target_encoded['attention_mask'][:-1]

            prompt_ids = prompt_encoded["input_ids"][: max_prompt_length]
            prompt_attention_mask = prompt_encoded["attention_mask"][:max_prompt_length]

            target_ids = target_encoded["input_ids"][: max_length - len(prompt_ids) - 1]
            target_attention_mask = target_encoded["attention_mask"][: max_length - len(prompt_ids) - 1]

            if tokenizer.bos_token_id is not None:
                seq_length = len(prompt_ids) + 1
                input_ids = prompt_ids + [tokenizer.bos_token_id] + target_ids + [tokenizer.eos_token_id]
                labels = [-100] * seq_length + input_ids[seq_length:]
                attention_mask = prompt_attention_mask + [1] + target_attention_mask + [1]
            else:
                seq_length = len(prompt_ids)
                input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
                labels = [-100] * seq_length + input_ids[seq_length:]
                attention_mask = prompt_attention_mask + target_attention_mask + [1]

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)

        processed_data["labels"] = labels_list
        processed_data["input_ids"] = input_ids_list
        processed_data["attention_mask"] = attention_mask_list

        return processed_data

    @staticmethod
    def _pad_to_max_length(examples, tokenizer, max_length):
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        labels_list = examples["labels"]
        input_ids_list = examples["input_ids"]
        attention_mask_list = examples["attention_mask"]

        for input_ids, attention_mask, labels in zip(input_ids_list, attention_mask_list, labels_list):
            l = len(input_ids)
            input_ids = torch.LongTensor(input_ids + [tokenizer.pad_token_id] * (max_length - l))
            labels = torch.LongTensor(labels + [-100] * (max_length - l))
            attention_mask = torch.LongTensor(attention_mask + [0] * (max_length - l))
            padded_input_ids.append(input_ids)
            padded_labels.append(labels)
            padded_attention_mask.append(attention_mask)

        return dict(
            input_ids=padded_input_ids,
            attention_mask=padded_attention_mask,
            labels=padded_labels
        )

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return self.tokenizer.__repr__()
