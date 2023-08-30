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
import pandas as pd
from transformers import LlamaTokenizer
from federatedml.nn.dataset.base import Dataset


PROMPT_TEMPLATE = "{prompt}"


class LLAMATokenizerDataset(Dataset):
    def __init__(self, text_max_length=256,
                 tokenizer_name_or_path=None,
                 padding=False, padding_side='left',
                 pad_token_id=0,
                 bos_token_id=1,
                 eos_token_id=2,
                 add_eos_token=True,
                 prompt_template=None,
                 add_special_tokens=False,
                 prompt_column="content",
                 response_column="summary",
                 ):

        super(LLAMATokenizerDataset, self).__init__()
        self.tokenizer = None
        self.padding = padding
        self.add_special_tokens = add_special_tokens
        self.max_length = text_max_length
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.tokenizer_name_or_path, add_eos_token=add_eos_token)
        self.tokenizer.pad_token_id = pad_token_id
        self.tokenizer.bos_token_id = bos_token_id
        self.tokenizer.eos_token_id = eos_token_id
        self.tokenizer.padding_side = padding_side

        self.prompt_template = prompt_template if prompt_template else PROMPT_TEMPLATE
        self.prompt_column = prompt_column
        self.response_column = response_column
        self._data = None

    def load(self, file_path):
        df = pd.read_json(file_path, lines=True)
        self._data = df.apply(self._process_data, axis=1)

    def _process_data(self, line):
        _prompt = line[self.prompt_column]
        _response = line[self.response_column]

        prompt = self.prompt_template.format_map(dict(prompt=_prompt))
        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=self.add_special_tokens,
            padding=self.padding)
        target_ids = self.tokenizer.encode(
            _response,
            add_special_tokens=self.add_special_tokens,
            padding=self.padding)

        if len(prompt_ids) > self.max_length - 2:
            prompt_ids = prompt_ids[: self.max_length - 2]
        if len(target_ids) > self.max_length - 2:
            target_ids = target_ids[: self.max_length - 2]

        input_ids = self.tokenizer.build_inputs_with_special_tokens(
            prompt_ids, target_ids)

        seq_length = len(prompt_ids) + 2
        labels = [-100] * seq_length + input_ids[seq_length:]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return self.tokenizer.__repr__()
