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
import pandas as pd
from fate.ml.nn.dataset.base import Dataset
from .tokenizers import get_prompt_tokenizer


PROMPT_TEMPLATE = "{prompt}"


class PromptDataset(Dataset):
    def __init__(self,
                 text_max_length=256,
                 tokenizer_name_or_path=None,
                 trust_remote_code=False,
                 padding=False,
                 padding_side='left',
                 pad_token=None,
                 pad_token_id=0,
                 bos_token_id=1,
                 eos_token_id=2,
                 add_eos_token=True,
                 prompt_template=None,
                 add_special_tokens=False,
                 prompt_column="content",
                 response_column="summary",
                 ):

        super(PromptDataset, self).__init__()
        self.tokenizer = None
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.padding = padding
        self.add_special_tokens = add_special_tokens
        self.max_length = text_max_length

        self.tokenizer = get_prompt_tokenizer(
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

        if "chatglm" in self.tokenizer_name_or_path.lower():
            if len(prompt_ids) > self.max_length - 1:
                prompt_ids = prompt_ids[: self.max_length - 1]
            if len(target_ids) > self.max_length - 2:
                target_ids = target_ids[: self.max_length - 2]

            input_ids = self.tokenizer.build_inputs_with_special_tokens(
                prompt_ids, target_ids)

            if "chatglm2" in self.tokenizer_name_or_path.lower():
                seq_length = input_ids.index(self.tokenizer.bos_token_id)
            else:
                seq_length = len(prompt_ids)
        else:
            if len(prompt_ids) > self.max_length - 2:
                prompt_ids = prompt_ids[: self.max_length - 2]
            if len(target_ids) > self.max_length - 1:
                target_ids = target_ids[: self.max_length - 1]

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
        return copy.deepcopy(self._data[item])

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return self.tokenizer.__repr__()
