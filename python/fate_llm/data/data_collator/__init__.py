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
from transformers.data import data_collator
from ..tokenizers import get_prompt_tokenizer


def get_data_collator(data_collator_name, tokenizer_name_or_path=None, pad_token=None, padding_side="left", **kwargs):
    if not hasattr(data_collator, data_collator_name):
        support_collator_list = list(filter(lambda module_name: "collator" in module_name.lower(), dir(data_collator)))
        return ValueError(f"data_collator's name={data_collator_name} does not in support list={support_collator_list}")

    tokenizer = get_prompt_tokenizer(tokenizer_name_or_path=tokenizer_name_or_path,
                                     pad_token=pad_token)

    return getattr(data_collator, data_collator_name)(tokenizer, **kwargs)
