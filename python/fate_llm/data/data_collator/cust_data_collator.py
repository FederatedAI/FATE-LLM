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
from ..tokenizers.cust_tokenizer import get_tokenizer


def get_data_collator(data_collator_name,
                      tokenizer_name_or_path=None,
                      pad_token=None,
                      bos_token=None,
                      eos_token=None,
                      pad_token_id=None,
                      bos_token_id=None,
                      eos_token_id=None,
                      trust_remote_code=False, **kwargs):
    if not hasattr(data_collator, data_collator_name):
        support_collator_list = list(filter(lambda module_name: "collator" in module_name.lower(), dir(data_collator)))
        return ValueError(f"data_collator's name={data_collator_name} does not in support list={support_collator_list}")

    tokenizer = get_tokenizer(tokenizer_name_or_path=tokenizer_name_or_path,
                              pad_token=pad_token,
                              bos_token=bos_token,
                              eos_token=eos_token,
                              pad_token_id=pad_token_id,
                              bos_token_id=bos_token_id,
                              eos_token_id=eos_token_id,
                              trust_remote_code=trust_remote_code)

    return getattr(data_collator, data_collator_name)(tokenizer, **kwargs)


def get_seq2seq_data_collator(tokenizer_name_or_path, **kwargs):
    return get_data_collator("DataCollatorForSeq2Seq", tokenizer_name_or_path=tokenizer_name_or_path, **kwargs)
