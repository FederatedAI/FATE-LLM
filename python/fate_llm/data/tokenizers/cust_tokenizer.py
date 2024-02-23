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
from transformers import AutoTokenizer


def get_tokenizer(
    tokenizer_name_or_path,
    trust_remote_code=False,
    padding_side=None,
    pad_token=None,
    bos_token=None,
    eos_token=None,
    pad_token_id=None,
    bos_token_id=None,
    eos_token_id=None,
    add_eos_token=True,
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=trust_remote_code,
        add_eos_token=add_eos_token
    )
    if padding_side is not None:
        tokenizer.padding_side = padding_side
    if pad_token is not None:
        tokenizer.add_special_tokens({'pad_token': pad_token})
    if bos_token is not None:
        tokenizer.add_special_tokens({'bos_token': bos_token})
    if eos_token is not None:
        tokenizer.add_special_tokens({"eos_token": eos_token})
    if pad_token_id is not None:
        tokenizer.pad_token_id = pad_token_id
    if bos_token_id is not None:
        tokenizer.bos_token_id = bos_token_id
    if eos_token_id is not None:
        tokenizer.eos_token_id = eos_token_id

    if "llama" in tokenizer_name_or_path.lower() or "gpt2" in tokenizer_name_or_path.lower():
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
