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
from torch.nn import Module
from transformers import GPT2ForTokenClassification, GPT2ForSequenceClassification
from fate_llm.model_zoo.ipr.sign_block import recursive_replace_layernorm


class SignGPT2ForTokenClassification(Module):

    def __init__(self, model_path=None, num_labels=4) -> None:
        super().__init__()
        if model_path is None:
            model_path = 'gpt2'

        self.model_path = model_path
        self.model = GPT2ForTokenClassification.from_pretrained(
            model_path, num_labels=num_labels)

        # replace layernorm by SignatureLayerNorm
        sub_gpt2 = self.model.transformer.h[10:]
        recursive_replace_layernorm(sub_gpt2)

    def forward(self, input_dict):
        return self.model(**input_dict)


class SignGPT2ForSequenceClassification(Module):

    def __init__(self, model_path=None, num_labels=2) -> None:
        super().__init__()
        if model_path is None:
            model_path = 'gpt2'

        self.model_path = model_path
        self.model = GPT2ForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels)

        # replace layernorm by SignatureLayerNorm
        sub_gpt2 = self.model.transformer.h[10:]
        recursive_replace_layernorm(sub_gpt2)

    def forward(self, input_dict):
        return self.model(**input_dict)
