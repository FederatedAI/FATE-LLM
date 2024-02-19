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
from fate_llm.model_zoo.pellm.parameter_efficient_llm import PELLM
from transformers import AutoConfig
from transformers import LlamaConfig
from transformers import LlamaForCausalLM


class LLaMa(PELLM):
    config_class = LlamaConfig
    enable_save_pretrained = True

    def __init__(self,
                 pretrained_path: str = None,
                 peft_type: str = None,
                 peft_config: dict = None) -> None:

        super().__init__(pretrained_path=pretrained_path,
                         peft_type=peft_type,
                         peft_config=peft_config)

    def init_base_lm(self):
        if self.config is not None:
            self._pe_lm = LlamaForCausalLM.from_pretrained(self.config_path,
                                                           config=self.config)
        elif self.config_path is not None:
            self._pe_lm = LlamaForCausalLM.from_pretrained(self.config_path)
        else:
            raise ValueError(
                'config_path to pretrained model folder cannot be None')

    def check_config(self, pretrain_path):
        config = AutoConfig.from_pretrained(pretrain_path)
        assert isinstance(
            config, LlamaConfig), 'The config of pretrained model must be LlamaConfig, but got {}'.format(
            type(config))
