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
from transformers import GPT2Config, AutoConfig
from transformers import GPT2ForSequenceClassification, AutoModelForCausalLM
from fate_llm.model_zoo.pellm.parameter_efficient_llm import PELLM


class GPT2(PELLM):
    config_class = GPT2Config
    model_loader = GPT2ForSequenceClassification

    def __init__(self,
                 config: dict = None,
                 pretrained_path: str = None,
                 peft_type: str = None,
                 peft_config: dict = None,
                 **kwargs) -> None:

        if pretrained_path is not None:
            self.check_config(pretrain_path=pretrained_path)
        if config is None and pretrained_path is None:
            config = GPT2Config().to_dict()
        super().__init__(
            config=config,
            pretrained_path=pretrained_path,
            peft_type=peft_type,
            peft_config=peft_config,
            **kwargs)

    def check_config(self, pretrain_path):
        config = AutoConfig.from_pretrained(pretrain_path)
        assert isinstance(
            config, GPT2Config), 'The config of pretrained model must be GPT2Config, but got {}'.format(
            type(config))


class GPT2CLM(GPT2):
    model_loader = AutoModelForCausalLM
