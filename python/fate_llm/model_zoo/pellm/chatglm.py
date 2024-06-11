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


class ChatGLM(PELLM):
    def __init__(self,
                 pretrained_path: str = None,
                 peft_type: str = None,
                 peft_config: dict = None,
                 pre_seq_len: int = None,
                 prefix_projection: bool = False,
                 **kwargs) -> None:

        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection

        super().__init__(pretrained_path=pretrained_path,
                         peft_type=peft_type,
                         peft_config=peft_config,
                         **kwargs
                         )

    def init_config(self):
        self.config = AutoConfig.from_pretrained(
            self.config_path, trust_remote_code=True)
        self.config.pre_seq_len = self.pre_seq_len
        self.config.prefix_projection = self.prefix_projection

    def add_peft(self):
        if self.pre_seq_len:
            self._pe_lm.half()
            self._pe_lm.transformer.prefix_encoder.float()
        else:
            super(ChatGLM, self).add_peft()
