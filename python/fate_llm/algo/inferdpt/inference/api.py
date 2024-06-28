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

from fate_llm.algo.inferdpt.inference.inference_base import Inference
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from typing import List


class APICompletionInference(Inference):

    def __init__(self, api_url: str, model_name: str, api_key: str = 'EMPTY', api_timeout=3600):
        from openai import OpenAI
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_url,
            timeout=api_timeout
        )

    def inference(self, docs: List[str], inference_kwargs: dict = {}) -> List[str]:
        completion = self.client.completions.create(model=self.model_name, prompt=docs, **inference_kwargs)
        rs_doc = [completion.choices[i].text for i in range(len(completion.choices))]
        return rs_doc