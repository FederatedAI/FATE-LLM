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

from fate_llm.inference.inference_base import Inference
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import tqdm


class QwenHFCompletionInference(Inference):

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def inference(self, docs: List[str], inference_kwargs: dict = {}) -> List[str]:
        self.model = self.model.eval()
        rs_list = []
        for d in tqdm.tqdm(docs):
            inputs = self.tokenizer(d, return_tensors='pt')
            inputs = inputs.to(self.model.device)
            inputs.update(inference_kwargs)
            pred = self.model.generate(**inputs)
            response = self.tokenizer.decode(pred.cpu()[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            rs_list.append(response)
        self.model = self.model.train()
        return rs_list

