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
from transformers import GenerationConfig
import logging
from typing import List


logger = logging.getLogger(__name__)


class VLLMInference(Inference):

    def __init__(self, model_path, num_gpu=1, dtype='float16', gpu_memory_utilization=0.9):
        from vllm import LLM
        self.llm = LLM(model=model_path, trust_remote_code=True, dtype=dtype, tensor_parallel_size=num_gpu, gpu_memory_utilization=gpu_memory_utilization)
        logger.info('vllm model init done, model path is {}'.format(model_path))

    def inference(self, docs: List[str], inference_kwargs: dict = {}) -> List[str]:
        
        from vllm import SamplingParams
        param = SamplingParams(**inference_kwargs)
        outputs = self.llm.generate(
            prompts=docs, 
            sampling_params=param)

        rs = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            rs.append(generated_text)

        return rs


