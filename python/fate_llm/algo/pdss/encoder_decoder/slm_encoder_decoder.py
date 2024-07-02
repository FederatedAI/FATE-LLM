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
import copy
from jinja2 import Template
from tqdm import tqdm
from fate.arch import Context
from typing import List, Dict, Union
from fate.ml.nn.dataset.base import Dataset
from fate_llm.algo.inferdpt.utils import InferDPTKit
from openai import OpenAI
import logging
from fate_llm.algo.inferdpt.inference.inference_base import Inference
from fate_llm.algo.inferdpt.inferdpt import InferDPTClient, InferDPTServer
from fate_llm.dataset.hf_dataset import HuggingfaceDataset


logger = logging.getLogger(__name__)


class SLMEncoderDecoderClient(InferDPTClient):

    def __init__(self, ctx: Context, local_inference_inst: Inference) -> None:
        self.ctx = ctx
        self.comm_idx = 0
        self.local_inference_inst = local_inference_inst
        self.local_inference_kwargs = {}

    def encode(self, docs: List[Dict[str, str]], format_template: str = None, verbose=False, perturb_doc_key: str ='perturbed_doc') -> List[Dict[str, str]]:
        
        template = Template(format_template)
        copy_docs = copy.deepcopy(docs)
        doc_to_infer = []
        for doc in tqdm(copy_docs):
            rendered_doc = template.render(**doc)
            doc_to_infer.append(rendered_doc)
        # perturb using local model inference
        self.doc_to_infer = doc_to_infer
        infer_result = self.local_inference_inst.inference(doc_to_infer, self.local_inference_kwargs)
        for doc, pr in zip(copy_docs, infer_result):
            doc[perturb_doc_key] = pr
        self.doc_with_p = copy_docs
        return copy_docs
    
    def decode(self, p_docs: List[Dict[str, str]], instruction_template: str = None, decode_template: str = None, verbose=False, 
            perturbed_response_key: str = 'perturbed_response', result_key: str = 'result',
            remote_inference_kwargs: dict = {}, local_inference_kwargs: dict = {}):
        return super().decode(p_docs, instruction_template, decode_template, verbose, perturbed_response_key, result_key, remote_inference_kwargs, local_inference_kwargs)

    def inference(self, docs: Union[List[Dict[str, str]], HuggingfaceDataset],
                encode_template: str,
                instruction_template: str,
                decode_template: str,
                verbose: bool = False,
                remote_inference_kwargs: dict = {},
                local_inference_kwargs: dict = {},
                perturb_doc_key: str = 'perturbed_doc',
                perturbed_response_key: str = 'perturbed_response',
                result_key: str = 'result',
                ) -> List[Dict[str, str]]:
        self.local_inference_kwargs = local_inference_kwargs
        return super().inference(docs, encode_template, instruction_template, decode_template, verbose, remote_inference_kwargs, \
            local_inference_kwargs, perturb_doc_key, perturbed_response_key, result_key)


class SLMEncoderDecoderServer(InferDPTServer):
    pass
