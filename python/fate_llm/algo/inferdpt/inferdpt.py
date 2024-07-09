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
from fate_llm.algo.inferdpt._encode_decode import EncoderDecoder
from fate_llm.dataset.hf_dataset import HuggingfaceDataset


logger = logging.getLogger(__name__)


class InferDPTClient(EncoderDecoder):

    def __init__(self, ctx: Context, inferdpt_pertub_kit: InferDPTKit, local_inference_inst: Inference,  epsilon: float = 3.0,) -> None:
        self.ctx = ctx
        self.kit = inferdpt_pertub_kit
        assert epsilon > 0, 'epsilon must be a float > 0'
        self.ep = epsilon
        self.comm_idx = 0
        self.local_inference_inst = local_inference_inst

    def encode(self, docs: List[Dict[str, str]], format_template: str = None, verbose=False, perturb_doc_key: str ='perturbed_doc') -> List[Dict[str, str]]:
        
        copy_docs = copy.deepcopy(docs)
        if format_template is not None:
            template = Template(format_template)
        else:
            template = None

        for doc in tqdm(copy_docs):
            if template is None:
                rendered_doc = str(doc)
            else:
                rendered_doc = template.render(**doc)
                if verbose:
                    logger.debug('doc to perturb {}'.format(rendered_doc))
            p_doc = self.kit.perturb(rendered_doc, self.ep)
            doc[perturb_doc_key] = p_doc

        return copy_docs
        
    def _remote_inference(self, docs: List[Dict[str, str]], 
                     inference_kwargs: dict = {},
                     format_template: str = None, 
                     perturbed_response_key: str = 'perturbed_response',
                     verbose=False
                     ) -> List[Dict[str, str]]:

        copy_docs = copy.deepcopy(docs)
        if format_template is not None:
            template = Template(format_template)
        else:
            template = None

        infer_docs = []
        for doc in tqdm(copy_docs):
            if template is None:
                rendered_doc = str(doc)
            else:
                rendered_doc = template.render(**doc)
                if verbose:
                    logger.debug('inference doc {}'.format(rendered_doc))

            infer_docs.append(rendered_doc)
            doc['perturbed_doc_with_instrcution'] = rendered_doc
            
        self.ctx.arbiter.put('client_data_{}'.format(self.comm_idx), (infer_docs, inference_kwargs))
        perturb_resp = self.ctx.arbiter.get('pdoc_{}'.format(self.comm_idx))
        self.comm_idx += 1
        for pr, doc in zip(perturb_resp, copy_docs):
             doc[perturbed_response_key] = pr

        return copy_docs

    def decode(self, p_docs: List[Dict[str, str]], instruction_template: str = None, decode_template: str = None, verbose=False, 
                     perturbed_response_key: str = 'perturbed_response', result_key: str = 'inferdpt_result',
                     remote_inference_kwargs: dict = {}, local_inference_kwargs: dict = {}):

        # inference using remote large models
        docs_with_infer_result = self._remote_inference(p_docs, format_template=instruction_template, verbose=verbose, inference_kwargs=remote_inference_kwargs, perturbed_response_key=perturbed_response_key)
        if decode_template is not None:
            dt = Template(decode_template)
            doc_to_decode = [dt.render(**i) for i in docs_with_infer_result]
        else:
            doc_to_decode = [str(i) for i in docs_with_infer_result]
        # local model decode
        final_result = self.local_inference_inst.inference(doc_to_decode, local_inference_kwargs)
        for final_r, d in zip(final_result, docs_with_infer_result):
            d[result_key] = final_r

        return docs_with_infer_result

    def inference(self, docs: Union[List[Dict[str, str]], HuggingfaceDataset],
                encode_template: str,
                instruction_template: str,
                decode_template: str,
                verbose: bool = False,
                remote_inference_kwargs: dict = {},
                local_inference_kwargs: dict = {},
                perturb_doc_key: str = 'perturbed_doc',
                perturbed_response_key: str = 'perturbed_response',
                result_key: str = 'inferdpt_result',
                ) -> List[Dict[str, str]]:
        
        assert (isinstance(docs, list) and isinstance(docs[0], dict)) or isinstance(docs, HuggingfaceDataset), 'Input doc must be a list of dict or HuggingfaceDataset'
        # perturb doc
        if isinstance(docs, HuggingfaceDataset):
            docs = [docs[i] for i in range(len(docs))]
        docs_with_p = self.encode(docs, format_template=encode_template, verbose=verbose, perturb_doc_key=perturb_doc_key)
        logger.info('encode done')
        # inference using perturbed doc
        final_result = self.decode(
            docs_with_p,
            instruction_template,
            decode_template,
            verbose,
            perturbed_response_key,
            result_key,
            remote_inference_kwargs,
            local_inference_kwargs,
        )
        logger.info('decode done')
        
        return final_result


class InferDPTServer(object):

    def __init__(self, ctx: Context, inference_inst: Inference) -> None:
        
        self.ctx = ctx
        self.inference_inst = inference_inst
        self.comm_idx = 0 

    def inference(self, verbose=False):

        client_data = self.ctx.guest.get('client_data_{}'.format(self.comm_idx))
        perturbed_docs, inference_kwargs = client_data

        if verbose:
            logger.info('got data {}'.format(client_data))

        logger.info('start inference')
        rs_doc = self.inference_inst.inference(perturbed_docs, inference_kwargs)
        self.ctx.guest.put('pdoc_{}'.format(self.comm_idx), rs_doc)
        self.comm_idx += 1

    def predict(self):
        self.inference()
