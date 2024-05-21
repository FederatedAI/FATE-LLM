import copy
from jinja2 import Template
from tqdm import tqdm
from fate.arch import Context
from typing import List, Dict, Any
from fate.ml.nn.dataset.base import Dataset
from fate_llm.algo.inferdpt.utils import InferDPTKit
from openai import OpenAI
import logging
from fate_llm.algo.inferdpt.inference.inference_base import Inference


logger = logging.getLogger(__name__)


class InferDPTClient(object):

    def __init__(self, ctx: Context, inferdpt_pertub_kit: InferDPTKit, local_inference_inst: Inference, epsilon=1.0) -> None:
        self.ctx = ctx
        self.kit = inferdpt_pertub_kit
        self.ep = epsilon
        self.comm_idx = 0
        self.local_inference_inst = local_inference_inst

    def perturb_doc(self, docs: List[Dict[str, Any]], format_template: str = None, verbose=False) -> List[Dict[str, Any]]:
        
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
                    logger.debug('doc to pertube {}'.format(rendered_doc))
            p_doc = self.kit.perturb(rendered_doc, self.ep)
            doc['perturbed_doc'] = p_doc

        return copy_docs
    
    def inference(self, docs: List[Dict[str, Any]], inference_kwargs: dict = {}, format_template: str = None, verbose=False) -> List[Dict[str, Any]]:

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
        for pr, doc in zip(perturb_resp, copy_docs):
             doc['perturbed_response'] = pr
        self.comm_idx += 1
        return copy_docs

    def predict(self, docs: List[Dict[str, Any]],
                doc_template: str,
                instruction_template: str,
                decode_template: str,
                verbose: bool = False,
                remote_inference_kwargs: dict = {},
                local_inference_kwargs: dict = {},
                ) -> List[Dict[str, Any]]:
        
        assert isinstance(docs, list) and isinstance(docs[0], dict), 'Input doc must be a list of dict'
        # perturb doc
        docs_with_p = self.perturb_doc(docs, format_template=doc_template, verbose=verbose)
        logger.info('perturb doc done')
        # inference using perturbed doc
        docs_with_infer_result = self.inference(docs_with_p, format_template=instruction_template, verbose=verbose, inference_kwargs=remote_inference_kwargs)
        logger.info('inferdpt remote inference done')
        # decode/generate final response using local model
        if decode_template is not None:
            dt = Template(decode_template)
            doc_to_decode = [dt.render(**i) for i in docs_with_infer_result]
        else:
            doc_to_decode = [str(i) for i in docs_with_infer_result]
        final_result = self.local_inference_inst.inference(doc_to_decode, local_inference_kwargs)
        logger.info('local decode/generate done')
        for final_r, d in zip(final_result, docs_with_infer_result):
            d['result'] = final_r
        return docs_with_infer_result


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
