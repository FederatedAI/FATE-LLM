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