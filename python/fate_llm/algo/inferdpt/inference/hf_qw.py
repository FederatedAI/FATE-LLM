from fate_llm.algo.inferdpt.inference.inference_base import Inference
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

