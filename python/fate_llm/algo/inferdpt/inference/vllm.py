from fate_llm.algo.inferdpt.inference.inference_base import Inference
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


