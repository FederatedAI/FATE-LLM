from fate_llm.algo.inferdpt.init._init import InferDPTInit
from fate_llm.algo.inferdpt.inference.api import APICompletionInference
from fate_llm.algo.inferdpt import inferdpt
from fate_llm.algo.inferdpt.utils import InferDPTKit


class InferDPTAPIClientInit(InferDPTInit):

    api_url = ''
    api_model_name = ''
    api_key = 'EMPTY'
    inferdpt_kit_path = ''
    eps = 3.0

    def __init__(self, ctx):
        super().__init__(ctx)
        self.ctx = ctx

    def get_inferdpt_inst(self):
        inference = APICompletionInference(api_url=self.api_url, model_name=self.api_model_name, api_key=self.api_key)
        kit = InferDPTKit.load_from_path(self.inferdpt_kit_path)
        inferdpt_client = inferdpt.InferDPTClient(self.ctx, kit, inference, epsilon=self.eps)
        return inferdpt_client


class InferDPTAPIServerInit(InferDPTInit):

    api_url = ''
    api_model_name = ''
    api_key = 'EMPTY'

    def __init__(self, ctx):
        super().__init__(ctx)
        self.ctx = ctx

    def get_inferdpt_inst(self):
        inference = APICompletionInference(api_url=self.api_url, model_name=self.api_model_name, api_key=self.api_key)
        inferdpt_server = inferdpt.InferDPTServer(self.ctx,inference_inst=inference)
        return inferdpt_server
