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

from fate_llm.algo.inferdpt.init._init import InferInit
from fate_llm.inference.api import APICompletionInference
from fate_llm.algo.fedcot.encoder_decoder.slm_encoder_decoder import SLMEncoderDecoderClient, SLMEncoderDecoderServer


class FedCoTEDAPIClientInit(InferInit):

    api_url = ''
    api_model_name = ''
    api_key = 'EMPTY'

    def __init__(self, ctx):
        super().__init__(ctx)
        self.ctx = ctx

    def get_inst(self):
        inference = APICompletionInference(api_url=self.api_url, model_name=self.api_model_name, api_key=self.api_key)
        client = SLMEncoderDecoderClient(self.ctx, inference)
        return client


class FedCoTEDAPIServerInit(InferInit):

    api_url = ''
    api_model_name = ''
    api_key = 'EMPTY'

    def __init__(self, ctx):
        super().__init__(ctx)
        self.ctx = ctx

    def get_inst(self):
        inference = APICompletionInference(api_url=self.api_url, model_name=self.api_model_name, api_key=self.api_key)
        return SLMEncoderDecoderServer(self.ctx, inference)
