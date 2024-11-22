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
import os

from fate_llm.algo.inferdpt.init._init import InferInit
from fate_llm.inference.api import APICompletionInference
from fate_llm.algo.inferdpt import inferdpt
from fate_llm.algo.inferdpt.utils import InferDPTKit
from fate_llm.algo.inferdpt.inferdpt import InferDPTClient, InferDPTServer


class InferDPTAPIClientInit(InferInit):

    api_url = ''
    api_model_name = ''
    api_key = os.environ.get("FATE_LLM_API_KEY", "")
    inferdpt_kit_path = ''
    eps = os.environ.get("FATE_LLM_INFERDPT_EPS", 3.0)

    def __init__(self, ctx, api_url=None, api_model_name=None, inferdpt_kit_path=None):
        super().__init__(ctx)
        self.ctx = ctx

        if api_url is not None:
            self.api_url = api_url
        if api_model_name is not None:
            self.api_model_name = api_model_name
        if inferdpt_kit_path is not None:
            self.inferdpt_kit_path = inferdpt_kit_path

    def get_inst(self)-> InferDPTClient:
        if self.api_key is None:
            raise ValueError("Please set the environment variable FATE_LLM_API_KEY for llm querying")
        if self.api_model_name is None or self.api_url is None:
            raise ValueError(f"api_model_name={self.api_model_name} and api_url={self.api_url} should not be None")
        inference = APICompletionInference(api_url=self.api_url, model_name=self.api_model_name, api_key=self.api_key)
        kit = InferDPTKit.load_from_path(self.inferdpt_kit_path)
        inferdpt_client = inferdpt.InferDPTClient(self.ctx, kit, inference, epsilon=self.eps)
        return inferdpt_client


class InferDPTAPIServerInit(InferInit):

    api_url = ''
    api_model_name = ''
    api_key = os.environ.get("FATE_LLM_API_KEY", "")

    def __init__(self, ctx, api_url, api_model_name):
        super().__init__(ctx)
        self.ctx = ctx
        self.api_url = api_url
        self.api_model_name = api_model_name

    def get_inst(self)-> InferDPTServer:
        if self.api_key is None:
            raise ValueError("Please set the environment variable FATE_LLM_API_KEY for llm querying")
        if self.api_model_name is None or self.api_url is None:
            raise ValueError(f"api_model_name={self.api_model_name} and api_url={self.api_url} should not be None")
        inference = APICompletionInference(api_url=self.api_url, model_name=self.api_model_name, api_key=self.api_key)
        inferdpt_server = inferdpt.InferDPTServer(self.ctx,inference_inst=inference)
        return inferdpt_server
