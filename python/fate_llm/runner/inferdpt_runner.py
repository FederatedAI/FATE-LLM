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

from fate.components.components.nn.nn_runner import (
    NNRunner,
    load_model_dict_from_path,
    dir_warning,
    loader_load_from_conf,
    run_dataset_func,
)
import os
from datetime import datetime
from fate.components.components.nn.nn_runner import NNRunner
from typing import Dict
from fate.components.components.nn.loader import Loader
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Union, Type, Callable, Optional
from typing import Literal
import logging
from fate_llm.algo.inferdpt.inferdpt import InferDPTClient, InferDPTServer
from fate_llm.algo.inferdpt.init._init import InferInit
from fate.components.components.nn.loader import Loader
from fate_llm.dataset.hf_dataset import HuggingfaceDataset, Dataset
from fate.arch.dataframe import DataFrame



logger = logging.getLogger(__name__)


class InferDPTRunner(NNRunner):

    def __init__(
        self,
        inferdpt_init_conf: Dict,
        encode_template: str = None,
        instruction_template: str = None,
        decode_template: str = None,
        dataset_conf: Optional[Dict] = None,
        remote_inference_kwargs: Dict = {},
        local_inference_kwargs: Dict = {},
        perturb_doc_key: str = 'perturbed_doc',
        perturbed_response_key: str = 'perturbed_response',
        result_key: str = 'inferdpt_result',
    ) -> None:
        self.inferdpt_init_conf = inferdpt_init_conf
        self.encode_template = encode_template
        self.instruction_template = instruction_template
        self.decode_template = decode_template
        self.dataset_conf = dataset_conf
        self.remote_inference_kwargs = remote_inference_kwargs
        self.local_inference_kwargs = local_inference_kwargs
        self.perturb_doc_key = perturb_doc_key
        self.perturbed_response_key = perturbed_response_key
        self.result_key = result_key

    def _get_inst(self):
        loader = Loader.from_dict(self.inferdpt_init_conf)
        init_inst = loader.load_item()(self.get_context())
        assert isinstance(init_inst, InferInit), 'Need a InferDPTInit class for initialization, but got {}'.format(type(init_inst))
        inferdpt_inst = init_inst.get_inst()
        logger.info('inferdpt inst loaded')
        return inferdpt_inst
    
    def client_setup(self):
        client_inst = self._get_inst()
        assert isinstance(client_inst, InferDPTClient), 'Client need to get an InferDPTClient class to run the algo'
        return client_inst

    def server_setup(self):
        server_inst = self._get_inst()
        assert isinstance(server_inst, InferDPTServer), 'Server need to get an InferDPTServer class to run the algo'
        return server_inst

    def _prepare_data(self, data, data_name):
        if data is None:
            return None
        if isinstance(data, DataFrame) and self.dataset_conf is None:
            raise ValueError('DataFrame format dataset is not supported, please use bind path to load your dataset')
        else:
            dataset = loader_load_from_conf(self.dataset_conf)
            if hasattr(dataset, "load"):
                logger.info("load path is {}".format(data))
                load_output = dataset.load(data)
                if load_output is not None:
                    dataset = load_output
                    return dataset
            else:
                raise ValueError(
                    f"The dataset {dataset} lacks a load() method, which is required for data parsing in the DefaultRunner. \
                                Please implement this method in your dataset class. You can refer to the base class 'Dataset' in 'fate.ml.nn.dataset.base' \
                                for the necessary interfaces to implement."
                )
        if dataset is not None and not issubclass(type(dataset), Dataset):
            raise TypeError(
                f"SetupReturn Error: {data_name}_set must be a subclass of fate built-in Dataset but got {type(dataset)}, \n"
                f"You can get the class via: from fate.ml.nn.dataset.table import Dataset"
            )
        return dataset

    def train(
        self,
        train_data: Optional[Union[str]] = None,
        validate_data: Optional[Union[str]] = None,
        output_dir: str = None,
        saved_model_path: str = None,
    ) -> None:
        if self.is_client():
            dataset_0 = self._prepare_data(train_data, "train_data")
            logger.info('dataset loaded')
            if dataset_0 is None:
                raise ValueError('You must provide dataset for inference')
            assert isinstance(dataset_0, HuggingfaceDataset), 'Currently only support HuggingfaceDataset for inference, but got {}'.format(type(dataset_0))
            logger.info('initializing inst')
            client_inst = self.client_setup()
            pred_rs = client_inst.inference(
                dataset_0, self.encode_template, self.instruction_template, self.decode_template, \
                remote_inference_kwargs=self.remote_inference_kwargs,
                local_inference_kwargs=self.local_inference_kwargs
            )
            logger.info('predict done')
            saving_path = output_dir + '/' + 'inference_result.pkl'
            logger.info('result save to path {}'.format(saving_path))
            torch.save(pred_rs, saving_path)
        elif self.is_server():
            server_inst = self.server_setup()
            server_inst.inference()
        else:
            raise ValueError('Unknown role')

    def predict(
        self, test_data: Optional[Union[str]] = None, output_dir: str = None, saved_model_path: str = None
    ):
        logger.warning('Predicting mode is not supported in this algorithms in current version, please use the train mode to run inferdpt inference.')
        return 


