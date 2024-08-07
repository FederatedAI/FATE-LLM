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
import torch
from fate.components.components.nn.nn_runner import (
    NNRunner,
    load_model_dict_from_path,
    dir_warning,
    loader_load_from_conf,
)
from fate_llm.model_zoo.hf_model import HFAutoModelForCausalLM
from fate.components.components.nn.loader import Loader
from fate.arch.dataframe import DataFrame
from fate.ml.nn.dataset.base import Dataset
from typing import Dict
from fate_llm.algo.pdss.pdss_trainer import PDSSTrainerClient, PDSSTraineServer
from fate_llm.algo.pdss.encoder_decoder.slm_encoder_decoder import SLMEncoderDecoderClient, SLMEncoderDecoderServer
from fate_llm.algo.inferdpt.init._init import InferInit
import torch.nn as nn
import torch.optim as optim
from fate_llm.trainer.seq2seq_trainer import Seq2SeqTrainingArguments
from typing import Union, Type, Callable, Optional
from transformers.trainer_utils import get_last_checkpoint
from typing import Literal
import logging


logger = logging.getLogger(__name__)



def _check_instances(
    model: nn.Module = None,
    optimizer: optim.Optimizer = None,
    train_args: Seq2SeqTrainingArguments = None,
    data_collator: Callable = None,
) -> None:
    
    if model is not None and not issubclass(type(model), nn.Module):
        raise TypeError(f"SetupReturn Error: model must be a subclass of torch.nn.Module but got {type(model)}")

    if optimizer is not None and not issubclass(type(optimizer), optim.Optimizer):
        raise TypeError(
            f"SetupReturn Error: optimizer must be a subclass of torch.optim.Optimizer but got {type(optimizer)}"
        )

    if train_args is not None and not isinstance(train_args, Seq2SeqTrainingArguments):
        raise TypeError(
            f"SetupReturn Error: train_args must be an instance of Seq2SeqTrainingArguments "
            f"but got {type(train_args)}"
        )

    if data_collator is not None and not callable(data_collator):
        raise TypeError(f"SetupReturn Error: data_collator must be callable but got {type(data_collator)}")


class PDSSRunner(NNRunner):
    def __init__(
        self,
        mode: Literal['train_only', 'infer_only', 'infer_and_train'],
        model_conf: Optional[Dict] = None,
        dataset_conf: Optional[Dict] = None,
        optimizer_conf: Optional[Dict] = None,
        training_args_conf: Optional[Dict] = None,
        data_collator_conf: Optional[Dict] = None,
        tokenizer_conf: Optional[Dict] = None,
        infer_inst_init_conf: Dict = None,
        encode_template: str = None,
        instruction_template: str = None,
        decode_template: str = None,
        remote_inference_kwargs: Dict = {},
        local_inference_kwargs: Dict = {},
        perturb_doc_key: str = 'perturbed_doc',
        perturbed_response_key: str = 'perturbed_response',
        result_key: str = 'infer_result',
    ) -> None:
        super(NNRunner, self).__init__()
        self.model_conf = model_conf
        self.dataset_conf = dataset_conf
        self.optimizer_conf = optimizer_conf
        self.training_args_conf = training_args_conf
        self.data_collator_conf = data_collator_conf
        self.mode = mode
        self.tokenizer_conf = tokenizer_conf
        self.infer_inst_init_conf = infer_inst_init_conf
        self.encode_template = encode_template
        self.instruction_template = instruction_template
        self.decode_template = decode_template
        self.remote_inference_kwargs = remote_inference_kwargs
        self.local_inference_kwargs = local_inference_kwargs
        self.perturb_doc_key = perturb_doc_key
        self.perturbed_response_key = perturbed_response_key
        self.result_key = result_key
        self._temp_data_path = ''

        # setup var
        self.trainer = None
        self.training_args = None

    def _get_infer_inst(self, init_conf):
        if init_conf is None:
            return None
        loader = Loader.from_dict(init_conf)
        init_inst = loader.load_item()(self.get_context())
        assert isinstance(init_inst, InferInit), 'Need a InferInit class for initialization, but got {}'.format(type(init_inst))
        infer_inst = init_inst.get_inst()
        logger.info('inferdpt inst loaded')
        return infer_inst
    
    def _prepare_data(self, data, data_name):
        if data is None:
            return None
        if isinstance(data, DataFrame) and self.dataset_conf is None:
            raise RuntimeError('DataFrame format dataset is not supported, please use bind path to load your dataset')
        else:
            dataset = loader_load_from_conf(self.dataset_conf)
            if hasattr(dataset, "load"):
                logger.info("load path is {}".format(data))
                import os
                if os.path.exists(data) and os.path.isdir(data):
                    self._temp_data_path = data
                    load_output = dataset.load(data)
                    if load_output is not None:
                        dataset = load_output
                        return dataset
                else:
                    raise RuntimeError('You must offer an existing folder path as data input, but got {}'.format(data))
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
    
    def client_setup(self, train_set=None, validate_set=None, output_dir=None, saved_model=None, stage="train"):

        ctx = self.get_context()
        model = loader_load_from_conf(self.model_conf)
        if isinstance(model, HFAutoModelForCausalLM):
            model = model.load()

        if model is None:
            raise ValueError(f"model is None, cannot load model from conf {self.model_conf}")
        if output_dir is None:
            output_dir = "./"

        resume_path = None
        if saved_model is not None:
            model_dict = load_model_dict_from_path(saved_model)
            model.load_state_dict(model_dict)
            logger.info(f"loading model dict from {saved_model} to model done")
            if get_last_checkpoint(saved_model) is not None:
                resume_path = saved_model
                logger.info(f"checkpoint detected, resume_path set to {resume_path}")

        # load optimizer
        if self.optimizer_conf:
            optimizer_loader = Loader.from_dict(self.optimizer_conf)
            optimizer_ = optimizer_loader.load_item()
            optimizer_params = optimizer_loader.kwargs
            optimizer = optimizer_(model.parameters(), **optimizer_params)
        else:
            optimizer = None

        # load collator func
        data_collator = loader_load_from_conf(self.data_collator_conf)

        # load tokenizer if import conf provided
        tokenizer = loader_load_from_conf(self.tokenizer_conf)

        # args
        dir_warning(self.training_args_conf)
        training_args = Seq2SeqTrainingArguments(**self.training_args_conf)
        # reset to default, saving to arbitrary path is not allowed in
        # DefaultRunner
        training_args.output_dir = output_dir
        training_args.resume_from_checkpoint = resume_path  # resume path
        self.training_args = training_args

        if self.training_args.world_size > 0 and self.training_args.local_rank == 0:
            infer_client = self._get_infer_inst(self.infer_inst_init_conf)
        else:
            infer_client = None # only rank 0 need to load the client
        
        # prepare trainer
        trainer = PDSSTrainerClient(
            ctx=ctx,
            training_args=training_args,
            train_set=train_set,
            val_set=validate_set,
            model=model,
            tokenizer=tokenizer,
            mode=self.mode,
            encode_template=self.encode_template,
            decode_template=self.decode_template,
            instruction_template=self.instruction_template,
            local_inference_kwargs=self.local_inference_kwargs,
            remote_inference_kwargs=self.remote_inference_kwargs,
            data_collator=data_collator,
            optimizer=optimizer,
            infer_client=infer_client,
            tmp_data_share_path=self._temp_data_path
        )

        return trainer

    def server_setup(self, stage="train"):
        trainer = PDSSTraineServer(
            ctx=self.get_context(),
            infer_server=self._get_infer_inst(self.infer_inst_init_conf)
        )
        return trainer

    def train(
        self,
        train_data: Optional[Union[str]] = None,
        validate_data: Optional[Union[str]] = None,
        output_dir: str = None,
        saved_model_path: str = None,
    ):
        if self.is_client():
            train_set = self._prepare_data(train_data, "train_data")
            validate_set = self._prepare_data(validate_data, "val_data")
            trainer = self.client_setup(
                train_set=train_set, validate_set=validate_set, output_dir=output_dir, saved_model=saved_model_path
            )
            self.trainer = trainer
            trainer.train()

            if self.mode == 'infer_only':
                # save result dataset to the output dir
                saving_path = output_dir + '/' + 'inference_result.pkl'
                torch.save(train_set.dataset, saving_path)
                logger.info('inference result saved to {}'.format(saving_path))
            else:
                if output_dir is not None:
                    if self.training_args.deepspeed and self.training_args.local_rank != 0:
                        pass
                    else:
                        trainer.save_model(output_dir)

        elif self.is_server():
            if self.mode == 'train_only':
                return 
            else:
                trainer = self.server_setup()
                trainer.train()

    def predict(self, test_data: Union[str], saved_model_path: str = None) -> None:
        logger.warning('The prediction mode is not supported by this algorithm in the current version. Please perform inference using locally saved models.')
        return 
