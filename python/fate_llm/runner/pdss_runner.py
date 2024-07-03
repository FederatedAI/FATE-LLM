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
from fate.components.components.nn.runner.homo_default_runner import DefaultRunner
from fate.components.components.nn.loader import Loader
from fate.ml.nn.trainer.trainer_base import HomoTrainerServer
from fate.arch.dataframe import DataFrame
from typing import Dict
from fate_llm.algo.pdss.pdss_trainer import PDSSTrainerClient, PDSSTraineServer
from fate_llm.algo.pdss.encoder_decoder.slm_encoder_decoder import SLMEncoderDecoderClient, SLMEncoderDecoderServer
from fate_llm.algo.inferdpt.inferdpt import InferDPTClient, InferDPTServer
import torch.nn as nn
import torch.optim as optim
from fate_llm.trainer.seq2seq_trainer import Seq2SeqTrainingArguments, HomoSeq2SeqTrainerClient
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


class Seq2SeqRunner(DefaultRunner):
    def __init__(
        self,
        model_conf: Optional[Dict] = None,
        dataset_conf: Optional[Dict] = None,
        optimizer_conf: Optional[Dict] = None,
        training_args_conf: Optional[Dict] = None,
        data_collator_conf: Optional[Dict] = None,
        tokenizer_conf: Optional[Dict] = None,
        mode: Literal['train_only', 'infer_only', 'infer_and_train'] = False,
        infer_client_init_conf: Dict = None,
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
        self.infer_client_init_conf = infer_client_init_conf
        self.encode_template = encode_template
        self.instruction_template = instruction_template
        self.decode_template = decode_template
        self.remote_inference_kwargs = remote_inference_kwargs
        self.local_inference_kwargs = local_inference_kwargs
        self.perturb_doc_key = perturb_doc_key
        self.perturbed_response_key = perturbed_response_key
        self.result_key = result_key

        assert isinstance(self.local_mode, bool), "local should be bool"
        # setup var
        self.trainer = None
        self.training_args = None


    def _get_inferdpt_inst(self, init_conf):
        loader = Loader.from_dict(init_conf)
        init_inst = loader.load_item()(self.get_context())
        assert isinstance(init_inst, InferDPTInit), 'Need a InferDPTInit class for initialization, but got {}'.format(type(init_inst))
        inferdpt_inst = init_inst.get_inferdpt_inst()
        logger.info('inferdpt inst loaded')
        return inferdpt_inst
    

    def client_setup(self, train_set=None, validate_set=None, output_dir=None, saved_model=None, stage="train"):

        ctx = self.get_context()
        model = loader_load_from_conf(self.model_conf)

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
        self.training_args = training_args
        # reset to default, saving to arbitrary path is not allowed in
        # DefaultRunner
        training_args.output_dir = output_dir
        training_args.resume_from_checkpoint = resume_path  # resume path

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
            remote_inference_kwargs=self.remote_inference_kwargs
        )

        return trainer

    def server_setup(self, stage="train"):
        trainer = None
        return trainer

    def predict(self, test_data: Union[str, DataFrame], saved_model_path: str = None) -> Union[DataFrame, None]:
        if self.is_client():
            test_set = self._prepare_data(test_data, "test_data")
            if self.trainer is not None:
                trainer = self.trainer
                logger.info("trainer found, skip setting up")
            else:
                trainer = self.client_setup(saved_model=saved_model_path, stage="predict")

            classes = run_dataset_func(test_set, "get_classes")
            match_ids = run_dataset_func(test_set, "get_match_ids")
            sample_ids = run_dataset_func(test_set, "get_sample_ids")
            match_id_name = run_dataset_func(test_set, "get_match_id_name")
            sample_id_name = run_dataset_func(test_set, "get_sample_id_name")

            if not self.training_args.predict_with_generate:
                return

            pred_rs = trainer.predict(test_set)

            if self.training_args and self.training_args.deepspeed and self.training_args.local_rank != 0:
                return

            rs_df = self.get_nn_output_dataframe(
                self.get_context(),
                pred_rs.predictions,
                pred_rs.label_ids if hasattr(pred_rs, "label_ids") else None,
                match_ids,
                sample_ids,
                match_id_name=match_id_name,
                sample_id_name=sample_id_name,
                dataframe_format="dist_df",
                task_type=self.task_type,
                classes=classes,
            )
            return rs_df
        else:
            # server not predict
            return
