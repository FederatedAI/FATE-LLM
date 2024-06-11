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
    load_model_dict_from_path,
    dir_warning,
    loader_load_from_conf,
    run_dataset_func,
)
from fate.ml.nn.homo.fedavg import FedAVGArguments
from fate_llm.homo.fedavg import Seq2SeqFedAVGClient, Seq2SeqFedAVGServer
from typing import Dict
from fate.components.components.nn.loader import Loader
from fate_llm.trainer.seq2seq_trainer import Seq2SeqTrainingArguments
from typing import Union, Optional
from transformers.trainer_utils import get_last_checkpoint
from typing import Literal
import logging
from fate.arch.dataframe import DataFrame
from fate_llm.runner.homo_seq2seq_runner import Seq2SeqRunner, _check_instances
from fate_llm.homo.offsite_tuning import OffsiteTuningTrainerClient, OffsiteTuningTrainerServer


logger = logging.getLogger(__name__)


SUPPORTED_ALGO = ["fedavg"]


class OTRunner(Seq2SeqRunner):

    def __init__(
        self,
        model_conf: Optional[Dict] = None,
        dataset_conf: Optional[Dict] = None,
        optimizer_conf: Optional[Dict] = None,
        training_args_conf: Optional[Dict] = None,
        fed_args_conf: Optional[Dict] = None,
        data_collator_conf: Optional[Dict] = None,
        tokenizer_conf: Optional[Dict] = None,
        task_type: Literal["causal_lm", "other"] = "causal_lm",
        save_trainable_weights_only: bool = False,
        aggregate_model: bool = False,
        algo: str = 'ot'
    ) -> None:
        super(OTRunner, self).__init__(
            algo, model_conf, dataset_conf, optimizer_conf, training_args_conf, fed_args_conf,
            data_collator_conf, tokenizer_conf, task_type, local_mode=False
        )

        self.aggregate_model = aggregate_model
        self.save_trainable_weights_only = save_trainable_weights_only

    def setup(self, train_set=None, validate_set=None, output_dir=None, saved_model=None, stage="train"):

        if stage == "predict":
            self.local_mode = True
            
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
        fed_args = FedAVGArguments(**self.fed_args_conf)

        # prepare trainer
        if self.is_client():
            trainer = OffsiteTuningTrainerClient(
                ctx=ctx,
                model=model,
                optimizer=optimizer,
                training_args=training_args,
                fed_args=fed_args,
                data_collator=data_collator,
                tokenizer=tokenizer,
                train_set=train_set,
                val_set=validate_set,
                save_trainable_weights_only=self.save_trainable_weights_only,
                aggregate_model=self.aggregate_model
            )

        elif self.is_server():
            trainer = OffsiteTuningTrainerServer(
                ctx=ctx,
                model=model,
                aggregate_model=self.aggregate_model
            )

        _check_instances(
            trainer=trainer,
            model=model,
            optimizer=optimizer,
            train_args=training_args,
            fed_args=fed_args,
            data_collator=data_collator,
        )

        return trainer

    def server_setup(self, stage="train"):
        if stage == "predict":
            self.local_mode = True
        if self.algo == "fedavg":
            server_class: Seq2SeqFedAVGServer = Seq2SeqFedAVGServer
        else:
            raise ValueError(f"algo {self.algo} not supported")
        ctx = self.get_context()
        trainer = server_class(ctx=ctx, local_mode=self.local_mode)
        _check_instances(trainer)
        return trainer
    

    def train(
        self,
        train_data: Optional[Union[str, DataFrame]] = None,
        validate_data: Optional[Union[str, DataFrame]] = None,
        output_dir: str = None,
        saved_model_path: str = None,
    ):
        
        if self.is_client():
            train_set = self._prepare_data(train_data, "train_data")
            validate_set = self._prepare_data(validate_data, "val_data")
            trainer = self.setup(
                train_set=train_set, validate_set=validate_set, output_dir=output_dir, saved_model=saved_model_path
            )
            self.trainer = trainer
            trainer.train()

        elif self.is_server():
            trainer = self.setup(
                train_set=None, validate_set=None, output_dir=output_dir, saved_model=saved_model_path
            )
            trainer.train()

        if output_dir is not None:
            if self.training_args.deepspeed and self.training_args.local_rank != 0:
                pass
            else:
                trainer.save_model(output_dir)
