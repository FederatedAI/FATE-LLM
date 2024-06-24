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
import json
from fate.components.components.nn.nn_runner import (
    load_model_dict_from_path,
    dir_warning,
    loader_load_from_conf,
    run_dataset_func,
)
from typing import Dict
from fate.components.components.nn.loader import Loader
from fate.ml.nn.homo.fedavg import FedAVGArguments
from typing import Union, Optional, Literal, List
from transformers.trainer_utils import get_last_checkpoint
import logging
from fate.arch.dataframe import DataFrame
from fate.components.components.nn.runner.homo_default_runner import DefaultRunner
from fate_llm.algo.fedmkt import FedMKTTrainingArguments, FedMKTSLM, FedMKTLLM

logger = logging.getLogger(__name__)


class FedMKTRunner(DefaultRunner):

    def __init__(
        self,
        algo: str = "fedmkt",
        model_conf: Optional[Dict] = None,
        optimizer_conf: Optional[Dict] = None,
        training_args_conf: Optional[Dict] = None,
        fed_args_conf: Optional[Dict] = None,
        pub_dataset_conf: Optional[Dict] = None,
        priv_dataset_conf: Optional[Dict] = None,
        data_collator_conf: Optional[Dict] = None,
        tokenizer_conf: Optional[Dict] = None,
        llm_tokenizer_conf: Optional[Dict] = None,
        slm_tokenizers_conf: List[Optional[Dict]] = None,
        llm_to_slm_vocab_mapping_path: str = None,
        slm_to_llm_vocab_mapping_paths: List[str] = None,
        task_type: Literal["causal_lm", "others"] = "causal_lm",
        save_trainable_weights_only: bool = False,
        pub_dataset_path: str = None,
    ) -> None:
        super(FedMKTRunner, self).__init__()
        self.algo = algo
        self.model_conf = model_conf
        self.optimizer_conf = optimizer_conf
        self.training_args_conf = training_args_conf
        self.fed_args_conf = fed_args_conf
        self.pub_dataset_conf = pub_dataset_conf
        self.priv_dataset_conf = priv_dataset_conf
        self.data_collator_conf = data_collator_conf
        self.tokenizer_conf = tokenizer_conf
        self.llm_tokenizer_conf = llm_tokenizer_conf
        self.slm_tokenizers_conf = slm_tokenizers_conf
        self.llm_to_slm_vocab_mapping_path = llm_to_slm_vocab_mapping_path
        self.slm_to_llm_vocab_mapping_paths = slm_to_llm_vocab_mapping_paths
        self.task_type = task_type
        self.pub_dataset_path = pub_dataset_path

        self.save_trainable_weights_only = save_trainable_weights_only

        self.training_args = None

        # check param
        if self.algo.lower() != "fedmkt":
            raise ValueError(f"algo should be fedmkt")
        if self.task_type not in ["causal_lm"]:
            raise ValueError("task_type should be causal_lm")

    def common_setup(self, saved_model=None, output_dir=None):
        ctx = self.get_context()

        if output_dir is None:
            output_dir = "./"

        model = loader_load_from_conf(self.model_conf)
        if model is None:
            raise ValueError(f"model is None, cannot load model from conf {self.model_conf}")

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

        # load tokenizer if import conf provided
        tokenizer = loader_load_from_conf(self.tokenizer_conf)

        # args
        dir_warning(self.training_args_conf)
        training_args = FedMKTTrainingArguments(**self.training_args_conf)
        # reset to default, saving to arbitrary path is not allowed in
        # DefaultRunner
        training_args.output_dir = output_dir
        training_args.resume_from_checkpoint = resume_path  # resume path

        self.training_args = training_args

        fed_args = FedAVGArguments(**self.fed_args_conf)

        pub_dataset = loader_load_from_conf(self.pub_dataset_conf)
        pub_dataset.load(self.pub_dataset_path)

        return ctx, model, optimizer, tokenizer, training_args, fed_args, pub_dataset

    def llm_setup(self, train_set=None, validate_set=None, output_dir=None, saved_model=None):
        ctx, model, optimizer, tokenizer, training_args, fed_args, pub_dataset = self.common_setup(
            output_dir=output_dir, saved_model=saved_model)

        if validate_set is not None:
            validate_dataset = loader_load_from_conf(self.pub_dataset_conf)
            validate_dataset.load(validate_set)
        else:
            validate_dataset = None

        slm_tokenizers = None
        if self.slm_tokenizers_conf:
            slm_tokenizers = [loader_load_from_conf(tokenizer_conf) for tokenizer_conf in self.slm_tokenizers_conf]

        slm_to_llm_vocab_mappings = []
        for vocab_mapping_path in self.slm_to_llm_vocab_mapping_paths:
            with open(vocab_mapping_path, "r") as fin:
                vocab_mapping = json.loads(fin.read())
                slm_to_llm_vocab_mappings.append(vocab_mapping)

        trainer = FedMKTLLM(
            ctx=ctx,
            model=model,
            training_args=training_args,
            fed_args=fed_args,
            train_set=pub_dataset,
            val_set=validate_dataset,
            tokenizer=tokenizer,
            slm_tokenizers=slm_tokenizers,
            slm_to_llm_vocab_mappings=slm_to_llm_vocab_mappings,
            save_trainable_weights_only=self.save_trainable_weights_only,
        )

        return trainer

    def slm_setup(self, train_set=None, validate_set=None, output_dir=None, saved_model=None):
        ctx, model, optimizer, tokenizer, training_args, fed_args, pub_dataset = self.common_setup(
            output_dir=output_dir, saved_model=saved_model)

        priv_dataset = loader_load_from_conf(self.priv_dataset_conf)
        priv_dataset.load(train_set)

        if validate_set is not None:
            validate_dataset = loader_load_from_conf(self.priv_dataset_conf)
            validate_dataset.load(validate_set)
        else:
            validate_dataset = None

        llm_tokenizer = loader_load_from_conf(self.llm_tokenizer_conf)

        with open(self.llm_to_slm_vocab_mapping_path, "r") as fin:
            vocab_mapping = json.loads(fin.read())

        priv_data_collator = loader_load_from_conf(self.data_collator_conf)

        trainer = FedMKTSLM(
            ctx=ctx,
            model=model,
            training_args=training_args,
            fed_args=fed_args,
            pub_train_set=pub_dataset,
            priv_train_set=priv_dataset,
            val_set=validate_dataset,
            tokenizer=tokenizer,
            save_trainable_weights_only=self.save_trainable_weights_only,
            llm_tokenizer=llm_tokenizer,
            llm_to_slm_vocab_mapping=vocab_mapping,
            data_collator=priv_data_collator
        )

        return trainer

    def train(
        self,
        train_data: Optional[Union[str, DataFrame]] = None,
        validate_data: Optional[Union[str, DataFrame]] = None,
        output_dir: str = None,
        saved_model_path: str = None,
    ):

        if self.is_client():
            trainer = self.slm_setup(train_set=train_data, validate_set=validate_data, output_dir=output_dir, saved_model=saved_model_path)
            trainer.train()
        else:
            trainer = self.llm_setup(
                train_set=train_data, validate_set=validate_data, output_dir=output_dir, saved_model=saved_model_path
            )
            trainer.train()

        self.trainer = trainer

        if self.training_args.deepspeed and self.training_args.local_rank != 0:
            pass
        else:
            trainer.save_model(output_dir)

    def predict(self, *args, **kwargs):
        pass
