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
import pandas as pd
from fate.components.components.nn.nn_runner import (
    load_model_dict_from_path,
    dir_warning,
    loader_load_from_conf,
    run_dataset_func,
)
from typing import Dict
from fate.arch.dataframe import PandasReader
from fate.components.components.nn.loader import Loader
from typing import Union, Optional, Literal
from transformers.trainer_utils import get_last_checkpoint
import logging
from fate.arch.dataframe import DataFrame
from fate.components.components.nn.runner.homo_default_runner import DefaultRunner
from fate_llm.algo.fdkt import FDKTTrainingArguments, FDKTSLM, FDKTLLM

logger = logging.getLogger(__name__)


class FDKTRunner(DefaultRunner):
    def __init__(
        self,
        algo: str = "fdkt",
        model_conf: Optional[Dict] = None,
        embedding_model_conf: Optional[Dict] = None,
        optimizer_conf: Optional[Dict] = None,
        training_args_conf: Optional[Dict] = None,
        dataset_conf: Optional[Dict] = None,
        data_collator_conf: Optional[Dict] = None,
        tokenizer_conf: Optional[Dict] = None,
        task_type: Literal["causal_lm", "others"] = "causal_lm",
        save_dp_model: bool = False,
    ) -> None:
        super(FDKTRunner, self).__init__()
        self.algo = algo
        self.model_conf = model_conf
        self.embedding_model_conf = embedding_model_conf
        self.optimizer_conf = optimizer_conf
        self.training_args_conf = training_args_conf
        self.dataset_conf = dataset_conf
        self.data_collator_conf = data_collator_conf
        self.tokenizer_conf = tokenizer_conf
        self.task_type = task_type
        self.save_dp_model = save_dp_model

        self.training_args = None

        # check param
        if self.algo.lower() != "fdkt":
            raise ValueError(f"algo should be fdkt")
        if self.task_type not in ["causal_lm"]:
            raise ValueError("task_type should be causal_lm")

        self.aug_data = None

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

        # load tokenizer if import conf provided
        tokenizer = loader_load_from_conf(self.tokenizer_conf)

        # args
        dir_warning(self.training_args_conf)
        training_args = FDKTTrainingArguments(**self.training_args_conf)
        # reset to default, saving to arbitrary path is not allowed in
        # DefaultRunner
        training_args.output_dir = output_dir
        training_args.resume_from_checkpoint = resume_path  # resume path

        self.training_args = training_args
        dataset = loader_load_from_conf(self.dataset_conf)

        return ctx, model, tokenizer, training_args, dataset

    def llm_setup(self, train_set=None, validate_set=None, output_dir=None, saved_model=None):
        ctx, model, tokenizer, training_args, dataset = self.common_setup(
            output_dir=output_dir, saved_model=saved_model)

        model = model.load()
        embedding_model = loader_load_from_conf(self.embedding_model_conf)
        if embedding_model is None:
            raise ValueError(f"model is None, cannot load model from conf {self.model_conf}")
        embedding_model = embedding_model.load()

        trainer = FDKTLLM(
            ctx=ctx,
            model=model,
            embedding_model=embedding_model,
            training_args=training_args,
            tokenizer=tokenizer,
            dataset=dataset,
        )

        return trainer

    def slm_setup(self, train_set=None, validate_set=None, output_dir=None, saved_model=None):
        ctx, model, tokenizer, training_args, dataset = self.common_setup(
            output_dir=output_dir, saved_model=saved_model)
        model = model.load()

        dataset.load(train_set)

        if self.data_collator_conf is not None:
            data_collator = loader_load_from_conf(self.data_collator_conf)
        else:
            data_collator = None

        optimizer_loader = Loader.from_dict(self.optimizer_conf)
        optimizer_ = optimizer_loader.load_item()
        optimizer_params = optimizer_loader.kwargs
        optimizer = optimizer_(model.parameters(), **optimizer_params)

        trainer = FDKTSLM(
            ctx=ctx,
            model=model,
            training_args=training_args,
            tokenizer=tokenizer,
            train_set=dataset,
            data_collator=data_collator,
            optimizer=optimizer,
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
            self.aug_data = trainer.aug_data()

            if self.save_dp_model:
                trainer.save_model(output_dir)

        else:
            trainer = self.llm_setup(
                train_set=train_data, validate_set=validate_data, output_dir=output_dir, saved_model=saved_model_path
            )
            trainer.aug_data()

    def predict(self, *args, **kwargs):
        if self.is_client():
            ctx = self.get_context()
            df = pd.DataFrame()
            texts = self.aug_data["inputs"]
            labels = self.aug_data["labels"]

            sample_id_name = "sample_id"
            match_id_name = "match_id"
            df[sample_id_name] = list(map(str, range(len(texts))))
            df[match_id_name] = list(map(str, range(len(texts))))
            import json
            texts = [json.dumps(text) for text in texts]
            labels = [json.dumps(label) for label in labels]
            df["inputs"] = texts
            df["labels"] = labels

            reader = PandasReader(sample_id_name=sample_id_name, match_id_name=match_id_name, dtype="object")
            return reader.to_frame(ctx, df)
