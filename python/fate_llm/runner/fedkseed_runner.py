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

import logging
from typing import Dict
from typing import Literal
from typing import Optional

from fate.components.components.nn.nn_runner import (
    NNRunner,
    load_model_dict_from_path,
    dir_warning,
    loader_load_from_conf,
)
from fate.components.components.nn.runner.homo_default_runner import DefaultRunner
from transformers.trainer_utils import get_last_checkpoint

from fate_llm.fedkseed.fedkseed import Trainer, FedKSeedTrainingArguments, ClientTrainer
from fate_llm.fedkseed.zo_utils import build_seed_candidates
from fate_llm.trainer.seq2seq_trainer import Seq2SeqTrainingArguments

logger = logging.getLogger(__name__)

SUPPORTED_ALGO = ["fedkseed"]


class FedKSeedRunner(DefaultRunner):
    def __init__(
            self,
            algo: str = "fedkseed",
            model_conf: Optional[Dict] = None,
            dataset_conf: Optional[Dict] = None,
            optimizer_conf: Optional[Dict] = None,
            training_args_conf: Optional[Dict] = None,
            fed_args_conf: Optional[Dict] = None,
            data_collator_conf: Optional[Dict] = None,
            tokenizer_conf: Optional[Dict] = None,
            task_type: Literal["causal_lm", "other"] = "causal_lm",
            local_mode: bool = False,
            save_trainable_weights_only: bool = False,
    ) -> None:
        super(NNRunner, self).__init__()
        self.algo = algo
        self.model_conf = model_conf
        self.dataset_conf = dataset_conf
        self.optimizer_conf = optimizer_conf
        self.training_args_conf = training_args_conf
        self.fed_args_conf = fed_args_conf
        self.data_collator_conf = data_collator_conf
        self.local_mode = local_mode
        self.tokenizer_conf = tokenizer_conf
        self.task_type = task_type
        self.save_trainable_weights_only = save_trainable_weights_only

        # check param
        if self.algo not in SUPPORTED_ALGO:
            raise ValueError(f"algo should be one of {SUPPORTED_ALGO}")
        if self.task_type not in ["causal_lm", "others"]:
            raise ValueError("task_type should be one of [binary, multi, regression, others]")
        assert isinstance(self.local_mode, bool), "local should be bool"

        # setup var
        self.trainer = None
        self.training_args = None

    def client_setup(self, train_set=None, validate_set=None, output_dir=None, saved_model=None, stage="train"):
        if self.algo != "fedkseed":
            raise ValueError(f"algo {self.algo} not supported")

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

        fedkseed_args = FedKSeedTrainingArguments(**self.fed_args_conf)
        training_args = Seq2SeqTrainingArguments(**self.training_args_conf)
        trainer = ClientTrainer(
            ctx=ctx,
            model=model,
            training_args=training_args,
            fedkseed_args=fedkseed_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
            train_dataset=train_set,
            eval_dataset=validate_set,
        )
        return trainer

    def server_setup(self, stage="train"):

        if self.algo != "fedkseed":
            raise ValueError(f"algo {self.algo} not supported")
        ctx = self.get_context()

        fedkseed_args = FedKSeedTrainingArguments(**self.fed_args_conf)
        training_args = Seq2SeqTrainingArguments(**self.training_args_conf)

        seed_candidates = build_seed_candidates(fedkseed_args.k, low=0, high=2 ** 32)
        trainer = Trainer(ctx=ctx, seed_candidates=seed_candidates, args=training_args)
        return trainer
