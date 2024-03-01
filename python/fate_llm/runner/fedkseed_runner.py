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

import transformers
from fate.components.components.nn.nn_runner import (
    NNRunner,
    dir_warning,
    loader_load_from_conf,
)
from fate.components.components.nn.runner.homo_default_runner import DefaultRunner

from fate_llm.homo.fedkseed.fedkseed import Trainer, FedKSeedTrainingArguments, ClientTrainer
from fate_llm.homo.fedkseed.zo_utils import build_seed_candidates
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

        model = maybe_loader_load_from_conf(self.model_conf)
        if model is None:
            raise ValueError(f"model is None, cannot load model from conf {self.model_conf}")

        if output_dir is None:
            output_dir = "./"

        tokenizer = transformers.AutoTokenizer.from_pretrained(**self.data_collator_conf["kwargs"]["tokenizer_params"])

        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        dir_warning(self.training_args_conf)

        training_args = Seq2SeqTrainingArguments(**self.training_args_conf)
        self.training_args = training_args
        training_args.output_dir = output_dir
        fedkseed_args = FedKSeedTrainingArguments(**self.fed_args_conf)
        logger.debug(f"training_args: {training_args}")
        logger.debug(f"fedkseed_args: {fedkseed_args}")
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
        trainer = Trainer(ctx=ctx, seed_candidates=seed_candidates, args=training_args, fedkseed_args=fedkseed_args)
        return trainer


def maybe_loader_load_from_conf(conf):
    from fate_llm.model_zoo.hf_model import HFAutoModelForCausalLM

    model = loader_load_from_conf(conf)
    if isinstance(model, HFAutoModelForCausalLM):
        model = model.load()
    return model
