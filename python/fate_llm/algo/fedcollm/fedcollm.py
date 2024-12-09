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
import torch
import logging
from fate_llm.algo.fedcollm.fedcollm_trainer import FedCoLLMTrainer
from typing import Dict, Optional, List, Callable, Union
from fate.arch import Context
from fate.ml.nn.trainer.trainer_base import FedArguments
from torch.utils.data import Dataset
from transformers.trainer_callback import TrainerCallback
from transformers import PreTrainedTokenizer
from transformers import Seq2SeqTrainer
from transformers.trainer_utils import EvalPrediction
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_utils import unwrap_model
from fate_llm.algo.fedmkt.utils.generate_logit_utils import generate_pub_data_logits
from fate.ml.aggregator import AggregatorClientWrapper, AggregatorServerWrapper
from fate_llm.algo.fedcollm.fedcollm_training_args import FedCoLLMTrainingArguments
from types import SimpleNamespace


logger = logging.getLogger(__name__)


class FedCoLLMBase(object):
    @staticmethod
    def update_model(model, updated_params):
        for updated_p, p in zip(updated_params, [p for p in model.parameters() if p.requires_grad]):
            p.data.copy_(t.Tensor(updated_p))


class SLM(FedCoLLMBase):
    def __init__(
        self,
        ctx: Context,
        model: torch.nn.Module,
        training_args: FedCoLLMTrainingArguments,
        fed_args: FedArguments = None,
        train_set=None,
        val_set: Dataset = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        data_collator: Callable = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = [],
        save_trainable_weights_only: bool = False,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super(SLM, self).__init__()
        self.ctx = ctx
        self.training_args = training_args
        self.fed_args = fed_args
        self.model = model
        self.tokenizer = tokenizer
        self.model_init = model_init
        self.callbacks = callbacks
        self.compute_metrics = compute_metrics
        self.save_trainable_weights_only = save_trainable_weights_only
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics

        self.data_collator = data_collator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_set = train_set

        self.val_set = val_set

        self.aggregator = self._init_aggregator(ctx, fed_args)

    def train(self):
        global_epochs = self.training_args.global_epochs

        for i, iter_ctx in self.ctx.on_iterations.ctxs_range(global_epochs):
            logger.info(f"begin {i}-th global kd process")
            training_args = self._get_slm_training_args()

            trainer = Seq2SeqTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                train_dataset=self.train_set,
                args=training_args,
                model_init=self.model_init if not i else None,
                compute_metrics=self.compute_metrics,
                callbacks=self.callbacks,
                optimizers=(self.optimizer, self.scheduler),
                preprocess_logits_for_metrics=self.preprocess_logits_for_metrics
            )

            logger.info(f"begin {i}-th private data training process")
            trainer.train()

            self.model = unwrap_model(trainer.model)
            self.aggregator.model_aggregation(iter_ctx, self.model)

    def _sync_slm_updated_params(self, iter_ctx):
        updated_params = iter_ctx.arbiter.get("slm_updated_params")
        self.update_model(self.model, updated_params)

    def _get_slm_training_args(self):
        return self.training_args.to_slm_seq_training_args()

    def _init_aggregator(self, ctx: Context, fed_args: FedArguments):
        aggregate_type = "weighted_mean"
        aggregator_name = "fedavg"
        aggregator = fed_args.aggregator
        return AggregatorClientWrapper(
            ctx, aggregate_type, aggregator_name, aggregator,
            sample_num=len(self.train_set), args=self.training_args
        )


class LLM(FedCoLLMBase):
    def __init__(
        self,
        ctx: Context,
        llm_model: torch.nn.Module,
        slm_model: torch.nn.Module,
        training_args: FedCoLLMTrainingArguments,
        fed_args: FedArguments = None,
        train_set=None,
        val_set: Dataset = None,
        llm_optimizer: torch.optim.Optimizer = None,
        slm_optimizer: torch.optim.Optimizer = None,
        llm_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        slm_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        data_collator: Callable = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        llm_model_init: Optional[Callable[[], PreTrainedModel]] = None,
        slm_model_init: Optional[Callable[[], PreTrainedModel]] = None,
        llm_compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        slm_compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        llm_callbacks: Optional[List[TrainerCallback]] = [],
        slm_callbacks: Optional[List[TrainerCallback]] = [],
        save_trainable_weights_only: bool = False,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super(LLM, self).__init__()
        self.ctx = ctx
        self.llm_model = llm_model
        self.slm_model = slm_model
        self.training_args = training_args
        self.fed_args = fed_args
        self.train_set = train_set
        self.val_set = val_set
        self.llm_optimizer = llm_optimizer
        self.slm_optimizer = slm_optimizer
        self.llm_lr_scheduler = llm_lr_scheduler
        self.slm_lr_scheduler = slm_lr_scheduler
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.llm_model_init = llm_model_init
        self.slm_model_init = slm_model_init
        self.llm_compute_metrics = llm_compute_metrics
        self.slm_compute_metrics = slm_compute_metrics
        self.llm_callbacks = llm_callbacks
        self.slm_callbacks = slm_callbacks
        self.save_trainable_weights_only = save_trainable_weights_only
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics

        self.aggregator = self._init_aggregator(ctx)

    def _init_aggregator(self, ctx: Context):
        return AggregatorServerWrapper(ctx)

    def _get_logits(self, model):
        if self.training_args.device.type == "cuda":
            model.cuda(self.training_args.device.type)

        fn_kwargs = {"model": model,
                     "training_args": self.training_args,
                     "data_collator": self.data_collator}

        return self.train_set.map(
            generate_pub_data_logits,
            batched=True,
            batch_size=self.training_args.per_device_train_batch_size,
            num_proc=None,
            load_from_cache_file=True,
            fn_kwargs=fn_kwargs
        )

    def on_epoch_begin(self, iter_ctx):
        self.aggregator.model_aggregation(iter_ctx)
        updated_slm_params = iter_ctx()
        self.update_model(self.slm_model, updated_slm_params)

    def _sync_slm_updated_params(self, iter_ctx):
        updated_params = [p for p in self.slm_model.parameters() if p.requires_grad]
        iter_ctx.guest.put("slm_updated_params", updated_params)
        if any(p.role == 'host' for p in self.ctx.parties):
            iter_ctx.hosts.put("slm_updated_params", updated_params)

    def _train_slm(self, iter_ctx, llm_pub_logits, epoch_idx):
        top_k_args = SimpleNamespace(
            top_k_logits_keep=self.training_args.top_k_logits_keep,
            top_k_strategy=self.training_args.top_k_strategy
        )

        self.train_set.set_return_with_idx()
        trainer = FedCoLLMTrainer(
            model=self.slm_model,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            train_dataset=self.train_set,
            args=self.training_args.to_slm_seq_training_args(),
            model_init=self.slm_model_init if not epoch_idx else None,
            compute_metrics=self.slm_compute_metrics,
            callbacks=self.slm_callbacks,
            optimizers=(self.slm_optimizer, self.slm_lr_scheduler),
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            top_k_args=top_k_args,
            distill_lambda=self.training_args.distill_lambda,
            distill_temperature=self.training_args.distill_temperature,
            max_length=max(len(d["input_ids"]) for d in self.train_set),
            vocab_size=self.training_args.vocab_size,
            dtype=next(self.slm_model.parameters()).dtype,
            other_logits=llm_pub_logits
        )

        trainer.train()
        self.slm_model = unwrap_model(trainer.model)
        self.train_set.reset_return_with_idx()

        self._sync_slm_updated_params(iter_ctx)

    def _train_llm(self, slm_pub_logits, epoch_idx):
        top_k_args = SimpleNamespace(
            top_k_logits_keep=self.training_args.top_k_logits_keep,
            top_k_strategy=self.training_args.top_k_strategy
        )

        self.train_set.set_return_with_idx()
        trainer = FedCoLLMTrainer(
            model=self.llm_model,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            train_dataset=self.train_set,
            args=self.training_args.to_llm_seq_training_args(),
            model_init=self.llm_model_init if not epoch_idx else None,
            compute_metrics=self.llm_compute_metrics,
            callbacks=self.llm_callbacks,
            optimizers=(self.llm_optimizer, self.llm_lr_scheduler),
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            top_k_args=top_k_args,
            distill_lambda=self.training_args.distill_lambda,
            distill_temperature=self.training_args.distill_temperature,
            max_length=max(len(d["input_ids"]) for d in self.train_set),
            vocab_size=self.training_args.vocab_size,
            dtype=next(self.slm_model.parameters()).dtype,
            other_logits=slm_pub_logits
        )

        trainer.train()
        self.llm_model = unwrap_model(trainer.model)
        self.train_set.reset_return_with_idx()

    def train(self):
        global_epochs = self.training_args.global_epochs

        for i, iter_ctx in self.ctx.on_iterations.ctxs_range(global_epochs):
            logger.info(f"begin {i}-th global kd process")

            self.on_epoch_begin(iter_ctx)
            logger.info(f"get pub data logits for llm of global epoch={i}")
            llm_pub_data_logits = self._get_logits(self.llm_model)

            logger.info(f"train slm of global epoch={i}")
            self._train_slm(iter_ctx, llm_pub_data_logits, i)

            logger.info(f"get pub data logits for trained slm of global epoch={i}")
            slm_pub_data_logits = self._get_logits(self.slm_model)

            logger.info(f"train llm of global epoch={i}")
            self._train_llm(slm_pub_data_logits, i)
