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
import datasets
from dataclasses import dataclass, field

import transformers

from ...trainer.seq2seq_trainer import Seq2SeqTrainingArguments
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
from fate_llm.algo.fedmkt.token_alignment.token_align import token_align
from fate_llm.algo.fedmkt.utils.generate_logit_utils import generate_pub_data_logits
from fate.ml.aggregator import AggregatorClientWrapper, AggregatorServerWrapper
from fate_llm.algo.fedmkt.fedmkt_trainer import FedMKTTrainer
from fate_llm.algo.fedmkt.fedmkt_data_collator import DataCollatorForFedMKT
from fate_llm.algo.fedmkt.utils.dataset_sync_util import sync_dataset


logger = logging.getLogger(__name__)


@dataclass
class FedMKTTrainingArguments(Seq2SeqTrainingArguments):
    """
    selection metric type
    """
    metric_type: str = field(default="ce")

    """
    top-k logits select params
    """
    top_k_logits_keep: int = field(default=128)
    top_k_strategy: str = field(default="highest")

    """
    distillation params
    """
    distill_loss_type: str = field(default="ce")
    kd_alpha: float = field(default=0.9)
    distill_temperature: float = field(default=1.0)
    server_public_data_local_epoch: int = field(default=1)
    client_public_data_local_epoch: int = field(default=1)
    client_priv_data_local_epoch: int = field(default=1)
    distill_strategy: str = field(default="greater")
    global_epochs: int = field(default=1)

    """
    token-alignment params
    """
    skip_align: bool = field(default=False)
    token_align_strategy: str = field(default="dtw")
    vocab_mapping_paths: Union[str, List[str]] = field(default=None)
    vocab_size: int = field(default=None)

    """
    homo training params
    """
    post_fedavg: bool = field(default=False)

    """
    slm training only
    """
    llm_training: bool = field(default=True)

    def to_dict(self):
        from dataclasses import fields
        from enum import Enum
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_dict_without_extra_args(self):
        args_dict = self.to_dict()
        args_dict.pop("metric_type")
        args_dict.pop("top_k_logits_keep")
        args_dict.pop("top_k_strategy")

        args_dict.pop("distill_loss_type")
        args_dict.pop("kd_alpha")
        args_dict.pop("distill_temperature")
        args_dict.pop("distill_strategy")
        args_dict.pop("server_public_data_local_epoch")
        args_dict.pop("client_public_data_local_epoch")
        args_dict.pop("client_priv_data_local_epoch")
        args_dict.pop("global_epochs")

        args_dict.pop("skip_align", False)
        args_dict.pop("token_align_strategy")
        args_dict.pop("vocab_mapping_paths", None)
        args_dict.pop("vocab_size", None)

        args_dict.pop("post_fedavg")

        args_dict.pop("llm_training", True)

        return args_dict

    def to_dict_with_client_priv_training_args(self):
        args_dict = self.to_dict_without_extra_args()

        args_dict["num_train_epochs"] = self.client_priv_data_local_epoch

        return args_dict

    def to_dict_with_client_kd_args(self):
        args_dict = self.to_dict_without_extra_args()

        args_dict["num_train_epochs"] = self.client_public_data_local_epoch

        return args_dict

    def to_dict_with_server_kd_args(self):
        args_dict = self.to_dict_without_extra_args()
        args_dict["num_train_epochs"] = self.server_public_data_local_epoch

        return args_dict


class FedMKTBase(object):
    def __init__(self, *args, **kwargs):
        self.model = None
        self.save_trainable_weights_only = None

    def save_model(
        self,
        output_dir: Optional[str] = None,
        state_dict=None
    ):
        if not self.save_trainable_weights_only:
            torch.save(self.model.state_dict(), output_dir + '/pytorch_model.bin')
        else:
            model = unwrap_model(self.model)

            if hasattr(model, "save_trainable"):
                model.save_trainable(output_dir)
            else:
                state_dict = {
                    k: p.to("cpu") for k,
                                       p in model.named_parameters() if p.requires_grad
                }

                torch.save(state_dict, output_dir + '/pytorch_model.bin')


class FedMKTSLM(FedMKTBase):
    def __init__(
        self,
        ctx: Context,
        model: torch.nn.Module,
        training_args: FedMKTTrainingArguments,
        fed_args: FedArguments = None,
        priv_train_set=None,
        pub_train_set=None,
        val_set: Dataset = None,
        priv_optimizer: torch.optim.Optimizer = None,
        pub_optimizer: torch.optim.Optimizer = None,
        priv_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        pub_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        data_collator: Callable = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = [],
        save_trainable_weights_only: bool = False,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        llm_tokenizer=None,
        llm_to_slm_vocab_mapping=None,
    ):
        super(FedMKTSLM, self).__init__()
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

        self.priv_data_collator = data_collator
        self.priv_optimizer = priv_optimizer
        self.pub_optimizer = pub_optimizer
        self.priv_scheduler = priv_scheduler
        self.pub_scheduler = pub_scheduler
        self.priv_train_set = priv_train_set
        self.pub_train_set = pub_train_set

        self.llm_tokenizer = llm_tokenizer
        self.llm_to_slm_vocab_mapping = llm_to_slm_vocab_mapping

        self.val_set = val_set

        self.aggregator = self._init_aggregator(ctx, fed_args)

        if not isinstance(self.pub_train_set, datasets.Dataset):
            self.pub_train_set = datasets.Dataset.from_list(list(self.pub_train_set))

    def train(self):
        global_epochs = self.training_args.global_epochs

        llm_pub_logits = None
        for i, iter_ctx in self.ctx.on_iterations.ctxs_range(global_epochs):
            logger.info(f"begin {i}-th global kd process")
            priv_data_training_args = self._get_priv_data_training_args()

            priv_trainer = Seq2SeqTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                data_collator=self.priv_data_collator,
                train_dataset=self.priv_train_set,
                args=priv_data_training_args,
                model_init=self.model_init if not i else None,
                compute_metrics=self.compute_metrics,
                callbacks=self.callbacks,
                optimizers=(self.priv_optimizer, self.priv_scheduler),
                preprocess_logits_for_metrics=self.preprocess_logits_for_metrics
            )

            logger.info(f"begin {i}-th private data training process")
            priv_trainer.train()

            self.model = unwrap_model(priv_trainer.model)

            logger.info(f"begin {i}-th public logits generation process")

            if self.training_args.world_size <= 1 or self.training_args.local_rank == 0:
                slm_pub_logits = self.pub_train_set.map(
                    generate_pub_data_logits,
                    batched=True,
                    batch_size=self.training_args.per_device_train_batch_size,
                    num_proc=None,
                    load_from_cache_file=True,
                    fn_kwargs={"model": self.model,
                               "training_args": self.training_args,
                               "data_collator": transformers.DataCollatorForSeq2Seq(self.tokenizer)}
                )

                if self.training_args.world_size > 1:
                    logger.info("sync slm_pub_logits")
                    sync_dataset(
                        slm_pub_logits, self.training_args.local_rank, self.training_args.world_size, self.training_args.device
                    )

                if self.training_args.llm_training:
                    logger.debug(f"send {i}-th public logits to llm")
                    iter_ctx.arbiter.put("slm_pub_logits", slm_pub_logits.to_dict())

                if self.training_args.llm_training or not i:
                    llm_pub_logits = datasets.Dataset.from_dict(iter_ctx.arbiter.get("llm_pub_logits"))
                    if self.training_args.world_size > 1:
                        logger.info("sync llm_pub_logits")
                        sync_dataset(llm_pub_logits, self.training_args.local_rank,
                                     self.training_args.world_size, self.training_args.device)
            else:
                slm_pub_logits = sync_dataset(
                    None, self.training_args.local_rank, self.training_args.world_size, self.training_args.device
                )

                if self.training_args.llm_training or not i:
                    llm_pub_logits = sync_dataset(None, self.training_args.local_rank,
                                                  self.training_args.world_size, self.training_args.device)

            logger.info(f"begin {i}-th token alignment process")
            aligned_dataset = token_align(
                base_model_logits_datasets=slm_pub_logits,
                blending_model_logits_dataset=llm_pub_logits,
                base_tokenizer=self.tokenizer,
                blending_tokenizer=self.llm_tokenizer,
                blending_to_base_mapping=self.llm_to_slm_vocab_mapping,
                blending_model_index=0,
                skip_align=self.training_args.skip_align,
                align_strategy=self.training_args.token_align_strategy
            )

            logger.info(f"begin {i}-th public logits kd process")
            fedmkt_trainer = self._init_trainer_for_distill(aligned_dataset)
            fedmkt_trainer.train()
            self.model = unwrap_model(fedmkt_trainer.model)

            if self.training_args.post_fedavg and (i + 1) % self.fed_args.aggregate_freq == 0:
                self.aggregator.model_aggregation(iter_ctx, self.model)

    def _init_trainer_for_distill(self, train_set):
        public_data_training_args = self._get_pub_data_kd_training_args()
        fedmkt_trainer = FedMKTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=public_data_training_args,
            train_dataset=train_set,
            eval_dataset=self.val_set,
            data_collator=DataCollatorForFedMKT(
                self.tokenizer,
                padding="max_length",
                max_length=max(len(d["input_ids"]) for d in train_set),
                blending_num=1,
                vocab_size=self.training_args.vocab_size,
                dtype=next(self.model.parameters()).dtype,
                distill_temperature=self.training_args.distill_temperature
            ),
            blending_num=1,
            lm_loss_weight=self.training_args.kd_alpha,
            distill_loss_type=self.training_args.distill_loss_type,
            distill_strategy=self.training_args.distill_strategy
        )

        return fedmkt_trainer

    def _get_priv_data_training_args(self):
        pre_args = self.training_args.to_dict_with_client_priv_training_args()
        post_args = Seq2SeqTrainingArguments(**pre_args)

        return post_args

    def _get_pub_data_kd_training_args(self):
        pre_args = self.training_args.to_dict_with_client_kd_args()
        post_args = Seq2SeqTrainingArguments(**pre_args)

        return post_args

    def _init_aggregator(self, ctx: Context, fed_args: FedArguments):
        if not self.training_args.post_fedavg:
            return None

        aggregate_type = "weighted_mean"
        aggregator_name = "fedavg"
        aggregator = fed_args.aggregator
        return AggregatorClientWrapper(
            ctx, aggregate_type, aggregator_name, aggregator,
            sample_num=len(self.pub_train_set), args=self.training_args
        )


class FedMKTLLM(FedMKTBase):
    def __init__(
        self,
        ctx: Context,
        model: torch.nn.Module,
        training_args: FedMKTTrainingArguments,
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
        slm_tokenizers: List = None,
        slm_to_llm_vocab_mappings: List[Dict] = None,
    ):
        super(FedMKTLLM, self).__init__()
        self.ctx = ctx
        self.model = model
        self.training_args = training_args
        self.fed_args = fed_args
        self.train_set = train_set
        self.val_set = val_set
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.model_init = model_init
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.save_trainable_weights_only = save_trainable_weights_only
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.slm_tokenizers = slm_tokenizers
        self.slm_to_llm_vocab_mappings = slm_to_llm_vocab_mappings

        self.aggregator = self._init_aggregator(ctx)

        if not isinstance(self.train_set, datasets.Dataset):
            self.train_set = datasets.Dataset.from_list(list(self.train_set))

    def _init_aggregator(self, ctx: Context):
        if not self.training_args.post_fedavg:
            return None
        return AggregatorServerWrapper(ctx)

    def generate_pub_data_logits(self, first_epoch=False):
        fn_kwargs = {"model": self.model,
                     "training_args": self.training_args,
                     "data_collator": transformers.DataCollatorForSeq2Seq(self.tokenizer)}
        if first_epoch and self.training_args.device.type == "cuda":
            self.model.cuda(self.training_args.device)

        return self.train_set.map(
            generate_pub_data_logits,
            batched=True,
            batch_size=self.training_args.per_device_train_batch_size,
            num_proc=None,
            load_from_cache_file=True,
            fn_kwargs=fn_kwargs
        )

    def on_epoch_begin(self, iter_ctx, epoch_idx, previous_pub_dataset):
        logger.info(f"on {epoch_idx}-epoch begin")
        if not self.training_args.llm_training:
            return

        if previous_pub_dataset is None:
            if self.training_args.world_size <= 1 or self.training_args.local_rank == 0:
                llm_pub_logits = self.generate_pub_data_logits(first_epoch=True if not epoch_idx else False)
                if self.training_args.world_size > 1:
                    sync_dataset(llm_pub_logits, self.training_args.local_rank,
                                 self.training_args.world_size, self.training_args.device)
            else:
                llm_pub_logits = sync_dataset(None, self.training_args.local_rank,
                                              self.training_args.world_size, self.training_args.device)
        else:
            llm_pub_logits = previous_pub_dataset

        slm_pub_logits_list = list()
        if self.training_args.world_size <= 1 or self.training_args.local_rank == 0:
            slm_pub_logits_list.append(datasets.Dataset.from_dict(iter_ctx.guest.get('slm_pub_logits')))
            if any(p.role == 'host' for p in self.ctx.parties):
                slm_pub_logits_list.extend(
                    datasets.Dataset.from_dict(client_logits) for client_logits in iter_ctx.hosts.get("slm_pub_logits")
                )
            if self.training_args.world_size > 1:
                logger.info("sync dataset to other rank")
                for slm_pub_logits in slm_pub_logits_list:
                    sync_dataset(slm_pub_logits, self.training_args.local_rank,
                                 self.training_args.world_size, self.training_args.device)
                    logger.info("end to sync")
        else:
            logger.info("sync dataset from rank 0")
            for _ in range(len(self.slm_tokenizers)):
                slm_pub_logits_list.append(
                    sync_dataset(None, self.training_args.local_rank,
                                 self.training_args.world_size, self.training_args.device)
                )
            logger.info("end to sync dataset from rank 0")

        aligned_dataset = llm_pub_logits
        for idx, slm_pub_logits in enumerate(slm_pub_logits_list):
            aligned_dataset = token_align(
                base_model_logits_datasets=aligned_dataset,
                blending_model_logits_dataset=slm_pub_logits,
                base_tokenizer=self.tokenizer,
                blending_tokenizer=self.slm_tokenizers[idx],
                blending_to_base_mapping=self.slm_to_llm_vocab_mappings[idx],
                blending_model_index=idx,
                skip_align=self.training_args.skip_align,
                align_strategy=self.training_args.token_align_strategy
            )

        return aligned_dataset

    def on_epoch_end(self, iter_ctx, epoch_idx):
        logger.info(f"on {epoch_idx}-epoch end")
        if not self.training_args.llm_training and epoch_idx > 1:
            return

        llm_pub_logits = self.generate_pub_data_logits(first_epoch=True if not self.training_args.llm_training else False)

        if self.training_args.world_size <= 1 or self.training_args.local_rank == 0:
            iter_ctx.guest.put("llm_pub_logits", llm_pub_logits.to_dict())
            if len(self.slm_tokenizers) > 1:
                iter_ctx.hosts.put("llm_pub_logits", llm_pub_logits.to_dict())

            if self.training_args.post_fedavg and (epoch_idx + 1) % self.fed_args.aggregate_freq == 0:
                self.aggregator.model_aggregation(iter_ctx)

            if self.training_args.world_size > 1:
                sync_dataset(
                    llm_pub_logits, self.training_args.local_rank, self.training_args.world_size, self.training_args.device
                )
        else:
            llm_pub_logits = sync_dataset(
                None, self.training_args.local_rank, self.training_args.world_size, self.training_args.device
            )

        return llm_pub_logits

    def _get_pub_data_kd_training_args(self):
        pre_args = self.training_args.to_dict_with_server_kd_args()
        post_args = Seq2SeqTrainingArguments(**pre_args)

        return post_args

    def train(self):
        global_epochs = self.training_args.global_epochs
        previous_pub_logits = None

        for i, iter_ctx in self.ctx.on_iterations.ctxs_range(global_epochs):
            logger.info(f"begin {i}-th global kd process")

            aligend_train_set = self.on_epoch_begin(iter_ctx, i, previous_pub_logits)
            if self.training_args.llm_training:

                public_data_training_args = self._get_pub_data_kd_training_args()
                fedmkt_trainer = FedMKTTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    args=public_data_training_args,
                    train_dataset=aligend_train_set,
                    eval_dataset=self.val_set,
                    data_collator=DataCollatorForFedMKT(
                        self.tokenizer,
                        padding="max_length",
                        max_length=max(len(d["input_ids"]) for d in aligend_train_set),
                        blending_num=len(self.slm_tokenizers),
                        vocab_size=self.training_args.vocab_size,
                        dtype=next(self.model.parameters()).dtype,
                        distill_temperature=self.training_args.distill_temperature
                    ),
                    blending_num=len(self.slm_tokenizers),
                    lm_loss_weight=self.training_args.kd_alpha,
                    distill_loss_type=self.training_args.distill_loss_type,
                    distill_strategy=self.training_args.distill_strategy
                )

                fedmkt_trainer.train()
                self.model = unwrap_model(fedmkt_trainer.model)

            previous_pub_logits = self.on_epoch_end(iter_ctx, i)
