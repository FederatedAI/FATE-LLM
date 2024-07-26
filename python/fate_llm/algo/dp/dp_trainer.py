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
import logging
import opacus
import os
import torch
from dataclasses import dataclass, field
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Callable
from .opacus_compatibility import add_layer_compatibility, add_optimizer_compatibility
from .opacus_compatibility.transformers_compate import prepare_position_ids

logger = logging.getLogger(__name__)


@dataclass
class DPTrainingArguments(Seq2SeqTrainingArguments):
    target_epsilon: float = field(default=3)
    target_delta: float = field(default=1e-5)
    freeze_embedding: bool = field(default=True)
    device_id: int = field(default=0)


class DPTrainer(object):
    def __init__(
        self,
        model: torch.nn.Module,
        training_args: DPTrainingArguments,
        train_set,
        loss_fn,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        data_collator: Callable = None,
        use_tqdm: bool = False,
    ):
        self.module = model
        self.training_args = training_args
        self.ori_optimizer = optimizer
        self.lr_scheduler = scheduler
        self.train_set = train_set
        self.data_collator = data_collator
        self.loss_fn = loss_fn
        self.use_tqdm = use_tqdm

        self.data_loader = DataLoader(
            dataset=self.train_set,
            shuffle=True,
            batch_size=self.training_args.per_device_train_batch_size,
            collate_fn=self.data_collator
        )

        if not self.training_args.use_cpu:
            self.module.cuda(self.training_args.device_id)

        if self.training_args.freeze_embedding:
            self.freeze_model_embedding()

        self.dp_model = None
        self.dp_optimizer = None
        self.privacy_engine = None
        self._init_dp_model()

    def _init_dp_model(self):
        self.module.train()

        # add compatibility for layer hooks
        add_layer_compatibility(opacus)

        self.privacy_engine = opacus.PrivacyEngine(accountant="rdp")
        self.dp_model, self.dp_optimizer, _ = self.privacy_engine.make_private_with_epsilon(
            module=self.module,
            optimizer=self.ori_optimizer,
            data_loader=self.data_loader,
            target_delta=self.training_args.target_delta,
            target_epsilon=self.training_args.target_epsilon,
            max_grad_norm=self.training_args.max_grad_norm,
            epochs=int(self.training_args.num_train_epochs),
        )

        add_optimizer_compatibility(self.dp_optimizer)

    def train(self):
        logger.info(f"begin dp training, total epochs={self.training_args.num_train_epochs}")
        for epoch in range(int(self.training_args.num_train_epochs)):
            logger.info(f"dp training on epoch={epoch}")
            self._train_an_epoch()

    def _train_an_epoch(self):
        if self.use_tqdm:
            data_loader = tqdm(self.data_loader)
        else:
            data_loader = self.data_loader

        for batch_idx, batch_data in enumerate(tqdm(data_loader)):
            input_ids = batch_data["input_ids"]
            labels = batch_data["labels"]

            if "attention_mask" not in batch_data:
                attention_mask = torch.ones(input_ids.shape)
            else:
                attention_mask = batch_data["attention_mask"]

            if not self.training_args.use_cpu:
                input_ids = input_ids.to(self.module.device)
                labels = labels.to(self.module.device)
                attention_mask = attention_mask.to(self.module.device)

            inputs = self._prepare_batch_input(input_ids)
            logits = self.dp_model(**inputs).logits

            loss = self.loss_fn(logits, labels, attention_mask)

            loss = loss.mean()
            loss.backward()

            if (batch_idx + 1) % self.training_args.gradient_accumulation_steps == 0 or \
                    batch_idx + 1 == len(self.data_loader):
                self.dp_optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.dp_optimizer.zero_grad()
            else:
                self.dp_optimizer.step()
                self.dp_optimizer.zero_grad()

    def _prepare_batch_input(self, input_ids) -> dict:
        position_ids = prepare_position_ids(self.module, input_ids)
        if not self.training_args.use_cpu:
            position_ids = position_ids.to(self.module.device)

        return dict(input_ids=input_ids, position_ids=position_ids)

    def freeze_model_embedding(self):
        self.module.get_input_embeddings().requires_grad_(False)

    def save_model(
        self,
        output_dir="./"
    ):
        if hasattr(self.module, "save_pretrained"):
            self.module.save_pretrained(output_dir)
        else:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(self.module.state_dict(), output_dir + '/pytorch_model.bin')
