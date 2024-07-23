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
import os.path

import torch
import logging
from dataclasses import dataclass, field
from ...trainer.seq2seq_trainer import Seq2SeqTrainingArguments
from typing import Optional, Callable
from fate.arch import Context
from transformers import PreTrainedTokenizer
from .utils.text_generate import slm_text_generate, general_text_generate
from .cluster.cluster import SentenceCluster
from fate_llm.inference.inference_base import Inference


logger = logging.getLogger(__name__)
SLM_SYNTHETIC_DATA = "slm_synthetic_data"
LLM_AUG_DATA = "llm_aug_data"


@dataclass
class FDKTTrainingArguments(Seq2SeqTrainingArguments):
    """
    slm parameters
    """
    dp_training: bool = field(default=True)
    target_epsilon: float = field(default=3)
    target_delta: float = field(default=1e-5)
    freeze_embedding: bool = field(default=False)
    device_id: int = field(default=0)
    slm_generation_config: dict = field(default=None)
    slm_generation_batch_size: dict = field(default=None)

    """
    slm generation config
    """
    seq_num_for_single_category: int = field(default=None)

    """
    dp loss params
    """
    label_smoothing_factor = 0.02
    loss_reduce = True

    """
    llm parameters
    """
    sample_num_per_cluster: int = field(default=None)
    filter_data_batch_size: int = field(default=2)
    filter_prompt_max_length: int = field(default=2048)
    filter_generation_config: dict = field(default=None)

    aug_generation_config: dict = field(default=None)
    aug_prompt_num: int = field(default=None)
    aug_data_batch_size: int = field(default=2)
    aug_prompt_max_length: int = field(default=2048)

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


class FDKTSLM(object):
    def __init__(
        self,
        ctx: Context,
        model: torch.nn.Module,
        training_args: FDKTTrainingArguments,
        train_set,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        data_collator: Callable = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super(FDKTSLM, self).__init__()
        self.ctx = ctx
        self.training_args = training_args
        self.train_set = train_set
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.data_collator = data_collator

        if not self.training_args.use_cpu:
            self.model.cuda(self.training_args.device_id)

    def aug_data(self):
        if self.training_args.dp_training:
            self.dp_train()

        prefix_prompt_ids_dict = self.train_set.get_generate_prompt(tokenize=True)
        generated_text = slm_text_generate(
            self.model,
            self.tokenizer,
            prompt_ids_dict=prefix_prompt_ids_dict,
            seq_num_for_single_category=self.training_args.seq_num_for_single_category,
            batch_size=self.training_args.slm_generation_batch_size,
            use_cpu=self.training_args.use_cpu,
            generation_config=self.training_args.slm_generation_config
        )

        if not self.training_args.use_cpu:
            self.model.cpu()
            torch.cuda.empty_cache()

        self.sync_synthetic_dataset(generated_text)

        return self.sync_aug_data()

    def dp_train(self):
        from ..dp import DPTrainer, DPTrainingArguments, get_model_class
        from .utils.dp_loss import SequenceCrossEntropyLoss
        dp_training_args = DPTrainingArguments(
            target_delta=self.training_args.target_delta,
            target_epsilon=self.training_args.target_epsilon,
            freeze_embedding=self.training_args.freeze_embedding,
            device_id=self.training_args.device_id,
            num_train_epochs=self.training_args.num_train_epochs,
            per_device_train_batch_size=self.training_args.per_device_train_batch_size,
            output_dir="/" if self.training_args.output_dir is None else self.training_args.output_dir
        )

        loss_fn = SequenceCrossEntropyLoss(
            get_model_class(self.model).__name__,
            label_smoothing=self.training_args.label_smoothing_factor,
            reduce=self.training_args.loss_reduce
        )

        dp_trainer = DPTrainer(
            model=self.model,
            training_args=dp_training_args,
            train_set=self.train_set,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            data_collator=self.data_collator,
            loss_fn=loss_fn
        )

        dp_trainer.train()

    def sync_synthetic_dataset(self, data):
        self.ctx.arbiter.put(SLM_SYNTHETIC_DATA, data)

    def sync_aug_data(self):
        return self.ctx.arbiter.get(LLM_AUG_DATA)

    def save_model(
        self,
        output_dir="./"
    ):
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)
        else:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(self.model.state_dict(), output_dir + '/pytorch_model.bin')


class FDKTLLM(object):
    def __init__(
        self,
        ctx: Context,
        embedding_model: torch.nn.Module,
        training_args: FDKTTrainingArguments,
        dataset,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        inference_inst: Optional[Inference] = None,
    ):
        super(FDKTLLM, self).__init__()
        self.ctx = ctx
        self.inference_inst = inference_inst
        self.embedding_model = embedding_model
        self.dataset = dataset
        self.training_args = training_args
        self.model = model
        self.tokenizer = tokenizer

        if self.inference_inst is None and (self.model is None or self.tokenizer is None):
            raise ValueError("Inference_inst and Model are both empty, should provided one")
        if self.model is not None and self.training_args.device_id is not None and not self.training_args.use_cpu:
            self.model.cuda(self.training_args.device_id)

    def sync_synthetic_data(self):
        return self.ctx.guest.get(SLM_SYNTHETIC_DATA)

    def sync_aug_data(self, aug_data):
        self.ctx.guest.put(LLM_AUG_DATA, aug_data)

    def aug_data(self):
        slm_data = self.sync_synthetic_data()

        filter_data = self.filter_data(slm_data)

        aug_prompts = self.dataset.prepare_augment(
            filter_data["inputs"],
            filter_data["labels"],
            aug_prompt_num=self.training_args.aug_prompt_num
        )

        aug_data = self._aug(aug_prompts)
        self.sync_aug_data(aug_data)

    def _aug(self, aug_prompts):
        aug_responses = general_text_generate(
            inference_inst=self.inference_inst,
            model=self.model,
            tokenizer=self.tokenizer,
            generation_config=self.training_args.aug_generation_config,
            prompts=aug_prompts,
            batch_size=self.training_args.aug_data_batch_size,
            use_cpu=self.training_args.use_cpu,
            prompt_max_length=self.training_args.aug_prompt_max_length
        )

        aug_data = self.dataset.abstract_from_augmented(aug_responses)

        return aug_data

    def filter_data(self, slm_data):
        clustered_sentences, clustered_labels = self.cluster_data(slm_data)
        filter_prompts = self.dataset.prepare_query_to_filter_clustered(clustered_sentences, clustered_labels)
        filter_responses = general_text_generate(
            inference_inst=self.inference_inst,
            model=self.model,
            tokenizer=self.tokenizer,
            generation_config=self.training_args.filter_generation_config,
            prompts=filter_prompts,
            batch_size=self.training_args.filter_data_batch_size,
            use_cpu=self.training_args.use_cpu,
            prompt_max_length=self.training_args.filter_prompt_max_length
        )

        filtered_sentences, filtered_labels = self.dataset.parse_clustered_response(
            clustered_sentence=clustered_sentences,
            clustered_labels=clustered_labels,
            response_list=filter_responses
        )

        return dict(
            inputs=filtered_sentences,
            labels=filtered_labels
        )

    def cluster_data(self, slm_data):
        sentences = slm_data["inputs"]
        labels = slm_data["labels"]

        n_clusters = (len(sentences) + self.training_args.sample_num_per_cluster - 1) // self.training_args.sample_num_per_cluster

        cluster_ret = SentenceCluster(model=self.embedding_model, n_clusters=n_clusters).cluster(sentences)

        clustered_sentences = [[] for _ in range(n_clusters)]
        clustered_labels = [[] for _ in range(n_clusters)]

        for sentence_id, cluster_id in enumerate(cluster_ret):
            clustered_sentences[cluster_id].append(sentences[sentence_id])
            clustered_labels[cluster_id].append(labels[sentence_id])

        return clustered_sentences, clustered_labels
