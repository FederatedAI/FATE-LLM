#
# NOTE: The implementations of FedMKTTrainer is modified from FuseAI/FuseLLM
# Copyright FuseAI
#
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
import torch
from torch.nn.functional import kl_div, log_softmax, cross_entropy
from transformers import Seq2SeqTrainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from fate_llm.algo.fedmkt.utils.vars_define import (
    SELF_TARGET_DIST,
    OTHER_TARGET_DIST,
    ALIGNED_OTHER_METRIC,
    METRIC,
)

logger = logging.getLogger(__name__)


class FedMKTTrainer(Seq2SeqTrainer):
    """
    modified from https://github.com/fanqiwan/FuseAI/blob/main/FuseLLM/src/utils/trainer.py#L22
    """
    blending_num: int = 2
    distill_loss_type: str = "ce"
    lm_loss_weight: float = 0.9
    distill_strategy = "greater"

    def __init__(self, *args, **kwargs):
        blending_num = kwargs.pop("blending_num", 1)
        distill_loss_type = kwargs.pop("distill_loss_type", "ce")
        lm_loss_weight = kwargs.pop("lm_loss_weight", 0.9)
        distill_strategy = kwargs.pop("distill_strategy", "greater")
        super(FedMKTTrainer, self).__init__(*args, **kwargs)
        self.blending_num = blending_num
        self.distill_loss_type = distill_loss_type
        self.lm_loss_weight = lm_loss_weight
        self.distill_strategy = distill_strategy

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        base_target_dist = inputs.pop(SELF_TARGET_DIST)
        base_metric = inputs.pop(METRIC)

        aligned_target_dists = []
        aligned_metrics = []
        for i in range(self.blending_num):
            aligned_target_dists.append(inputs.pop(f"{OTHER_TARGET_DIST}_{i}"))
            aligned_metrics.append(inputs.pop(f"{ALIGNED_OTHER_METRIC}_{i}"))

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        batch_size, seq_len, vocab_size = outputs["logits"].size(0), outputs["logits"].size(1), outputs["logits"].size(2)

        aligned_rewards = []
        for i in range(self.blending_num):
            aligned_rewards.append((1 / torch.exp(torch.tensor(aligned_metrics[i], dtype=torch.bfloat16))).to(loss.device))

        base_reward = (1 / torch.exp(torch.tensor(base_metric, dtype=torch.bfloat16))).to(loss.device)

        if self.distill_strategy == "greater":
            base_reward_expanded = base_reward.unsqueeze(-1).unsqueeze(-1).expand_as(base_target_dist)
            aligned_rewards_expanded = [
                aligned_rewards[i].unsqueeze(-1).unsqueeze(-1).expand_as(aligned_target_dists[i])
                for i in range(self.blending_num)
            ]
            target_dist_list = []
            reward_list = []
            if base_target_dist is not None:
                target_dist_list.append(base_target_dist)
                reward_list.append(base_reward_expanded)

            target_dist_list.extend(aligned_target_dists)
            reward_list.extend(aligned_rewards_expanded)

            stacked_dists = torch.stack(target_dist_list, dim=-1)
            stacked_rewards = torch.stack(reward_list, dim=-1)
            max_reward_indices = torch.argmax(stacked_rewards, dim=-1, keepdim=True)
            target_dist = torch.gather(stacked_dists, -1, max_reward_indices).squeeze(-1)
        elif self.distill_strategy == "weighted_mean":
            weights = torch.stack(
                [base_reward] + aligned_rewards, dim=1
            )
            normalized_weights = torch.softmax(weights, dim=1)
            weight_labels = normalized_weights[:, 0].unsqueeze(1).unsqueeze(2) * base_target_dist
            for i in range(self.blending_num):
                weight_labels += normalized_weights[:, i + 1].unsqueeze(1).unsqueeze(2) * aligned_target_dists[i]

            target_dist = (
                weight_labels
            )
        else:
            raise ValueError(f"distill_strategy={self.distill_strategy}")

        if self.distill_loss_type == "ce":
            loss_lm = cross_entropy(
                input=outputs["logits"].view(-1, vocab_size),
                target=target_dist.view(-1, vocab_size),
                reduction="none",
            ).view(batch_size, -1)
        elif self.distill_loss_type == "kl":
            loss_lm = kl_div(
                input=log_softmax(outputs["logits"], dim=-1),
                target=target_dist,
                log_target=False,
                reduction="none").sum(dim=-1)
        else:
            raise ValueError(f"Not implement distill_loss_type={self.distill_loss_type}")

        loss_lm = (loss_lm * inputs["attention_mask"]).sum() / inputs["attention_mask"].sum()
        loss = self.lm_loss_weight * loss + (1.0 - self.lm_loss_weight) * loss_lm

        return (loss, outputs) if return_outputs else loss
