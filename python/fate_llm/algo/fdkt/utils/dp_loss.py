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
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


NUMERICAL_STABILITY_CONSTANT = 1e-13


class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self, model_type, label_smoothing=-1, reduce=None):
        super().__init__()
        self.model_type = model_type
        self.label_smoothing = label_smoothing
        self.reduce = reduce

    def forward(self, logits, targets, mask):
        return sequence_cross_entropy_with_logits(logits, targets, mask, self.label_smoothing, self.reduce, self.model_type)


def sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce, model_type):
    if model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        logits = logits[:, :-1].contiguous()
        targets = targets[:, 1:]
        mask = torch.ones_like(targets).float()

    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    targets_flat = targets.reshape(-1, 1).long()

    if label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / float(num_classes)
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)

    negative_log_likelihood = negative_log_likelihood_flat.view(-1, logits.shape[1])

    loss = negative_log_likelihood * mask

    if reduce:
        loss = loss.sum(1) / (mask.sum(1) + NUMERICAL_STABILITY_CONSTANT)

        if reduce is "batch":
            loss = loss.mean()

    return loss
