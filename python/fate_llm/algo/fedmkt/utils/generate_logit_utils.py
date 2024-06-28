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
import torch.nn.functional as F
import gc
from fate_llm.algo.fedmkt.utils.vars_define import (
    PER_STEP_LOGITS,
    PER_STEP_INDICES,
    METRIC
)


class Metric(object):
    @classmethod
    def cal_metric(cls, logits, input_ids, attention_mask, labels, training_args):
        if training_args.metric_type == "ce":
            return cls.cal_ce(logits, input_ids, attention_mask, labels, training_args)
        else:
            raise NotImplemented(f"metric={training_args.metric_type} is not implemented yet")

    @classmethod
    def cal_ce(cls, logits, input_ids, attention_mask, labels, training_args):
        metric = F.cross_entropy(logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                                 labels[..., 1:].contiguous().view(-1), reduction="none").view(logits.size(0), -1)

        metric = (metric * attention_mask[..., 1:]).sum(dim=-1) / attention_mask[..., 1:].sum(dim=-1)

        return metric


class LogitsSelection(object):
    @classmethod
    def select_logits(cls, logits, training_args):
        if training_args.top_k_strategy == "highest":
            return cls.select_highest(logits, training_args.top_k_logits_keep)
        else:
            raise NotImplemented(f"logits selection strategy={training_args.top_k_strategy} is not implemented")

    @classmethod
    def select_highest(cls, logits, top_k_logits_keep):
        top_k_logits, top_k_indices = torch.topk(logits.cuda(), k=top_k_logits_keep)
        logits.cpu()

        return top_k_logits, top_k_indices


def generate_pub_data_logits(inputs, model, training_args, data_collator):
    input_keys = ["attention_mask", "input_ids", "labels"]
    inputs_per_batched = [dict() for _ in range(len(inputs[input_keys[1]]))]
    for key in input_keys:
        if key not in inputs:
            continue

        for idx, _in in enumerate(inputs[key]):
            inputs_per_batched[idx][key] = _in

    if "attention_mask" not in inputs:
        for idx in range(len(inputs_per_batched)):
            inputs_per_batched[idx]["attention_mask"] = [1] * len(inputs_per_batched[idx]["input_ids"])

    inputs_per_batched = data_collator(inputs_per_batched)

    input_ids = inputs_per_batched["input_ids"]
    attention_mask = inputs_per_batched["attention_mask"]
    labels = inputs_per_batched["labels"]

    device = next(model.parameters()).device
    if device.type == "cuda":
        input_ids = input_ids.cuda(device)
        attention_mask = attention_mask.cuda(device)
        labels = labels.cuda(device)

    model.eval()

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        metric = Metric.cal_metric(logits, input_ids, attention_mask, labels, training_args)

        input_ids.cpu()
        del input_ids
        attention_mask.cpu()
        del attention_mask
        labels.cpu()
        del labels
        logits.cpu()
        metric.cpu()

        if training_args.top_k_logits_keep is None:
            raise ValueError("Please specify top_k_logits_keep, fulling save will leak to memory exceeds")

        selected_logits, selected_indices = LogitsSelection.select_logits(logits=logits, training_args=training_args)
        selected_logits.cpu()
        selected_indices.cpu()

        inputs[PER_STEP_LOGITS] = selected_logits
        inputs[PER_STEP_INDICES] = selected_indices
        inputs[METRIC] = metric

        del logits

        gc.collect()

    model.train()

    return inputs
