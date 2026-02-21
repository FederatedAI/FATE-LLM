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
from torch.nn.functional import kl_div, log_softmax, softmax
from transformers import Seq2SeqTrainer
from fate_llm.algo.fedmkt.utils.generate_logit_utils import LogitsSelection
from fate_llm.algo.fedmkt.utils.vars_define import (
    PER_STEP_LOGITS,
    PER_STEP_INDICES,
)
from types import SimpleNamespace

logger = logging.getLogger(__name__)


def computing_kd_loss(src_logits, dst_logits, loss_mask):
    src_logits = src_logits[loss_mask]
    dst_logits = dst_logits[loss_mask]

    return kl_div(
        log_softmax(src_logits, dim=-1, dtype=torch.float32),
        dst_logits,
        log_target=False,
        reduction="none").sum(dim=-1)


def recovery_logits(
    top_k_logits,
    top_k_indices,
    batch_size,
    max_length,
    vocab_size,
    dtype,
    device,
    pad_id,
    distill_temperature
):
    logits = torch.zeros(batch_size, max_length, vocab_size).to(dtype).to(device)
    for i in range(batch_size):
        base_seq_len = len(top_k_logits[i])
        for j in range(max_length):
            if j < base_seq_len:
                base_logits = torch.tensor(top_k_logits[i][j], dtype=dtype)
                base_prob = softmax(base_logits / distill_temperature, -1)
                base_indices = torch.tensor(top_k_indices[i][j])
                base_prob = base_prob.to(device)
                base_indices = base_indices.cuda(device)
                logits[i][j] = logits[i][j].scatter_(-1, base_indices, base_prob)
            else:  # padding position
                logits[i][j][pad_id] = 1.0

    return logits


class FedCoLLMTrainer(Seq2SeqTrainer):
    distill_lambda: float = 1.0
    distill_temperature: float = 1.0
    other_logits = None
    dtype: torch.dtype = torch.bfloat16
    vocab_size: int = None
    max_length: int = None
    top_k_args: SimpleNamespace = None

    def __init__(self, **kwargs):
        distill_lambda = kwargs.pop("distill_lambda", 1.0)
        distill_temperature = kwargs.pop("distill_temperature", 1.0)
        other_logits = kwargs.pop("other_logits")
        vocab_size = kwargs.pop("vocab_size")
        max_length = kwargs.pop("max_length")
        top_k_args = kwargs.pop("top_k_args")
        super(FedCoLLMTrainer, self).__init__(**kwargs)
        self.distill_lambda = distill_lambda
        self.distill_temperature = distill_temperature
        self.other_logits = other_logits
        self.pad_id = self.tokenizer.pad_token_id
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.top_k_args = top_k_args

    def compute_loss(self,  model, inputs, return_outputs=False):
        lm_outputs = model(**inputs['inputs'])
        lm_loss = lm_outputs.loss
        logits = lm_outputs.logits
        other_logits = self.other_logits[inputs["indexes"]]

        batch_size = logits.shape[0]

        top_k_logits, top_k_indices = LogitsSelection.select_logits(logits, self.top_k_args)

        dst_logits = recovery_logits(
            other_logits[PER_STEP_INDICES],
            other_logits[PER_STEP_INDICES],
            batch_size,
            self.max_length,
            self.vocab_size,
            self.dtype,
            logits.device,
            self.pad_id,
            self.distill_temperature
        )

        src_logits = recovery_logits(
            top_k_logits,
            top_k_indices,
            batch_size,
            self.max_length,
            self.vocab_size,
            self.dtype,
            logits.device,
            self.pad_id,
            self.distill_temperature
        )

        loss_mask = (inputs["inputs"]["labels"] != -100)
        kl_loss = computing_kd_loss(src_logits, dst_logits, loss_mask=loss_mask).sum()

        return lm_loss + self.distill_lambda * kl_loss
