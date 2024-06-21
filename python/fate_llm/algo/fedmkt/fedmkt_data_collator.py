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
from torch.nn.functional import softmax
from transformers import DataCollatorForSeq2Seq
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from typing import Optional, Any, Union
import logging
from fate_llm.algo.fedmkt.utils.vars_define import (
    ALIGNED_OTHER_LOGITS,
    ALIGNED_OTHER_INDICES,
    PER_STEP_LOGITS,
    PER_STEP_INDICES,
    SELF_TARGET_DIST,
    OTHER_TARGET_DIST
)


logger = logging.getLogger(__name__)


class DataCollatorForFedMKT(DataCollatorForSeq2Seq):
    """modified from https://github.com/fanqiwan/FuseAI/blob/main/FuseLLM/src/utils/data_collator.py#L135"""
    """
    Copyright FuseAI

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    blending_num: int = 1
    distill_teacher_temperature: float = 1.0
    vocab_size: int = None
    dtype: torch.dtype = torch.bfloat16

    def __init__(self, *args, **kwargs):
        blending_num = kwargs.pop("blending_num", 4)
        vocab_size = kwargs.pop("vocab_size", None)
        dtype = kwargs.pop("dtype", torch.dtype)
        super(DataCollatorForFedMKT, self).__init__(*args, **kwargs)
        self.blending_num = blending_num
        self.vocab_size = vocab_size if vocab_size is not None else len(self.tokenizer.get_vocab())
        self.pad_id = self.tokenizer.pad_token_id
        self.dtype = dtype

    def __call__(self, features, return_tensors=None):
        extra_features = dict()
        feature_keys = list(features[0].keys())
        for f_key in feature_keys:
            if f_key not in ["input_ids", "attention_mask", "labels"]:
                extra_features[f_key] = []
                for feature in features:
                    extra_features[f_key].append(feature.pop(f_key))

        features = super().__call__(features=features, return_tensors=return_tensors)

        features.update(extra_features)

        batch_size = features["input_ids"].size(0)
        base_target_dist = torch.zeros(batch_size, self.max_length, self.vocab_size).to(self.dtype)
        aligned_target_dists = [torch.zeros(batch_size, self.max_length, self.vocab_size).to(self.dtype)
                                for _ in range(self.blending_num)]

        for i in range(batch_size):
            base_seq_len = len(features[PER_STEP_LOGITS][i])
            for j in range(self.max_length):
                if j < base_seq_len:
                    base_logits = torch.tensor(features[PER_STEP_LOGITS][i][j], dtype=self.dtype)
                    base_prob = softmax(base_logits / self.distill_teacher_temperature, -1)
                    base_indices = torch.tensor(features[PER_STEP_INDICES][i][j])
                    base_target_dist[i][j] = base_target_dist[i][j].scatter_(-1, base_indices, base_prob)

                    for k in range(self.blending_num):
                        per_step_aligned_indices_key = f"{ALIGNED_OTHER_INDICES}_{k}"
                        per_step_aligned_logits_key = f"{ALIGNED_OTHER_LOGITS}_{k}"
                        if len(features[per_step_aligned_indices_key][i][j]) > 0:
                            aligned_logits = torch.tensor(features[per_step_aligned_logits_key][i][j], dtype=self.dtype)
                            aligned_prob = softmax(aligned_logits / self.distill_teacher_temperature, -1)
                            aligned_indices = torch.tensor(features[per_step_aligned_indices_key][i][j])
                            aligned_target_dists[k][i][j] = aligned_target_dists[k][i][j].scatter_(-1, aligned_indices, aligned_prob)
                        else:
                            aligned_target_dists[k][i][j] = base_target_dist[i][j]

                else:  # padding position
                    base_target_dist[i][j][self.pad_id] = 1.0
                    for k in range(self.blending_num):
                        aligned_target_dists[k][i][j][self.pad_id] = 1.0

        features.pop(PER_STEP_LOGITS)
        features.pop(PER_STEP_INDICES)
        for i in range(self.blending_num):
            features.pop(f"{ALIGNED_OTHER_LOGITS}_{i}")
            features.pop(f"{ALIGNED_OTHER_INDICES}_{i}")
            features[f"{OTHER_TARGET_DIST}_{i}"] = aligned_target_dists[i]

        features[SELF_TARGET_DIST] = base_target_dist

        return features
