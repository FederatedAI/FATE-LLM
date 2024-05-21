from typing import List

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names


def get_decay_parameter_names(model) -> List[str]:
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm

    NOTE: This function is copied from transformers
    # Copyright 2020-present the HuggingFace Inc. team.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def get_optimizer_parameters_grouped_with_decay(model, weight_decay: float) -> List[dict]:
    """
    Get the parameters grouped by whether they should have weight decay applied
    """
    decay_parameters = get_decay_parameter_names(model)
    params_no_decay = []
    params_decay = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if n in decay_parameters:
                params_decay.append(p)
            else:
                params_no_decay.append(p)
    grouped_parameters_with_decay = [
        {"params": params_no_decay, "weight_decay": 0.0},
        {"params": params_decay, "weight_decay": weight_decay},
    ]
    return grouped_parameters_with_decay
