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
import transformers
from fate_llm.model_zoo.pellm.parameter_efficient_llm import PELLM
from transformers.modeling_utils import unwrap_model


def get_model_class(model):
    if isinstance(model, PELLM):
        model = model._pe_lm

    model = unwrap_model(model)

    return model.__class__


def prepare_position_ids(model, input_ids):
    if get_model_class(model) == transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel:
        return _get_position_ids_for_gpt2(input_ids)
    else:
        raise ValueError(f"Can not prepare position_ids for model_type={model.__class__}")


def _get_position_ids_for_gpt2(input_ids):
    past_length = 0
    position_ids = torch.arange(past_length, input_ids.shape[-1] + past_length, dtype=torch.long,
                                device=input_ids.device)
    position_ids = position_ids.unsqueeze(0)
    position_ids = position_ids.repeat(input_ids.shape[0], 1)

    return position_ids
