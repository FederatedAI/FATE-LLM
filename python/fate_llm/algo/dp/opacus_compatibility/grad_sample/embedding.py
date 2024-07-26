#
#  Copyright (c) Meta Platforms, Inc. and affiliates.
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
from typing import Dict


# the function is modified from https://github.com/pytorch/opacus/blob/main/opacus/grad_sample/embedding.py#L25,
# avoid dtype error when backprops's dtype isn't torch.float32
def compute_embedding_grad_sample(
    layer: nn.Embedding, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``nn.Embedding`` layer.

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    activations = activations[0]
    ret = {}
    if layer.weight.requires_grad:
        saved = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True

        batch_size = activations.shape[0]
        if batch_size == 0:
            ret[layer.weight] = torch.zeros_like(layer.weight).unsqueeze(0)
            return ret

        index = (
            activations.unsqueeze(-1)
            .expand(*activations.shape, layer.embedding_dim)
            .reshape(batch_size, -1, layer.embedding_dim)
        )
        grad_sample = torch.zeros(
            batch_size, *layer.weight.shape, device=layer.weight.device, dtype=backprops.dtype
        )
        grad_sample.scatter_add_(
            1, index, backprops.reshape(batch_size, -1, layer.embedding_dim)
        )
        torch.backends.cudnn.deterministic = saved
        ret[layer.weight] = grad_sample

    return ret
