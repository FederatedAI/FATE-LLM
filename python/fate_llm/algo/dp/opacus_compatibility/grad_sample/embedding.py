from typing import Dict
import types

import torch
import torch.nn as nn


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
