from typing import List

import torch


def probability_from_amps(amps: List[List[float]], clip):
    """
    Get the probability distribution from the amplitude history

    formula: amp_i = clamp(amp_i, -clip, clip).abs().mean()
             amp_i = (amp_i - min(amp)) / (max(amp) - min(amp))
             prob_i = softmax(amp)_i

    :param amps: list of amplitude history
    :param clip: the clipping value
    :return:
    """
    amps = [torch.Tensor(amp) for amp in amps]
    amp = torch.stack([amp.clamp_(-clip, clip).abs_().mean() for amp in amps])
    return (amp - amp.min()).div_(amp.max() - amp.min() + 1e-10).softmax(0)


def directional_derivative_step(
    param_groups: List[dict],
    directional_derivative_seed: int,
    directional_derivative_value: torch.FloatTensor,
    lr: float = None,
    weight_decay: float = None,
) -> torch.FloatTensor:
    """
    perform a step update for the parameters of the model
    along the random direction z with the learning rate lr and the step size grad_projected_value

    Input:
    - param_groups (List[dict]): list of parameter groups
    - directional_derivative_seed (int): seed for the random direction
    - directional_derivative_value (torch.FloatTensor): the step size
    - lr (float, optional): learning rate
    - weight_decay (float, optional): weight decay
    """

    torch.manual_seed(directional_derivative_seed)
    for param_group in param_groups:
        weight_decay = param_group["weight_decay"] if weight_decay is None else weight_decay
        lr = param_group["lr"] if lr is None else lr
        for param in param_group["params"]:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if weight_decay is not None:
                param.data = param.data - lr * (directional_derivative_value * z + weight_decay * param.data)

            else:
                param.data = param.data - lr * (directional_derivative_value * z)

    return directional_derivative_value


def build_seed_candidates(k, low=0, high=2**32):
    """
    Build seed candidates for the random walk optimizer
    """
    return torch.randint(low, high, size=(k,), dtype=torch.long)


def get_even_seed_probabilities(k):
    """
    Get the even seed probabilities, i.e., 1/k for each seed
    """
    return torch.ones(k) / k
