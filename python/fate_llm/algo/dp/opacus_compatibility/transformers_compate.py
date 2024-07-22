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
