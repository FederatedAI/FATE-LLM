import torch as t
from torch import nn


def get_dropout_emulator_and_adapters(transformer_layers: nn.ModuleList, emulator_layer_num: int, adapter_top_layer_num: int, adapter_bottom_layer_num: int):
    
    assert adapter_bottom_layer_num > 0 and adapter_top_layer_num > 0, "adapter layer num must be greater than 0"
    assert emulator_layer_num < len(transformer_layers), "emulator layer num must be less than the number of transformer layers"
    assert adapter_bottom_layer_num + adapter_top_layer_num < len(transformer_layers), "adapter layer num must be less than the number of transformer layers"
    assert emulator_layer_num < len(transformer_layers) and emulator_layer_num > 0, "emulator layer num must be less than the number of transformer layers"

    bottom_idx = adapter_bottom_layer_num
    top_idx = len(transformer_layers) - adapter_top_layer_num
    bottom_layers = transformer_layers[:bottom_idx]
    top_layers = transformer_layers[top_idx:]
    kept_layers = transformer_layers[bottom_idx:top_idx]
    emulator = nn.ModuleList()
    stride = (len(kept_layers)-1) / (emulator_layer_num-1)

    for i in range(emulator_layer_num):
        idx = int(round(i * stride))
        emulator.append(kept_layers[idx])

    return nn.ModuleList(emulator), nn.ModuleList(bottom_layers), nn.ModuleList(top_layers)


class OffsiteTuningMainModel(t.nn.Module):
    
    """
    The model of the model provider in the offsite tuning setting.
    Implements the model provider's interface so that FATE framework
    can get the emulator, the adapter from this class and send them to 
    the data provider party for training
    You can use built in tools to get the drop out emulator, see the original paper for 
    details: https://arxiv.org/pdf/2302.04870.pdf
    """

    def __init__(self,):
        super().__init__()

    def get_adapter_top(self):
        raise NotImplementedError()

    def get_adapter_bottom(self):
        raise NotImplementedError()

    def get_emulator(self):
        raise NotImplementedError()

    def forward(self, **kwargs):
        raise NotImplementedError()

    def get_submodel_weights(self) -> dict:
        submodel_weights = {
            "emulator": {k: v.detach().cpu().numpy() for k, v in self.get_emulator().state_dict().items()},
            "adapter_top": {k: v.detach().cpu().numpy() for k, v in self.get_adapter_top().state_dict().items()},
            "adapter_bottom": {k: v.detach().cpu().numpy() for k, v in self.get_adapter_bottom().state_dict().items()}
        }

        return submodel_weights


class OffiteTuningSubModel(t.nn.Module):

    def __init__(self):
        super().__init__()
    
    def get_adapter_top(self):
        raise NotImplementedError()

    def get_adapter_bottom(self):
        raise NotImplementedError()

    def forward(self, **kwargs):
        raise NotImplementedError()

    def load_submodel_weights(self, submodel_weights: dict):
        emulator_weights = {k: t.tensor(v) for k, v in submodel_weights['emulator'].items()}
        adapter_top_weights = {k: t.tensor(v) for k, v in submodel_weights['adapter_top'].items()}
        adapter_bottom_weights = {k: t.tensor(v) for k, v in submodel_weights['adapter_bottom'].items()}

        emulator = self.get_emulator()
        adapter_top = self.get_adapter_top()
        adapter_bottom = self.get_adapter_bottom()

        emulator.load_state_dict(emulator_weights)
        adapter_top.load_state_dict(adapter_top_weights)
        adapter_bottom.load_state_dict(adapter_bottom_weights)
    



