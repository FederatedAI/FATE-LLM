from fate_llm.model_zoo.offsite_tuning.offsite_tuning_model import OffiteTuningSubModel, OffsiteTuningMainModel, get_dropout_emulator_and_adapters
from transformers import GPT2LMHeadModel, GPT2Config
from torch import nn
import torch



class GPT2LMHeadMainModel(OffsiteTuningMainModel):

    def __init__(self, model_name_or_path, emulator_layer_num: int, 
                        adapter_top_layer_num: int = 2, adapter_bottom_layer_num: int = 2, sync_embedding=True):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model_name_or_path = model_name_or_path
        self.emulator, self.adapter_bottom, self.adapter_top = get_dropout_emulator_and_adapters(
            transformer_layers=self.model.transformer.h,
            emulator_layer_num=emulator_layer_num,
            adapter_top_layer_num=adapter_top_layer_num,
            adapter_bottom_layer_num=adapter_bottom_layer_num
        )
        self.sync_embedding = sync_embedding
        if self.sync_embedding:
            self.wte = self.model.transformer.wte
            self.wpe = self.model.transformer.wpe

    def get_emulator(self):
        return self.emulator

    def get_adapter_bottom(self):
        return self.adapter_bottom

    def get_adapter_top(self):
        return self.adapter_top

    def get_submodel_weights(self) -> dict:
        weight_dict = super().get_submodel_weights()
        if self.sync_embedding:
            weight_dict["wte"] = {k: v.detach().cpu().numpy() for k, v in self.wte.state_dict().items()}
            weight_dict["wpe"] = {k: v.detach().cpu().numpy() for k, v in self.wpe.state_dict().items()}

        return weight_dict


class GPT2LMHeadSubModel(OffiteTuningSubModel):

    def __init__(self, config_name_or_path,
                emulator_layer_num: int, adapter_top_layer_num: int = 2, adapter_bottom_layer_num: int = 2):
        
        super().__init__()
        self.total_layer_num = emulator_layer_num + adapter_top_layer_num + adapter_bottom_layer_num
        config = GPT2Config.from_pretrained(config_name_or_path)
        config.num_hidden_layers = self.total_layer_num
        # initialize a model without pretrained weights
        self.model = GPT2LMHeadModel(config)
        self.emulator, self.adapter_bottom, self.adapter_top = get_dropout_emulator_and_adapters(
            transformer_layers=self.model.transformer.h,
            emulator_layer_num=emulator_layer_num,
            adapter_top_layer_num=adapter_top_layer_num,
            adapter_bottom_layer_num=adapter_bottom_layer_num
        )
        self.wte = self.model.transformer.wte
        self.wpe = self.model.transformer.wpe

    def get_emulator(self):
        return self.emulator

    def get_adapter_bottom(self):
        return self.adapter_bottom

    def get_adapter_top(self):
        return self.adapter_top

    def load_submodel_weights(self, weight_dict: dict):
        if "wte" in weight_dict:
            self.wte.load_state_dict({k: torch.tensor(v) for k, v in weight_dict['wte'].items()})
        if "wpe" in weight_dict:
            self.wpe.load_state_dict({k: torch.tensor(v) for k, v in weight_dict['wpe'].items()})
        super().load_submodel_weights(weight_dict)


if __name__ == "__main__":
    
    from transformers import GPT2Model

    model = GPT2LMHeadMainModel('gpt2-xl', 12, 2, 2, sync_embedding=True)
    model_sub = GPT2LMHeadSubModel('gpt2-xl', 12, 2, 2)
    model_sub.load_submodel_weights(model.get_submodel_weights())
