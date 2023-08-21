from fate_llm.model_zoo.offsite_tuning.offsite_tuning_model import OffsiteTuningSubModel, OffsiteTuningMainModel, get_dropout_emulator_and_adapters
from transformers import GPT2LMHeadModel, GPT2Config
from torch import nn
import torch


class GPT2LMHeadMainModel(OffsiteTuningMainModel):

    def __init__(self, model_name_or_path, emulator_layer_num: int, adapter_top_layer_num: int = 2, 
                       adapter_bottom_layer_num: int = 2):
        
        self.model_name_or_path = model_name_or_path
        super().__init__(emulator_layer_num, adapter_top_layer_num, adapter_bottom_layer_num)

    def get_base_model(self):
        return GPT2LMHeadModel.from_pretrained(self.model_name_or_path)

    def get_model_transformer_blocks(self, model: GPT2LMHeadModel):
        return model.transformer.h
    
    def forward(self, x):
        return self.model(**x)

    def get_additional_parameter(self, model) -> dict:
        return {
            'wte': model.transformer.wte,
            'wpe': model.transformer.wpe,
            'last_ln_f': model.transformer.ln_f
        }

    def forward(self, x):
        return self.model(**x)


class GPT2LMHeadSubModel(OffsiteTuningSubModel):

    def __init__(self, model_name_or_path, emulator_layer_num: int, adapter_top_layer_num: int = 2, 
                       adapter_bottom_layer_num: int = 2, fp16_mix_precision=False, partial_weight_decay=None):
        
        self.model_name_or_path = model_name_or_path
        self.emulator_layer_num = emulator_layer_num
        self.adapter_top_layer_num = adapter_top_layer_num
        self.adapter_bottom_layer_num = adapter_bottom_layer_num
        super().__init__(emulator_layer_num, adapter_top_layer_num, adapter_bottom_layer_num, fp16_mix_precision)
        self.partial_weight_decay = partial_weight_decay

    def get_base_model(self):
        total_layer_num = self.emulator_layer_num + self.adapter_top_layer_num + self.adapter_bottom_layer_num
        config = GPT2Config.from_pretrained(self.model_name_or_path)
        config.num_hidden_layers = total_layer_num
        # initialize a model without pretrained weights
        return GPT2LMHeadModel(config)

    def get_model_transformer_blocks(self, model: GPT2LMHeadModel):
        return model.transformer.h
    
    def forward(self, x):
        return self.model(**x)

    def get_additional_parameter(self, model) -> dict:
        return {
            'wte': model.transformer.wte,
            'wpe': model.transformer.wpe,
            'last_ln_f': model.transformer.ln_f
        }

    def forward(self, x):
        return self.model(**x)

    def parameters(self, recurse=True):
        if self.partial_weight_decay is None:
            return super().parameters(recurse)
        elif isinstance(self.partial_weight_decay, float):
            no_decay = ["bias", "layer_norm.weight"]
            return [
                {
                    "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.partial_weight_decay
                },
                {
                    "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0
                }
            ]
        else:
            raise ValueError(f"partial_weight_decay should be None or float, but got {self.partial_weight_decay}")


if __name__ == "__main__":
    
    from transformers import GPT2Model

    model = GPT2LMHeadMainModel('gpt2-xl', 12, 2, 2)
    model_sub = GPT2LMHeadSubModel('gpt2-xl', 12, 2, 2, fp16_mix_precision=True)
    model_sub.load_submodel_weights(model.get_submodel_weights())
