from fate_llm.model_zoo.offsite_tuning.offsite_tuning_model import OffsiteTuningSubModel, OffsiteTuningMainModel, get_dropout_emulator_and_adapters, split_numpy_array, recover_numpy_array
from transformers import LlamaConfig, LlamaForCausalLM


class LlamaMainModel(OffsiteTuningMainModel):

    def __init__(
            self,
            model_name_or_path,
            emulator_layer_num: int,
            adapter_top_layer_num: int = 2,
            adapter_bottom_layer_num: int = 2):

        self.model_name_or_path = model_name_or_path
        super().__init__(
            emulator_layer_num,
            adapter_top_layer_num,
            adapter_bottom_layer_num)

    def get_base_model(self):
        return LlamaForCausalLM.from_pretrained(self.model_name_or_path)

    def get_model_transformer_blocks(self, model: LlamaForCausalLM):
        return model.model.layers
    
    def forward(self, x):
        return self.model(**x)

    def get_additional_param_state_dict(self):
        # get parameter of additional parameter
        model = self.model
        param_dict = {
            'wte': model.model.embed_tokens,
            'last_ln_f': model.model.norm
        }

        addition_weights = self.get_numpy_state_dict(param_dict)

        wte = addition_weights.pop('wte')
        wte_dict = split_numpy_array(wte, 25, 'wte')
        addition_weights.update(wte_dict)
        return addition_weights

    def load_additional_param_state_dict(self, submodel_weights: dict):
        # load additional weights:
        model = self.model
        param_dict = {
            'wte': model.model.embed_tokens,
            'last_ln_f': model.model.norm
        }

        new_submodel_weight = {}
        new_submodel_weight['last_ln_f'] = submodel_weights['last_ln_f']
        wte_dict = {}
        for k, v in submodel_weights.items():
            if 'wte' in k:
                wte_dict[k] = v
        wte = recover_numpy_array(wte_dict, 'wte')
        new_submodel_weight['wte'] = wte
        self.load_numpy_state_dict(param_dict, new_submodel_weight)

    def forward(self, x):
        return self.model(**x)


class LlamaSubModel(OffsiteTuningSubModel):

    def __init__(
            self,
            model_name_or_path,
            emulator_layer_num: int,
            adapter_top_layer_num: int = 2,
            adapter_bottom_layer_num: int = 2,
            fp16_mix_precision=False,
            partial_weight_decay=None):

        self.model_name_or_path = model_name_or_path
        self.emulator_layer_num = emulator_layer_num
        self.adapter_top_layer_num = adapter_top_layer_num
        self.adapter_bottom_layer_num = adapter_bottom_layer_num
        super().__init__(
            emulator_layer_num,
            adapter_top_layer_num,
            adapter_bottom_layer_num,
            fp16_mix_precision)
        self.partial_weight_decay = partial_weight_decay

    def get_base_model(self):
        total_layer_num = self.emulator_layer_num + \
            self.adapter_top_layer_num + self.adapter_bottom_layer_num
        config = LlamaConfig.from_pretrained(self.model_name_or_path)
        config.num_hidden_layers = total_layer_num
        # initialize a model without pretrained weights
        return LlamaForCausalLM(config)

    def get_model_transformer_blocks(self, model: LlamaForCausalLM):
        return model.model.layers

    def forward(self, x):
        return self.model(**x)

    def get_additional_param_state_dict(self):
        # get parameter of additional parameter
        model = self.model
        param_dict = {
            'wte': model.model.embed_tokens,
            'last_ln_f': model.model.norm
        }

        addition_weights = self.get_numpy_state_dict(param_dict)

        wte = addition_weights.pop('wte')
        wte_dict = split_numpy_array(wte, 25, 'wte')
        addition_weights.update(wte_dict)
        return addition_weights

    def load_additional_param_state_dict(self, submodel_weights: dict):
        # load additional weights:
        model = self.model
        param_dict = {
            'wte': model.model.embed_tokens,
            'last_ln_f': model.model.norm
        }

        new_submodel_weight = {}
        new_submodel_weight['last_ln_f'] = submodel_weights['last_ln_f']
        wte_dict = {}
        for k, v in submodel_weights.items():
            if 'wte' in k:
                wte_dict[k] = v
        wte = recover_numpy_array(wte_dict, 'wte')
        new_submodel_weight['wte'] = wte
        self.load_numpy_state_dict(param_dict, new_submodel_weight)

    def forward(self, x):
        return self.model(**x)

    def parameters(self, recurse=True):
        if self.partial_weight_decay is None:
            return super().parameters(recurse)
        elif isinstance(self.partial_weight_decay, float):
            no_decay = ["bias", "layer_norm.weight"]
            return [
                {
                    "params": [
                        p for n, p in self.named_parameters() if not any(
                            nd in n for nd in no_decay)], "weight_decay": self.partial_weight_decay}, {
                    "params": [
                        p for n, p in self.named_parameters() if any(
                            nd in n for nd in no_decay)], "weight_decay": 0.0}]
        else:
            raise ValueError(
                f"partial_weight_decay should be None or float, but got {self.partial_weight_decay}")