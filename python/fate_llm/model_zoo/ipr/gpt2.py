from torch.nn import Module
from transformers import GPT2ForTokenClassification, GPT2ForSequenceClassification
from federatedml.nn.model_zoo.ipr.sign_block import recursive_replace_layernorm


class SignGPT2ForTokenClassification(Module):

    def __init__(self, model_path=None, num_labels=4) -> None:
        super().__init__()
        if model_path is None:
            model_path = 'gpt2'
        
        self.model_path = model_path
        self.model = GPT2ForTokenClassification.from_pretrained(model_path, num_labels=num_labels)

        # replace layernorm by SignatureLayerNorm
        sub_gpt2 = self.model.transformer.h[10:]
        recursive_replace_layernorm(sub_gpt2)

    def forward(self, input_dict):
        return self.model(**input_dict)


class SignGPT2ForSequenceClassification(Module):

    def __init__(self, model_path=None, num_labels=2) -> None:
        super().__init__()
        if model_path is None:
            model_path = 'gpt2'
        
        self.model_path = model_path
        self.model = GPT2ForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

        # replace layernorm by SignatureLayerNorm
        sub_gpt2 = self.model.transformer.h[10:]
        recursive_replace_layernorm(sub_gpt2)

    def forward(self, input_dict):
        return self.model(**input_dict)
