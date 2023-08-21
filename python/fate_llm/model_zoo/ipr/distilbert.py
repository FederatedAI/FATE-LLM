from torch.nn import Module
from transformers import DistilBertForSequenceClassification, DistilBertForTokenClassification
from fate_llm.model_zoo.sign_block import recursive_replace_layernorm


class SignDistilBertForTokenClassification(Module):

    def __init__(self, model_path=None, num_labels=4) -> None:
        super().__init__()
        if model_path is None:
            model_path = 'distilbert-base-uncased'

        self.model_path = model_path
        self.model = DistilBertForTokenClassification.from_pretrained(
            model_path, num_labels=num_labels)

        # replace layernorm by SignatureLayerNorm
        sub_distilbert = self.model.distilbert.transformer.layer[3:]
        recursive_replace_layernorm(
            sub_distilbert,
            layer_name_set={'output_layer_norm'})

    def forward(self, input_dict):
        return self.model(**input_dict)


class SignDistilBertForSequenceClassification(Module):

    def __init__(
            self,
            model_path=None,
            num_labels=4,
            problem_type=None) -> None:
        super().__init__()
        if model_path is None:
            model_path = 'distilbert-base-uncased'

        self.model_path = model_path
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels, problem_type=problem_type)

        # replace layernorm by SignatureLayerNorm
        sub_distilbert = self.model.distilbert.transformer.layer[3:]
        recursive_replace_layernorm(
            sub_distilbert,
            layer_name_set={'output_layer_norm'})

    def forward(self, input_dict):
        return self.model(**input_dict)
