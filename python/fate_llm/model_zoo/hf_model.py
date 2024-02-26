from transformers import AutoModelForCausalLM


class HFAutoModelForCausalLM:

    def __init__(self, pretrained_model_name_or_path, *model_args, **kwargs) -> None:
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_args = model_args
        self.kwargs = kwargs

    def load(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_name_or_path, *self.model_args, **self.kwargs
        )
        return model
