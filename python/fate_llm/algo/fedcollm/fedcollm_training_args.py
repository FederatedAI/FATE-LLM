#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from dataclasses import dataclass, field
from ...trainer.seq2seq_trainer import Seq2SeqTrainingArguments


@dataclass
class FedCoLLMTrainingArguments(Seq2SeqTrainingArguments):
    """
    top-k logits select params
    """
    top_k_logits_keep: int = field(default=128)
    top_k_strategy: str = field(default="highest")

    """
    distillation params
    """
    distill_lambda: float = field(default=1.0)
    distill_temperature: float = field(default=1.0)
    server_public_data_local_epoch: int = field(default=1)
    client_public_data_local_epoch: int = field(default=1)
    client_priv_data_local_epoch: int = field(default=1)
    global_epochs: int = field(default=1)

    extra_args = ["top_k_logits_keep", "top_k_strategy", "distill_lambda",
                  "distill_temperature", "server_public_data_local_epoch",
                  "client_public_data_local_epoch", "client_priv_data_local_epoch",
                  "global_epochs"]

    def to_dict(self):
        from dataclasses import fields
        from enum import Enum
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def _pop_extra(self):
        args = self.to_dict()
        for arg in self.extra_args:
            args.pop(arg)

        return args

    def to_slm_seq_training_args(self):
        args = self._pop_extra()
        args["num_train_epochs"] = self.client_priv_data_local_epoch

        return Seq2SeqTrainingArguments(**args)

    def to_fedco_slm_training_args(self):
        args = self._pop_extra()
        args["num_train_epochs"] = self.client_pub_data_local_epoch

        return Seq2SeqTrainingArguments(**args)

    def to_fedco_llm_training_args(self):
        args = self._pop_extra()
        args["num_train_epochs"] = self.server_pub_data_local_epoch

        return Seq2SeqTrainingArguments(**args)
