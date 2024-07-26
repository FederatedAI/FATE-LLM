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
import torch
from transformers import AutoModelForCausalLM


class HFAutoModelForCausalLM:

    def __init__(self, pretrained_model_name_or_path, *model_args, **kwargs) -> None:
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_args = model_args
        self.kwargs = kwargs
        if "torch_dtype" in self.kwargs and self.kwargs["torch_dtype"] != "auto":
            dtype = self.kwargs.pop("torch_dtype")
            self.kwargs["torch_dtype"] = getattr(torch, dtype)

    def load(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_name_or_path, *self.model_args, **self.kwargs
        )
        return model
