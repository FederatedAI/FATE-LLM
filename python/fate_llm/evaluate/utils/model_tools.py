#
#  Copyright 2024 The FATE Authors. All Rights Reserved.
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

import os
from transformers import AutoModel, AutoTokenizer
from lm_eval.models.huggingface import HFLM


def load_model_from_path(model_path, peft_path=None, peft_config=None, model_args=None):
    model_args = model_args or {}
    if peft_path is None:
        if os.path.isfile(model_path):
            return HFLM(pretrained=model_path, **model_args)
        else:
            raise ValueError(f"given model path is not valid, please check: {model_path}")
    else:
        import torch
        from peft import PeftModel, PeftConfig, LoraConfig, TaskType, get_peft_model
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model.half()
        model.eval()
        peft_config = peft_config or {}
        peft_config=LoraConfig(**peft_config)
        model = get_peft_model(model, peft_config)
        model.load_state_dict(torch.load(peft_path), strict=False)
        model.model.half()
        HFLM(pretrained=model, tokenizer=tokenizer, **model_args)


def load_model(model_path, peft_path=None, model_args=None):
    model_args = model_args or {}
    return HFLM(pretrained=model_path, peft_path=peft_path, **model_args)


def load_by_loader(loader_name=None, loader_conf_path=None, peft_path=None):
    #@todo: find loader fn & return loaded model
    pass