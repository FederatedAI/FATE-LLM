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
from sentence_transformers import SentenceTransformer
from typing import Any, Optional, Dict, Union


class SentenceTransformerModel(object):
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        prompts: Optional[Dict[str, str]] = None,
        default_prompt_name: Optional[str] = None,
        cache_folder: Optional[str] = None,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        token: Optional[Union[bool, str]] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        truncate_dim: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.prompts = prompts
        self.default_prompt_name = default_prompt_name
        self.cache_folder = cache_folder
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.local_files_only = local_files_only
        self.token = token
        self.use_auth_token = use_auth_token
        self.truncate_dim = truncate_dim
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.config_kwargs = config_kwargs

    def load(self):
        model = SentenceTransformer(
            model_name_or_path=self.model_name_or_path,
            device=self.device,
            prompts=self.prompts,
            default_prompt_name=self.default_prompt_name,
            cache_folder=self.cache_folder,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            local_files_only=self.local_files_only,
            token=self.token,
            use_auth_token=self.use_auth_token,
            truncate_dim=self.truncate_dim,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
            config_kwargs=self.config_kwargs
        )

        return model
