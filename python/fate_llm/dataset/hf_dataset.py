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
import os
from typing import Optional, Union, Sequence, Mapping, Dict

from datasets import load_dataset, Features, Split, DownloadConfig, DownloadMode, VerificationMode, Version, load_from_disk
from transformers import AutoTokenizer

from fate.ml.nn.dataset.base import Dataset

# avoid tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class HuggingfaceDataset(Dataset):
    """
    A dataset class for huggingface datasets
    """

    def __init__(
            self,
            name: Optional[str] = None,
            data_dir: Optional[str] = None,
            data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
            split: Optional[Union[str, Split]] = None,
            cache_dir: Optional[str] = None,
            features: Optional[Features] = None,
            download_config: Optional[DownloadConfig] = None,
            download_mode: Optional[Union[DownloadMode, str]] = None,
            verification_mode: Optional[Union[VerificationMode, str]] = None,
            ignore_verifications="deprecated",
            keep_in_memory: Optional[bool] = None,
            save_infos: bool = False,
            revision: Optional[Union[str, Version]] = None,
            token: Optional[Union[bool, str]] = None,
            use_auth_token="deprecated",
            task="deprecated",
            streaming: bool = False,
            num_proc: Optional[int] = None,
            storage_options: Optional[Dict] = None,
            trust_remote_code: bool = None,
            tokenizer_params: Optional[Dict] = None,
            tokenizer_apply_params: Optional[Dict] = None,
            load_from_disk: Optional[bool] = False,
            inplace_load: Optional[bool] = True,
            data_split_key: Optional[str] = None,
            **config_kwargs,
    ):
        self.name = name
        self.data_dir = data_dir
        self.data_files = data_files
        self.split = split
        self.cache_dir = cache_dir
        self.features = features
        self.download_config = download_config
        self.download_mode = download_mode
        self.verification_mode = verification_mode
        self.ignore_verifications = ignore_verifications
        self.keep_in_memory = keep_in_memory
        self.save_infos = save_infos
        self.revision = revision
        self.token = token
        self.use_auth_token = use_auth_token
        self.task = task
        self.streaming = streaming
        self.num_proc = num_proc
        self.storage_options = storage_options
        self.trust_remote_code = trust_remote_code
        self.tokenizer_params = tokenizer_params
        self.tokenizer_apply_params = tokenizer_apply_params
        self.config_kwargs = config_kwargs
        self.load_from_disk = load_from_disk
        self.inplace_load = inplace_load
        self.data_split_key = data_split_key
        self.ds = None

        super(HuggingfaceDataset, self).__init__()

    def load(self, file_path):
        if not self.load_from_disk:
            ds = load_dataset(path=file_path, name=self.name, data_dir=self.data_dir, data_files=self.data_files,
                                split=self.split, cache_dir=self.cache_dir, features=self.features,
                                download_config=self.download_config, download_mode=self.download_mode,
                                verification_mode=self.verification_mode, ignore_verifications=self.ignore_verifications,
                                keep_in_memory=self.keep_in_memory, save_infos=self.save_infos, revision=self.revision,
                                token=self.token, use_auth_token=self.use_auth_token, task=self.task,
                                streaming=self.streaming, num_proc=self.num_proc, storage_options=self.storage_options,
                                trust_remote_code=self.trust_remote_code, **self.config_kwargs)
        else:
            ds = load_from_disk(file_path)

        if self.data_split_key is not None:
            ds = ds[self.data_split_key]

        if self.inplace_load:
            self.ds = ds
        else:
            return ds

    def __getitem__(self, idx):
        if self.ds is None:
            raise ValueError('Dataset is not loaded')
        return self.ds[idx]

    def __len__(self):
        if self.ds is None:
            raise ValueError('Dataset is not loaded')
        return len(self.ds)


class Dolly15K(HuggingfaceDataset):
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"
    RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
    DEFAULT_SEED = 42
    INTRO_BLURB = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    )
    PROMPT_NO_INPUT_FORMAT = """{intro}
{instruction_key}
{instruction}

{response_key}
{response}

{end_key}""".format(
        intro=INTRO_BLURB,
        instruction_key=INSTRUCTION_KEY,
        instruction="{instruction}",
        response_key=RESPONSE_KEY,
        response="{response}",
        end_key=END_KEY,
    )

    # This is a training prompt that contains an input string that serves as context for the instruction.  For example,
    # the input might be a passage from Wikipedia and the intruction is to extract some information from it.
    PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input_key}
{input}

{response_key}
{response}

{end_key}""".format(
        intro=INTRO_BLURB,
        instruction_key=INSTRUCTION_KEY,
        instruction="{instruction}",
        input_key=INPUT_KEY,
        input="{input}",
        response_key=RESPONSE_KEY,
        response="{response}",
        end_key=END_KEY,
    )

    def __init__(self, *args, **kwargs):
        super(Dolly15K, self).__init__(*args, **kwargs)
        self.inplace_load = False

    def load(self, file_path):
        dataset = super().load(file_path)
        return self._post_process(dataset)

    def _post_process(self, dataset):

        def _add_text(rec):
            instruction = rec["instruction"]
            response = rec["response"]
            context = rec.get("context")

            if not instruction:
                raise ValueError(f"Expected an instruction in: {rec}")

            if not response:
                raise ValueError(f"Expected a response in: {rec}")

            # For some instructions there is an input that goes along with the instruction, providing context for the
            # instruction.  For example, the input might be a passage from Wikipedia and the instruction says to extract
            # some piece of information from it.  The response is that information to extract.  In other cases there is
            # no input.  For example, the instruction might be open QA such as asking what year some historic figure was
            # born.
            if context:
                rec["text"] = self.PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response,
                                                                   input=context)
            else:
                rec["text"] = self.PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
            return rec

        dataset = dataset.map(_add_text)

        tokenizer = AutoTokenizer.from_pretrained(**self.tokenizer_params)

        def tokenize_function(examples):
            return tokenizer(examples["text"], **self.tokenizer_apply_params)

        dataset = dataset.map(tokenize_function, batched=True)
        return dataset
