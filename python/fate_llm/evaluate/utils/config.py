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
import click
import yaml
import typing
from pathlib import Path
from ._io import set_logger, echo


DEFAULT_FATE_LLM_BASE_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
FATE_LLM_BASE_PATH = os.getenv("FATE_LLM_BASE_PATH") or DEFAULT_FATE_LLM_BASE_PATH

# DEFAULT_TASK_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tasks"))
DEFAULT_FATE_LLM_TASK_PATH = os.path.abspath(os.path.join(FATE_LLM_BASE_PATH, "tasks"))
FATE_LLM_TASK_PATH = os.getenv("FATE_LLM_TASK_PATH") or DEFAULT_FATE_LLM_TASK_PATH

_default_eval_config =  Path(FATE_LLM_BASE_PATH).resolve() / 'llm_eval_config.yaml'

template = """# args for evaluate
batch_size: 10
model_args:
    device: cuda
    dtype: auto
    trust_remote_code: true
num_fewshot: 0
"""


def create_eval_config(path: Path, override=False):
    if path.exists() and not override:
        raise FileExistsError(f"{path} exists")

    with path.open("w") as f:
        f.write(template)


def default_eval_config():
    if not _default_eval_config.exists():
        create_eval_config(_default_eval_config)
    return _default_eval_config


class Config(object):
    def __init__(self, config):
        self.update_conf(**config)

    def update_conf(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def load(path: typing.Union[str, Path], **kwargs):
        if isinstance(path, str):
            path = Path(path)
        config = {}
        if path is not None:
            with path.open("r") as f:
                config.update(yaml.safe_load(f))

        config.update(kwargs)
        return Config(config)

    @staticmethod
    def load_from_file(path: typing.Union[str, Path]):
        """
        Loads conf content from yaml file. Used to read in parameter configuration
        Parameters
        ----------
        path: str, path to conf file, should be absolute path

        Returns
        -------
        dict, parameter configuration in dictionary format

        """
        if isinstance(path, str):
            path = Path(path)
        config = {}
        if path is not None:
            file_type = path.suffix
            with path.open("r") as f:
                if file_type == ".yaml":
                    config.update(yaml.safe_load(f))
                else:
                    raise ValueError(f"Cannot load conf from file type {file_type}")
        return config


def parse_config(config):
    try:
        config_inst = Config.load(config)
    except Exception as e:
        raise RuntimeError(f"error parse config from {config}") from e
    return config_inst


def _set_namespace(namespace):
    Path(f"logs/{namespace}").mkdir(exist_ok=True, parents=True)
    set_logger(f"logs/{namespace}/exception.log")
    echo.set_file(click.open_file(f'logs/{namespace}/stdout', "a"))
