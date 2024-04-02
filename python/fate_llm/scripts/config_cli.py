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


import click
import yaml
from pathlib import Path
from ..utils.config import create_eval_config, default_eval_config
from ._options import LlmSharedOptions
from ..utils._io import echo

@click.group("eval_config", help="fate_llm evaluate config")
def eval_config_group():
    """
    eval_config fate_llm
    """
    pass


@eval_config_group.command(name="new")
def _new():
    """
    create new fate_llm eval config from template
    """
    create_eval_config(Path("llm_eval_config.yaml"))
    click.echo(f"create eval_config file: llm_eval_config.yaml")


@eval_config_group.command(name="edit")
@LlmSharedOptions.get_shared_options(hidden=True)
@click.pass_context
def _edit(ctx, **kwargs):
    """
    edit fate_llm eval_config file
    """
    ctx.obj.update(**kwargs)
    eval_config = ctx.obj.get("eval_config")
    print(f"eval_config: {eval_config}")
    click.edit(filename=eval_config)


@eval_config_group.command(name="show")
def _show():
    """
    show fate_test default eval_config path
    """
    click.echo(f"default eval_config path is {default_eval_config()}")
