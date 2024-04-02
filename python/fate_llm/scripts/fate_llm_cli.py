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

from typing import Union
from .eval_cli import run_evaluate
from .config_cli import eval_config_group
from ._options import LlmSharedOptions


commands = {
    "evaluate": run_evaluate,
    "config": eval_config_group
}


class FATELlmCLI(click.MultiCommand):

    def list_commands(self, ctx):
        return list(commands)

    def get_command(self, ctx, name):
        if name not in commands and name in commands_alias:
            name = commands_alias[name]
        if name not in commands:
            ctx.fail("No such command '{}'.".format(name))
        return commands[name]

@click.command(cls=FATELlmCLI, help="A collection of tools to run FATE Llm Evaluation.",
               context_settings=dict(help_option_names=["-h", "--help"]))
@LlmSharedOptions.get_shared_options()
@click.pass_context
def fate_llm_cli(ctx, **kwargs):
    ctx.ensure_object(LlmSharedOptions)
    ctx.obj.update(**kwargs)


if __name__ == '__main__':
    fate_llm_cli(obj=LlmSharedOptions())