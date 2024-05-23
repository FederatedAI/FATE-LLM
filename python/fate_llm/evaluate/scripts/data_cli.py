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
import copy
import click
import yaml
import warnings

from typing import Union
from ._options import LlmSharedOptions
from ..utils.llm_evaluator import download_task
from ..utils._io import echo

@click.command('download_data')
@click.option('-t', '--tasks', required=False, type=str, multiple=True, default=None,
              help='tasks whose data will be downloaded')
# @click.argument('other_args', nargs=-1)
@LlmSharedOptions.get_shared_options(hidden=True)
@click.pass_context
def download_data(ctx, tasks, **kwargs):
    """
    Evaluate a pretrained model with specified parameters.
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()

    if tasks is None or len(tasks) == 0:
        tasks = None
        echo.echo(f"No task is given, will download data for all built-in tasks.", fg='red')
    else:
        echo.echo(f"given tasks: {tasks}", fg='red')
    download_task(tasks)
