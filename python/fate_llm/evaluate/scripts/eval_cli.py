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
from ..utils.config import default_eval_config
from ..utils.llm_evaluator import evaluate, init_tasks, aggregate_table
from ..utils.model_tools import load_by_loader
from ..utils._io import echo
from ..utils._parser import LlmSuite

@click.command('evaluate')
@click.option('-i', '--include', required=True, type=click.Path(exists=True),
              help='Path to model and metrics conf')
@click.option('-c', '--eval-config', type=click.Path(exists=True), help='Path to FATE Llm evaluation config. '
                                                                        'If not provided, use default config.')
@click.option('-o', '--result-output', type=click.Path(),
              help='Path to save evaluation results.')
# @click.argument('other_args', nargs=-1)
@LlmSharedOptions.get_shared_options(hidden=True)
@click.pass_context
def run_evaluate(ctx, include, eval_config, result_output, **kwargs):
    """
    Evaluate a pretrained model with specified parameters.
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    yes = ctx.obj["yes"]

    echo.echo(f"include: {include}", fg='red')
    try:
        # include = os.path.abspath(include)
        suite = LlmSuite.load(include)
    except Exception as e:
        raise ValueError(f"Invalid include path: {include}, please check. {e}")

    if not eval_config:
        eval_config = default_eval_config()

    eval_config_dict = {}
    with eval_config.open("r") as f:
        eval_config_dict.update(yaml.safe_load(f))

    if not yes and not click.confirm("running?"):
        return
    # init tasks
    init_tasks()
    run_suite_eval(suite, eval_config_dict, result_output)


def run_job_eval(job, eval_conf):
    job_eval_conf = {}
    job_eval_conf.update(eval_conf)

    if job.eval_conf_path:
        # job-level eval conf takes priority
        with open(job.eval_conf_path, 'r') as f:
            job_eval_conf.update(yaml.safe_load(f))
    # get loader
    if job.loader:
        if job.peft_path:
            model = load_by_loader(loader_name=job.loader,
                                   loader_conf_path=loader_conf_path,
                                   peft_path=job.peft_path)
        else:
            model = load_by_loader(loader_name=job.loader,
                                   loader_conf_path=loader_conf_path)
        result = evaluate(model=model, tasks=job.tasks, include_path=job.include_path, **job_eval_conf)
    else:
        # feed in pretrained & peft path
        job_eval_conf["model_args"]["pretrained"] = job.pretrained_model_path
        if job.peft_path:
            job_eval_conf["model_args"]["peft"] = job.peft_path
        result = evaluate(tasks=job.tasks, include_path=job.include_path, **job_eval_conf)
    return result


def run_suite_eval(suite, eval_conf, output_path=None):
    suite_results = dict()
    for pair in suite.pairs:
        job_results = dict()
        for job in pair.jobs:
            if not job.evaluate_only:
                # give warning that job will be skipped
                warnings.warn(f"Job {job.job_name} will be skipped since no pretrained model is provided")
                continue
            result = run_job_eval(job, eval_conf)
            job_results[job.job_name] = result
        suite_results[pair.pair_name] = job_results
    suite_writers = aggregate_table(suite_results)
    for pair_name, pair_writer in suite_writers.items():
        echo.sep_line()
        echo.echo(f"Pair: {pair_name}")
        echo.sep_line()
        echo.echo(pair_writer.dumps())
        echo.stdout_newline()

    if output_path:
        with open(output_path, 'w') as f:
            for pair_name, pair_writer in suite_writers.items():
                pair_writer.dumps(f)
