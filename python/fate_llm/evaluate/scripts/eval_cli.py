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

from typing import List, Tuple, Union, Optional
from transformers import AutoTokenizer
from lm_eval.models.huggingface import HFLM
import torch
import transformers
from fate_llm.evaluate.utils.model_tools import load_by_loader_OT, load_by_loader_PDSS
from fate_llm.evaluate.utils import llm_evaluator
from transformers import AutoModelForCausalLM

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
    # namespace = ctx.obj["namespace"]
    yes = ctx.obj["yes"]

    echo.echo(f"include: {include}", fg='red')
    try:
        # include = os.path.abspath(include)
        suite = LlmSuite.load(include)
    except Exception as e:
        raise ValueError(f"Invalid include path: {include}, please check. {e}")

    if not eval_config:
        eval_config = default_eval_config()

    if not os.path.exists(eval_config):
        eval_config = None

    if not yes and not click.confirm("running?"):
        return
    # init tasks
    init_tasks()
    # run_suite_eval(suite, eval_config_dict, result_output)
    run_suite_eval(suite, eval_config, result_output)

class MyCustomLM(HFLM):
    def __init__(self, pretrained: torch.nn.Module, 
                 model_path: str, 
                 tokenizer: Optional[Union[str, transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]] = None, 
                 rank=0, world_size=1, **kwargs):
        # 调用父类的构造函数
        super().__init__(pretrained=model_path, **kwargs)

        # 设置设备
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 将预训练的模型加载到指定设备
        self._model = pretrained.to(self._device)

        # 初始化分词器
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = tokenizer

        # 添加一个新的pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # 初始化其他所需的属性
        self._rank = rank
        self._world_size = world_size
        self.batch_size_per_gpu = 4

        # 从预训练模型中获取配置
        self._config = model_path  # 保证config的获取
        self._max_length = self._config.max_length if hasattr(self._config, 'max_length') else 1024
        self._logits_cache = None

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        # 实现滚动对数似然度
        return [(0.0, True) for _ in requests]

    def generate_until(self, requests: List[Tuple[str, str]]) -> List[str]:
        # 简单的返回生成的文本占位符，根据实际情况实现
        return ["Generated text" for _ in requests]

def run_job_eval(job, eval_conf):
    job_eval_conf = {}
    if isinstance(eval_conf, dict):
        job_eval_conf.update(eval_conf)
    elif eval_conf is not None and os.path.exists(eval_conf):
        with open(eval_conf, 'r') as f:
            job_eval_conf.update(yaml.safe_load(f))
    # echo.echo(f"Evaluating job: {job.job_name} with tasks: {job.tasks}")
    if job.eval_conf_path:
        # job-level eval conf takes priority
        with open(job.eval_conf_path, 'r') as f:
            job_eval_conf.update(yaml.safe_load(f))
    # get loader
    if job.loader:
        if job.peft_path:
            model = load_by_loader(loader_name=job.loader,
                                   # loader_conf_path=loader_conf_path,
                                   peft_path=job.peft_path)
   
            result = evaluate(model=model, tasks=job.tasks, include_path=job.include_path, **job_eval_conf)
        if job.model_weights_format:
            if job.loader == 'ot':
                loaded_model = load_by_loader_OT(trained_weights_path=job.model_weights_format,model_path=job.pretrained_model_path)
            if job.loader == 'pdss':
                loaded_model = load_by_loader_PDSS(trained_weights_path=job.model_weights_format,model_path=job.pretrained_model_path)
            
            # loaded_model  
            loaded_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))     
            # MyCustomLM
            gpt2_lm = MyCustomLM(pretrained=loaded_model,model_path=job.pretrained_model_path)     
            # llm_evaluator
            llm_evaluator.init_tasks()
            result = llm_evaluator.evaluate(model=gpt2_lm, tasks=job.tasks)


    else:
        # feed in pretrained & peft path
        job_eval_conf["model_args"]["pretrained"] = job.pretrained_model_path
        echo.echo(f"DEBUG: job_eval_conf = {job_eval_conf}")
        if job.peft_path:
            job_eval_conf["model_args"]["peft"] = job.peft_path
            echo.echo(f"DEBUG: job_eval_conf = {job_eval_conf}")
            echo.echo(f"DEBUG: job.include_path = {job.include_path}")
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
            echo.echo(f"Evaluating job: {job.job_name} with tasks: {job.tasks}")
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
