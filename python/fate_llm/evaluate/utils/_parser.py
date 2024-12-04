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
import yaml
import typing
from pathlib import Path

class LlmJob(object):
    def __init__(self, job_name: str, script_path: Path=None, conf_path: Path=None, model_task_name: str=None,
                 pretrained_model_path: Path=None, peft_path: Path=None, model_weights: Path=None,
                 eval_conf_path: Path=None, loader: str=None, loader_conf_path: Path=None,
                 tasks: typing.List[str]=None, include_path: Path=None, peft_path_format: str=None, model_weights_format: str=None,requires_untar: str=None):
        self.job_name = job_name
        self.script_path = script_path
        self.conf_path = conf_path
        self.model_task_name = model_task_name
        self.pretrained_model_path = pretrained_model_path
        self.peft_path = peft_path
        self.loader = loader
        self.loader_conf_path = loader_conf_path
        self.eval_conf_path = eval_conf_path
        self.tasks = tasks
        self.include_path = include_path
        self.evaluate_only = self.script_path is None
        self.peft_path_format = peft_path_format
        self.model_weights_format = model_weights_format
        self.model_weights = model_weights
        self.requires_untar = requires_untar

class LlmPair(object):
    def __init__(
            self, pair_name: str, jobs: typing.List[LlmJob]
    ):
        self.pair_name = pair_name
        self.jobs = jobs


class LlmSuite(object):
    def __init__(
            self, pairs: typing.List[LlmPair], path: Path, dataset=None
    ):
        self.pairs = pairs
        self.path = path
        self.dataset = dataset
        self.suite_name = Path(self.path).stem
        self._final_status = {}

    @staticmethod
    def load(path: Path):
        if isinstance(path, str):
            path = Path(path)
        with path.open("r") as f:
            testsuite_config = yaml.safe_load(f)

        pairs = []
        for pair_name, pair_configs in testsuite_config.items():
            if pair_name == "data":
                continue
            jobs = []
            for job_name, job_configs in pair_configs.items():
                # with train
                script_path = job_configs.get("script", None)
                if script_path and not os.path.isabs(script_path):
                    script_path = path.parent.joinpath(script_path).resolve()

                conf_path = job_configs.get("conf", None)
                if conf_path and not os.path.isabs(conf_path):
                    conf_path = path.parent.joinpath(conf_path).resolve()

                model_task_name = job_configs.get("model_task_name", None)

                # evaluate only
                pretrained_model_path = job_configs.get("pretrained", None)
                if pretrained_model_path and not os.path.isabs(pretrained_model_path):
                    # make path absolute, else keep original pretrained model name
                    if "yaml" in pretrained_model_path or "/" in pretrained_model_path:
                        pretrained_model_path = path.parent.joinpath(pretrained_model_path).resolve()

                peft_path = job_configs.get("peft", None)
                if peft_path and not os.path.isabs(peft_path):
                    peft_path = path.parent.joinpath(peft_path).resolve()

                model_weights = job_configs.get("weights", None)
                if model_weights and not os.path.isabs(model_weights):
                    model_weights = path.parent.joinpath(model_weights).resolve()

                requires_untar = job_configs.get("untar", None)
                if requires_untar and not os.path.isabs(requires_untar):
                    requires_untar = path.parent.joinpath(requires_untar).resolve()
                    
                eval_conf_path = job_configs.get("eval_conf", None)
                if eval_conf_path and not os.path.isabs(eval_conf_path):
                    eval_conf_path = path.parent.joinpath(eval_conf_path).resolve()

                loader = job_configs.get("loader", None)
                # loader_conf
                loader_conf = job_configs.get("loader_conf", None)
                if isinstance(loader_conf, dict):
                    loader_conf_data = loader_conf
                    loader_conf_path = None  
                elif isinstance(loader_conf, str):
                    loader_conf_path = path.parent.joinpath(loader_conf).resolve()
                    loader_conf_data = None  
                else:
                    loader_conf_data = None
                    loader_conf_path = None

                tasks = job_configs.get("tasks", [])
                include_path = job_configs.get("include_path", "")
                if include_path and not os.path.isabs(include_path):
                    include_path = path.parent.joinpath(job_configs["include_path"]).resolve()

                peft_path_format = job_configs.get("peft_path_format",None)

                model_weights_format = job_configs.get("model_weights_format",None)

                requires_untar = job_configs.get("requires_untar",None)
                
                jobs.append(
                    LlmJob(
                        job_name=job_name, script_path=script_path, conf_path=conf_path,
                        model_task_name=model_task_name,
                        pretrained_model_path=pretrained_model_path, peft_path=peft_path, eval_conf_path=eval_conf_path,
                        loader=loader, loader_conf_path=loader_conf_data, tasks=tasks, include_path=include_path,
                        peft_path_format=peft_path_format,
                        model_weights_format=model_weights_format,
                        model_weights=model_weights,
                        requires_untar=requires_untar
                    )
                )

            pairs.append(
                LlmPair(
                    pair_name=pair_name, jobs=jobs
                )
            )
        suite = LlmSuite(pairs=pairs, path=path)
        return suite
    
    def update_status(
            self, pair_name, job_name, job_id=None, status=None, exception_id=None, time_elapsed=None, event=None
    ):
        for k, v in locals().items():
            if k != "job_name" and k != "pair_name" and v is not None:
                if self._final_status.get(f"{pair_name}-{job_name}"):
                    setattr(self._final_status[f"{pair_name}-{job_name}"], k, v)

    def get_final_status(self):
        return self._final_status

    