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
    def __init__(self, job_name: str, script_path: Path, conf_path: Path, pretrained_model_path: Path,
                 peft_path: Path, eval_conf_path: Path, loader: str, loader_conf_path: Path,
                 tasks: typing.List[str], include_path: Path):
        self.job_name = job_name
        self.script_path = script_path
        self.conf_path = conf_path
        self.pretrained_model_path = pretrained_model_path
        self.peft_path = peft_path
        self.loader = loader
        self.loader_conf_path = loader_conf_path
        self.eval_conf_path = eval_conf_path
        self.tasks = tasks
        self.include_path = include_path
        self.evaluate_only = self.script_path is None


class LlmPair(object):
    def __init__(
            self, pair_name: str, jobs: typing.List[LlmJob]
    ):
        self.pair_name = pair_name
        self.jobs = jobs


class LlmSuite(object):
    def __init__(
            self, pairs: typing.List[LlmPair], path: Path
    ):
        self.pairs = pairs
        self.path = path

    @staticmethod
    def load(path: Path):
        if isinstance(path, str):
            path = Path(path)
        with path.open("r") as f:
            testsuite_config = yaml.safe_load(f)

        pairs = []
        for pair_name, pair_configs in testsuite_config.items():
            jobs = []
            for job_name, job_configs in pair_configs.items():
                # with train
                script_path = job_configs.get("script", None)
                if script_path and not os.path.isabs(script_path):
                    script_path = path.parent.joinpath(script_path).resolve()

                conf_path = job_configs.get("conf", None)
                if conf_path and not os.path.isabs(conf_path):
                    conf_path = path.parent.joinpath(conf_path).resolve()

                # evaluate only
                pretrained_model_path = job_configs.get("pretrained", None)
                if pretrained_model_path and not os.path.isabs(pretrained_model_path):
                    # make path absolute, else keep original pretrained model name
                    if "yaml" in pretrained_model_path or "/" in pretrained_model_path:
                        pretrained_model_path = path.parent.joinpath(pretrained_model_path).resolve()

                peft_path = job_configs.get("peft", None)
                if peft_path and not os.path.isabs(peft_path):
                    peft_path = path.parent.joinpath(peft_path).resolve()

                eval_conf_path = job_configs.get("eval_conf", None)
                if eval_conf_path and not os.path.isabs(eval_conf_path):
                    eval_conf_path = path.parent.joinpath(eval_conf_path).resolve()

                loader = job_configs.get("loader", None)
                if job_configs.get("loader_conf"):
                    loader_conf_path = path.parent.joinpath(job_configs["loader_conf"]).resolve()
                else:
                    loader_conf_path = ""
                tasks = job_configs.get("tasks", [])
                include_path = job_configs.get("include_path", "")
                if include_path and not os.path.isabs(include_path):
                    include_path = path.parent.joinpath(job_configs["include_path"]).resolve()

                jobs.append(
                    LlmJob(
                        job_name=job_name, script_path=script_path, conf_path=conf_path,
                        pretrained_model_path=pretrained_model_path, peft_path=peft_path, eval_conf_path=eval_conf_path,
                        loader=loader, loader_conf_path=loader_conf_path, tasks=tasks, include_path=include_path
                    )
                )

            pairs.append(
                LlmPair(
                    pair_name=pair_name, jobs=jobs
                )
            )
        suite = LlmSuite(pairs=pairs, path=path)
        return suite

