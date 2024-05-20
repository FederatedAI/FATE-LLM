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

import yaml
import os


def local_fn_constructor(loader, node):
    return node


def local_fn_representer(dumper, data):
    return data


def dump_yaml(dict, path):
    yaml.add_representer(yaml.ScalarNode, local_fn_representer)
    with open(path, 'w') as f:
        yaml.dump(dict, f)

class Task:
    _task_name = ""
    _task_dir = ""
    _task_conf_file = ""
    script_dir = os.path.dirname(__file__)

    @property
    def task_name(self):
        return self._task_name

    @property
    def task_template(self):
        yaml.add_constructor("!function", local_fn_constructor)
        with open(os.path.abspath(os.path.join(self.script_dir, self._task_dir, self._task_conf_file)), "rb") as f:
            task_template = yaml.full_load(f)
        return task_template

    @property
    def task_scr_dir(self):
        return os.path.abspath(os.path.join(self.script_dir, self._task_dir))

    @property
    def task_conf_path(self):
        return os.path.abspath(os.path.join(self.script_dir, self._task_dir, self._task_conf_file))


class Dolly(Task):
    _task_name = "dolly-15k"
    _task_dir = "dolly_15k"
    _task_conf_file = "default_dolly_15k.yaml"


build_in_tasks = {"dolly-15k": Dolly()}
