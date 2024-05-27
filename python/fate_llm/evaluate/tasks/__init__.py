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
    _task_source_url = ""
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

    @property
    def task_source_url(self):
        return self._task_source_url

    def download_from_source(self):
        raise NotImplementedError(f"Should not be called here.")


class Dolly(Task):
    _task_name = "dolly-15k"
    _task_dir = "dolly_15k"
    _task_conf_file = "default_dolly_15k.yaml"

    def download_from_source(self):
        try:
            from datasets import load_dataset
            data = load_dataset("databricks/databricks-dolly-15k", split="train")
            filename = os.path.join(self.task_scr_dir, "databricks-dolly-15k.jsonl")
            data.to_json(filename)
            return True
        except Exception as e:
            print(f"Failed to download data from source: {e}")
            return False


class AdvertiseGen(Task):
    _task_name = "advertise-gen"
    _task_dir = "advertise_gen"
    _task_conf_file = "default_advertise_gen.yaml"
    _task_source_url = ["https://cloud.tsinghua.edu.cn/seafhttp/files/3781289a-5a60-44b1-b5f1-a04364e3eb9d/AdvertiseGen.tar.gz",
                        "https://docs.google.com/uc?export=download&id=13_vf0xRTQsyneRKdD1bZIr93vBGOczrk"]

    def download_from_source(self):
        from ..utils.data_tools import download_data
        result = download_data(self.task_scr_dir, self.task_source_url[0])
        if not result:
            print(f"retry with address: {self.task_source_url[1]}")
            return download_data(self.task_scr_dir, self.task_source_url[1])
        return result


build_in_tasks = {"dolly-15k": Dolly(),
                  "advertise-gen": AdvertiseGen()}
