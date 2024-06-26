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

# this file is used to evaluate the model on fate-llm built-in tasks and user-given tasks

import os
import tempfile
import yaml
import shutil
import warnings
from pytablewriter import MarkdownTableWriter

import lm_eval
from lm_eval.utils import load_yaml_config
from ..tasks import build_in_tasks, dump_yaml
from .config import FATE_LLM_BASE_PATH, FATE_LLM_TASK_PATH


def evaluate(tasks, model="hf", model_args=None, include_path=None, task_manager=None, show_result=False, **kwargs):
    """
    Evaluate the model on given tasks. Simplified uses for built-in tasks.
    Parameters
    ----------
    tasks: str or List[str], task name(s)
    model: str or model object, model to be evaluated,
        select from lm_eval supported types: {"hf-auto", "hf", "huggingface", "vllm"}
    model_args: model args, str or dict
    include_path: task path for tasks not in built-in tasks
    task_manager: lm_eval.TakManger object
    kwargs

    Returns
    -------

    """
    if task_manager:
        if not isinstance(task_manager, lm_eval.tasks.TaskManager):
            raise ValueError(f"'task_manager' must be of TaskManager type.")
    elif include_path:
        task_manager = lm_eval.tasks.TaskManager(include_path=str(include_path))
    else:
        task_manager = lm_eval.tasks.TaskManager(include_path=str(FATE_LLM_TASK_PATH))
    task_names = []
    if isinstance(tasks, str):
        task_names.append(tasks)

    elif isinstance(tasks, list):
        for task in tasks:
            if isinstance(task, str):
                task_names.append(task)
            else:
                raise ValueError(f"tasks: {task}  of type {type(task)} not valid, please check.")

    else:
        raise ValueError(f"tasks: {tasks}  of type {type(tasks)} not valid, please check.")

    results = lm_eval.simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=task_names,
        task_manager=task_manager,
        **kwargs
    )
    if show_result:
        result_table = lm_eval.utils.make_table(results)
        print(result_table)
    return results


def aggregate_table(results):
    """
    adapted from lm_eval.utils.make_table:
    https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.2/lm_eval/utils.py    Aggregate results from different models with same tasks
    Parameters
    ----------
    results: dict, results from different models

    Returns
    -------

    """

    suite_writers = dict()
    for pair_name, pair_results in results.items():
        # job_count = len(pair_results)
        all_jobs = list(pair_results.keys())

        md_writer = MarkdownTableWriter()

        values = []
        task_results = dict()
        # print(f"pair results: {pair_results}")
        for job_name, result_dict in pair_results.items():
            if "results" in result_dict and result_dict["results"]:
                column = "results"
            else:
                column = "groups"
            for k, dic in result_dict[column].items():

                if "alias" in dic:
                    # task alias
                    k = dic.pop("alias")

                for (mf), v in dic.items():
                    m, _, f = mf.partition(",")
                    if m.endswith("_stderr"):
                        continue

                    if m + "_stderr" + "," + f in dic:
                        se = dic[m + "_stderr" + "," + f]
                        if se != "N/A":
                            se = "%.4f" % se
                        v = "%.4f ± %s" % (v, se)
                    else:
                        v = "%.4f" % v
                    task_results.setdefault(k, {}).setdefault(job_name, {})[m] = v

        # job names as columns
        # print(f"task results: {task_results}")
        for task_name, task_result in task_results.items():
            metrics = {inner_key for inner_dict in task_result.values() for inner_key, value in inner_dict.items()}
            for metric in metrics:
                row = [f"{task_name}({metric})"]
                for job_name in all_jobs:
                    if job_name in task_result:
                        row.append(task_result[job_name].get(metric, "N/A"))
                    else:
                        row.append("N/A")
                values.append(row)

        all_headers = ["Task"] + list(pair_results.keys())
        md_writer.headers = all_headers
        md_writer.value_matrix = values
        suite_writers[pair_name] = md_writer
    return suite_writers


def get_task_template(task):
    if not isinstance(task, str) or task not in build_in_tasks:
        raise ValueError(f"{task} not found in build in task, please check input.")
    result = build_in_tasks.get(task).task_template

    return result


def export_config(config, task, export_dir=None, export_sub_dir=None):
    scr_dir = build_in_tasks.get(task).task_scr_dir
    if export_dir is None:
        export_dir = os.path.dirname(scr_dir)

    if export_sub_dir is None:
        temp_dir = tempfile.mkdtemp()
        # make sure the relative path in new file will work
        full_export_dir = os.path.join(export_dir, os.path.basename(temp_dir))
        os.rename(temp_dir, full_export_dir)
    else:
        full_export_dir = os.path.join(export_dir, export_sub_dir)
    copy_directory_to_dst(scr_dir, full_export_dir, build_in_tasks.get(task).task_conf_path, config)

    return full_export_dir


def copy_directory_to_dst(src_dir, dst_dir, target_conf_file, new_conf: dict):
    """parent_dir = os.path.dirname(src_dir)

    temp_dir = tempfile.mkdtemp()
    # make sure the relative path in new file will work
    temp_dir_in_parent = os.path.join(parent_dir, os.path.basename(temp_dir))
    os.rename(temp_dir, temp_dir_in_parent)"""

    for item in os.listdir(src_dir):

        src_item = os.path.join(src_dir, item)
        dst_item = os.path.join(dst_dir, item)
        if os.path.isdir(src_item):
            shutil.copytree(src_item, dst_item)
        else:
            if item == target_conf_file:
                # write new conf file
                dump_yaml(new_conf, dst_item)
            else:
                shutil.copy2(src_item, dst_item)
            # shutil.copy2(src_item, dst_item)


def contains_subdirectory(path, subdirectories):
    base_name = os.path.basename(path)
    if base_name in subdirectories:
        return True

    for root, dirs, files in os.walk(path):
        for d in dirs:
            if d in subdirectories:
                return True

    return False

def delete_config(target_dir, force=False):
    if not force:
        # check if target dir in any of the build in tasks, only rm dir for build in tasks if force=True
        all_build_in_dir = {task.task_scr_dir for task in build_in_tasks.values()}
        if contains_subdirectory(target_dir, all_build_in_dir):
            warnings.warn(f"Built-in task(s) found in given target directory, please check input or set `force`=True.")
            return
        shutil.rmtree(target_dir)


def set_environ_fate_llm_base(path):
    if path:
        os.environ["FATE_LLM_BASE_PATH"] = path


def set_environ_fate_llm_task_base(path):
    if path:
        os.environ["FATE_LLM_TASK_PATH"] = path


def init_tasks(root_path=None):
    """

    Parameters
    ----------
    root_path: str, default None, root path for all local datasets in built-in tasks, {$root_path}/{$data_files};
    if not provided, current file path will be used to generate root

    Returns
    -------

    """
    for task in build_in_tasks.values():
        conf_path = task.task_conf_path
        parent_path = os.path.dirname(conf_path)
        task_template = task.task_template
        data_args = task_template.get("dataset_kwargs")
        if data_args:
            data_files = data_args.get("data_files")
            if isinstance(data_files, str):
                if data_files.endswith("jsonl") or data_files.endswith("json"):
                    if root_path:
                        parent_dir = os.path.basename(parent_path)
                        new_conf_path = os.path.join(root_path, parent_dir, os.path.basename(conf_path))
                    else:
                        new_conf_path = os.path.join(parent_path, data_files)
                    task_template["dataset_kwargs"]["data_files"] = new_conf_path
            elif isinstance(data_files, dict):
                for k, v in data_files.items():
                    if root_path:
                        parent_dir = os.path.basename(parent_path)
                        new_conf_path = os.path.join(root_path, parent_dir, os.path.basename(conf_path))
                    else:
                        new_conf_path = os.path.join(parent_path, v)
                    task_template["dataset_kwargs"]["data_files"][k] = new_conf_path

        try:
            dump_yaml(task_template, conf_path)
        except FileNotFoundError:
            raise ValueError(f"Cannot find task config {conf_path}, please check.")
        except Exception:
            raise ValueError(f"Initialization failed.")

def download_task(tasks=None):
    if tasks is None:
        tasks = list(build_in_tasks.keys())
    i = 1
    if isinstance(tasks, str):
        tasks = [tasks]
    n = len(tasks)
    for task in tasks:
        task_obj = build_in_tasks.get(task)
        if task_obj is None:
            print(f"Task {task} not found in built-in tasks, please check.")
            continue
        result = task_obj.download_from_source()
        if result:
            print(f"Finish downloading {i}/{n} th task data: {task}, saved to {task_obj.task_scr_dir}.\n")
        else:
            print(f"Failed to download {i}/{n} th task data to {task_obj.task_scr_dir}.\n")
        i += 1
