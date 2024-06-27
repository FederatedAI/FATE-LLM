## FATE-LLM Python SDK

FATE-LLM Python SDK provides simple API for evaluating large language models.
Built on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/), our evaluation tool may be used on pre-trained models from Huggingface, local-built models, as well as FATE-LLM models. 
[Built-in datasets](#built-in-tasks) currently include Dolly-15k and Advertise Generation.
Below shows how to evaluate given llm model in few lines. For quick single-model evaluation, below steps should suffice, however, if comparative evaluation among multiple models is desired, CLI is recommended.

```python
    from lm_eval.models.huggingface import HFLM
    from fate_llm.evaluate.utils import llm_evaluator

    # download data for built-in tasks if running fate-llm evaluation for the first time 
    # alternatively, use CLI `fate-llm data download` to download data
    llm_evaluator.download_task("dolly-15k")
    # set paths of built-in tasks
    llm_evaluator.init_tasks()
    # load model
    bloom_lm = HFLM(pretrained='bloom-560')
    # if loading local model, specify peft storage location
    # gpt2_lm = HFLM(pretrained='bloom-560m', peft_path_format="path/to/peft")
    # run evaluation
    llm_evaluator.evaluate(model=bloom_lm, tasks="dolly-15k", show_result=True)
```

When network allows, or if already cached, tasks from lm-evaluation may be provided for evaluation in similar style.

```python
    from lm_eval.models.huggingface import HFLM
    from fate_llm.evaluate.utils import llm_evaluator
    # load model
    bloom_lm = HFLM(pretrained='bloom-560')
    # if loading local model, specify peft storage location
    # bloom_lm = HFLM(pretrained='bloom-560m', peft_path_format="path/to/peft")
    # run evaluation
    llm_evaluator.evaluate(model=gpt2_lm, tasks="ceval", show_result=True)
```

## FATE-LLM Command Line Interface

FATE LLM provides built-in tasks for comparing evaluation results of different llm models. 
Alternatively, user may provide arbitrary tasks for evaluation.

### install

```bash
cd {path_to_fate_llm}/python
pip install -e .
```

### command options

```bash
fate_llm --help
```

#### evaluate:


1. in:

   ```bash
   fate_llm evaluate -i <path1 to *.yaml>
   ```

   will run llm at
   *path1*

2. eval-config:

    ```bash
    fate_llm evaluate -i <path1 to *.yaml> -c <path2>
    ```
  

   will run llm testsuites in *path1* with evaluation configuration set to *path2*

3. result-output:

    ```bash
    fate_llm evaluate -i <path1 contains *.yaml> -o <path2>
    ```

    will run llm testsuites in *path1* with evaluation result output stored in *path2*

### config

```bash
fate_llm config --help
```

1. new:
    ```bash
    fate_llm config new
    ```

    will create a new evaluation configuration file in current directory

2. show:

    ```bash
    fate_llm config show
    ```

    will show current evaluation configuration 

3. edit:

    ```bash
    fate_llm config edit 
    ```

    will edit evaluation configuration

### data
    
    ```bash
    fate_llm data --help
    ```
1. download:

    ```bash
    fate_llm data download -t <task1> -t <task2> ...
    ```

    will download corresponding data for given tasks 


### FATE-LLM Eval job configuration

Configuration of jobs should be specified in a yaml file. 

A FATE-LLM testsuite includes the following elements:

- job group: each group includes arbitrary number of jobs with paths
  to corresponding script and configuration

    - job: name of evaluation job to be run, must be unique within each group
      list
        - pretrained: path to pretrained model, should be either mmodel name from Hugginface or relative path to
          testsuite
        - peft: path to peft file, should be relative to testsuite, 
          optional
        - tasks: list of tasks to be evaluated, optional for jobs skipping evaluation
        - include_path: should be specified if tasks are user-defined
        - eval_conf: path to evaluation configuration file, should be
          relative to testsuite; if not provided, will use default conf

      ```yaml
          bloom_lora:
            pretrained: "bloom-560m"
            peft_path_format: "{{fate_base}}/fate_flow/model/{{job_id}}/guest/{{party_id}}/{{model_task_name}}/0/output/output_model/model_directory"
            tasks:
              - "dolly-15k"

      ```

- llm suite

  ```yaml
     bloom_suite:
      bloom_zero_shot:
        pretrained: "bloom-560m"
        tasks:
          - "dolly-15k"
  ```
  
## Built-in Tasks

Currently, we include the following tasks in FATE-LLM Evaluate:

| Task Name |     Alias     | Task Type  | Metric  |                                  source                                   |
|:---------:|:-------------:|:----------:|:-------:|:-------------------------------------------------------------------------:|
| Dolly-15k |   dolly-15k   | generation | rouge-L |  [link](https://huggingface.co/datasets/databricks/databricks-dolly-15k)  |
|   ADGEN   | advertise-gen | generation | rouge-L |                                 [link](https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/README_en.md#instructions)                                  |

Use corresponding alias to reference tasks in the system.
