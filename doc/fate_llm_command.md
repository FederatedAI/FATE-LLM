
## FATE-Llm Command

FATE Llm provides built-in tasks for comparing evaluation results of different llm models. 
Altenatively, user may provide arbitrary tasks for evaluation.

### command options

```bash
fate_llm --help
```

#### evaluate:


1. in:

   ```bash
   fate_llm evaluate -i <path1 contains *.yaml>
   ```

   will run llm testsuite in
   *path1*

2. eval-config:

    ```bash
    fate_llm evaluate -i <path1 contains *.yaml> -c <path2>
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


### FATE-Llm Eval job configuration

Configuration of jobs should be specified in a yaml file. 

A FATE-Llm testsuite includes the following elements:

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
     hetero_nn_sshe_binary_0:
      bloom_zero_shot:
        pretrained: "bloom-560m"
        tasks:
          - "dolly-15k"
  ```
