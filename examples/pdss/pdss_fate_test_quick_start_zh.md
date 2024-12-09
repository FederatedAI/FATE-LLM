####  pdss_fate_test_quick_start_zh

##### 1. 数据准备

​	准备QA数据集，本样例中使用arc_easy数据集。可以使用qa_dataset.py中提供的工具对arc_easy数据集进行分词，并保存分词结果。记住将save_path修改为自己的路径。数据处理脚本如下

~~~
# arc_easy数据处理脚本，稍微修改即可执行，执行需在fate环境下执行
from datasets
import load_dataset
dataset = load_dataset("arc_easy")
dataset.save_to_disk('path_to_save/arc_easy')

from fate_llm.dataset.pdss_dataset import PrefixDataset

pds = PrefixDataset(
        tokenizer_path='/data/models/Qwen-14B/',
        predict_input_template="""Predict:
Question:{{question}}
Choices:{{choices.text}}
Answer:
    """,
        predict_output_template="""{{choices.text[choices.label.index(answerKey)]}}<end>""",
        rationale_input_template="""Explain:
Question:{{question}}
Choices:{{choices.text}}
Rationale:
    """,
        rationale_output_template="""{{infer_result}}<end>""",
        max_input_length=128,
        max_target_length=128,
        split_key='train'
    )

pds.load('path_to_save/arc_easy')
~~~

​	将数据集路径与名称和命名空间绑定。记住自己的数据集保存路径的路径。

~~~
flow table bind --namespace experiment --name arc_easy --path path_to_save/arc_easy
~~~

##### 2. 配置文件修改

​	 test_pdss_llmsuite.yaml 仅需修改数据与模型的实际路径

~~~
data:
  - file: path_to_save/arc_easy  #实际需要放置的路径
    table_name: arc_easy
    namespace: experiment
    role: guest_0
  - file: path_to_save/arc_easy  #实际需要放置的路径
    table_name: arc_easy
    namespace: experiment
    role: host_0
pdss_lora_vs_zero_shot:
  pdss_lora:
    pretrained: "Qwen"   # 模型放置的实际路径
    script: "./pdss.py"
    conf: "./pdss_config.yaml"
    loader: "pdss"
    model_weights_format: "{{fate_base}}/fate_flow/model/{{job_id}}/guest/{{party_id}}/{{model_task_name}}/0/output/output_model/model_directory/pytorch_model.bin"
    tasks:
      - "arc_easy"
  bloom_zero_shot:
    pretrained: "Qwen-14B"   # 模型放置的实际路径
    tasks:
      - "arc_easy"
~~~

 	pdss_config.yaml 仅需修改下模型的实际路径

~~~
# pdss_config.yaml 开头片段
data:
  guest:
    namespace: experiment
    name: arc_easy
  host:
    namespace: experiment
    name: arc_easy

# 模型配置
model:
  pretrained_model_name_or_path: "Qwen-14B" # 模型放置的实际路径

data_collator:
  tokenizer_name_or_path: "Qwen-14B" # 模型的tokenizer放置的实际路径

inference:
  client:
    api_url: "http://127.0.0.1:9999/v1"
    api_model_name: "Qwen-14B" # 模型放置的实际路径
    api_key: "demo"
    inferdpt_kit_path: "/examples/pdss" # inferdpt_kit_path数据放置的实际路径际路径
  server:
    api_url: "http://127.0.0.1:9999/v1"
    api_model_name: "Qwen-14B" # 模型放置的实际路径
    api_key: "demo"

~~~

##### 3. 运行

~~~
# 创建 vllm 环境
python -m venv vllm_venv
source vllm_venv/bin/activate
pip install vllm==0.4.3
pip install numpy==1.26.4 # numpy >= 2.0.0 will raise error, so reinstall numpy<2.0.0

# Qwen1.5-0.5为本地llm模型保存路径
export CUDA_VISIBLE_DEVICES=1,2
nohup python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port 9999 --model Qwen1.5-0.5 --dtype=half --enforce-eager --api-key demo --device cuda -tp 2 &
~~~
~~~
# 环境准备
cd $fate_base
source fate/bin/init_env.sh

# 命令执行
fate_test llmsuite -i fate_llm/examples/pdss/ --yes
~~~

##### 4. 问题定位

​	任务运行失败，报错，会在当前执行命令的目录下生成一个logs目录，找到对应的任务，检查stdout，或者exception.log检查报错原因。

