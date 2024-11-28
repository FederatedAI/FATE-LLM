####  offsite_tuning_fate_test_quick_start_zh

##### 1. 数据准备

​	准备QA数据集，本样例中使用sciq数据集。可以使用qa_dataset.py中提供的工具对sciq数据集进行标记，并保存标记结果。记住将save_path修改为自己的路径。数据处理脚本如下

~~~
# sciq数据处理脚本，稍微修改即可执行，执行需在fate环境下执行
import os
from fate_llm.dataset.qa_dataset 
import tokenize_qa_dataset
from transformers import AutoTokenizer
from fate_llm.dataset.qa_dataset import QaDataset

tokenizer_name_or_path = 'gpt2'  # 实际模型放置路径
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

if 'llama' in tokenizer_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, unk_token="<unk>",  bos_token="<s>", eos_token="</s>", add_eos_token=True)   
    tokenizer.pad_token = tokenizer.eos_token
else:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
if 'gpt2' in tokenizer_name_or_path:
    tokenizer.pad_token = tokenizer.eos_token


# bind data path to name & namespace
save_path = '/example/data/sciq'  #实际需要放置的路径，可自定义
rs = tokenize_qa_dataset('sciq', tokenizer, save_path, seq_max_len=600)

ds = QaDataset(tokenizer_name_or_path=tokenizer_name_or_path)
ds.load(save_path)

print(len(ds))  # train set length
print(ds[0]['input_ids'].__len__()) # first sample length
~~~

​	将数据集路径与名称和命名空间绑定。记住自己的数据集保存路径的路径。

~~~
flow table bind --namespace experiment --name sciq --path /example/data/sciq
~~~

##### 2. 配置文件修改

​	 test_offsite_tuning_llmsuite.yaml 仅需修改数据与模型的实际路径

~~~
data:
  - file: /example/data/sciq  #实际需要放置的路径
    table_name: sciq
    namespace: experiment
    role: guest_0
  - file: /example/data/sciq  #实际需要放置的路径
    table_name: sciq
    namespace: experiment
    role: host_0
offsite_tuning_lora_vs_zero_shot:
  offsite_tuning_lora:
    pretrained: "gpt2"   # 模型放置的实际路径
    script: "./offsite_tuning.py"
    conf: "./offsite_tuning_config.yaml"
    loader: "ot"
    model_weights_format: "{{fate_base}}/fate_flow/model/{{job_id}}/guest/{{party_id}}/{{model_task_name}}/0/output/output_model/model_directory/pytorch_model.bin"
    tasks:
      - "sciq"
  bloom_zero_shot:
    pretrained: "gpt2"   # 模型放置的实际路径
    tasks:
      - "sciq"
~~~

 	offsite_tuning_config.yaml 仅需修改下模型的实际路径

~~~
# offsite_tuning_config.yaml 开头片段
data:
  guest:
    namespace: experiment
    name: sciq
  host:
    namespace: experiment
    name: sciq

# 模型路径
pretrained_model_path: "gpt2" # 模型放置的实际路径
~~~

##### 3. 运行

~~~
# 环境准备
cd /fate/
source /fate/bin/init_env.sh

# 命令执行
fate_test llmsuite -i fate_llm/examples/offsite_tuning/ --yes
~~~

##### 4. 问题定位

​	任务运行失败，报错，会在当前执行命令的目录下生成一个logs目录，找到对应的任务，检查stdout，或者exception.log检查报错原因。

