####  fedmkt_fate_test_quick_start_zh

##### 1. 数据准备

​数据集：ARC-Challenge
ARC-Challenge 是一个包含 7,787 个真正的小学水平多项选择科学问题的数据集，旨在鼓励高级问答研究。

您可以参考以下链接了解有关 [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc) 的更多详细信息

从 huggingface 下载 ARC-Challenge 数据集并将其分成五个部分，部分“通用”用于公共数据集，其他部分用于 slms（opt2、gpt2、llama、opt）的训练。

~~~
# ARC-Challenge数据处理脚本，稍微修改即可执行，执行需在fate环境下执行
import datasets

data = datasets.load_dataset("ai2_arc", "ARC-Challenge", download_mode="force_redownload", ignore_verifications=True)
train_data = data.pop("train")

seed=123
n = train_data.shape[0]
client_num = 4
process_data_output_dir = "" # processed data saved directory should be specified, it will be used in later.

client_data_num = n // (client_num + 1)

for i in range(client_num):
    splits = train_data.train_test_split(train_size=client_data_num, shuffle=True, seed=seed)
    client_name = f"client_{i}"
    data[client_name] = splits["train"]
    train_data = splits["test"]

if train_data.shape[0] == client_data_num:
    data["common"] = train_data
else:
    data["common"] = train_data.train_test_split(
        train_size=client_data_num, shuffle=True, seed=args.seed
    )["train"]

data.save_to_disk(process_data_output_dir)
~~~

​	将数据集路径与名称和命名空间绑定。记住自己的数据集保存路径的路径。

~~~
flow table bind --namespace experiment --name arc_challenge --path path_to_save/arc_challenge
~~~

##### 2. 配置文件修改

​	 test_fedmkt_llmsuite.yaml 仅需修改数据与模型的实际路径

~~~
data:
  - file: path_to_save/arc_challenge  #实际需要放置的路径
    table_name: arc_challenge
    namespace: experiment
    role: guest_0
  - file: path_to_save/arc_challenge  #实际需要放置的路径
    table_name: arc_challenge
    namespace: experiment
    role: host_0
fedmkt_lora_vs_zero_shot:
  fedmkt_lora:
    pretrained: "Sheared-LLaMa-1.3B"   # 模型放置的实际路径
    script: "./fedmkt.py"
    conf: "./fedmkt_config.yaml"
    peft_path_format: "{{fate_base}}/fate_flow/model/{{job_id}}/guest/{{party_id}}/{{model_task_name}}/0/output/output_model/model_directory/"
    tasks:
      - "arc_challenge"
  bloom_zero_shot:
    pretrained: "Sheared-LLaMa-1.3B"   # 模型放置的实际路径
    tasks:
      - "arc_challenge"
~~~

 	fedmkt_config.yaml 仅需修改下模型的实际路径

~~~
# pdss_config.yaml 开头片段
data:
  guest:
    namespace: experiment
    name: arc_challenge
  host:
    namespace: experiment
    name: arc_challenge

# 配置路径
paths:
  process_data_output_dir: "examples/data/arc_e" # 数据放置的实际路径
  llm_pretrained_path: "Sheared-LLaMa-1.3B" # 模型放置的实际路径
  slm_0_pretrained_path: "opt-1.3b" # 模型放置的实际路径
  slm_1_pretrained_path: "gpt2" # 模型放置的实际路径
  vocab_mapping_directory: "vocab_mapping_datas" # vocab_mapping数据放置的实际路径
~~~

##### 3. 运行

~~~
# 环境准备
cd /fate/
source fate/bin/init_env.sh

# 命令执行
fate_test llmsuite -i fate_llm/examples/fedmkt/ --yes
~~~

##### 4. 问题定位

​	任务运行失败，报错，会在当前执行命令的目录下生成一个logs目录，找到对应的任务，检查stdout，或者exception.log检查报错原因。

