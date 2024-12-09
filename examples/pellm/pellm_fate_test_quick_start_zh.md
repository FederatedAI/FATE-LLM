#### pellm_fate_test_quick_start_zh

##### 1. 数据准备

​	用的是一个广告测试生成的数据集，可以从以下链接下载数据集并将其放置在fate_llm/examples/data文件夹中， 同时 $fate_base/fate_llm/python/evaluate/tasks/advertise_gen下也要放置数据train.json, dev.json
​	[data_link_1](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view)

##### 2. 配置文件修改

​	test_pellm_llmsuite.yaml仅需修改数据与模型的实际路径

~~~
# fate_llm/examples/pellm/test_pellm_llmsuite.yaml

data:
  - file: examples/data/AdvertiseGen/train.json   # 下载数据放置的实际路径
    table_name: ad
    namespace: experiment
    role: guest_0
  - file: examples/data/AdvertiseGen/train.json   # 下载数据放置的实际路径
    table_name: ad
    namespace: experiment
    role: host_0
bloom_lora_vs_zero_shot:
  bloom_lora:
    pretrained: "bloom-560m"    # 模型放置的实际路径
    script: "./test_bloom_lora.py"
    conf: "./bloom_lora_config.yaml"
    peft_path_format: "{{fate_base}}/fate_flow/model/{{job_id}}/guest/{{party_id}}/{{model_task_name}}/0/output/output_model/model_directory"
    tasks:
      - "advertise-gen"
  bloom_zero_shot:
    pretrained: "bloom-560m"
    tasks:
      - "advertise-gen"
~~~

  bloom_lora_config.yaml仅需修改模型路径

~~~
#  bloom_lora_config.yaml开头片段

data:
  guest:
    namespace: experiment
    name: ad
  host:
    namespace: experiment
    name: ad
epoch: 1
batch_size: 4
lr: 5e-4
pretrained_model_path: bloom-560m  # 模型放置的实际路径
~~~

##### 3. 运行

~~~
# 环境准备
cd /data/projects/fate/
source /data/projects/fate/bin/init_env.sh

# 命令执行
fate_test llmsuite -i fate_llm/examples/pellm/ --yes
~~~

##### 4. 问题定位

​	任务运行失败，报错，会在当前执行命令的目录下生成一个logs目录，找到对应的任务，检查stdout，或者exception.log检查报错原因。




