####  fdkt_fate_test_quick_start_zh

##### 1. 数据准备

###### Dataset: Yelp
从 Yelp 数据集中处理并采样了“健康”子域的[数据](https://arxiv.org/abs/1509.01626)，数据集可以从[此处](https://www.yelp.com/dataset)下载。下载数据集后，执行以下命令解压下载的数据集。

```shell
tar -xvf yelp_dataset.tar
```
以下代码将对“健康”子域的 5000 条数据行进行采样，并将在文件夹“./processed_data/Health/train.json”下生成训练数据
~~~
# Health数据处理脚本，稍微修改即可执行，执行需在fate环境下执行
import os
import json
import sys
import random
from pathlib import Path
random.seed(42)


base_dir = "./"
business_data_path = os.path.join(base_dir, 'yelp_academic_dataset_business.json')
review_data_path = os.path.join(base_dir, 'yelp_academic_dataset_review.json')

business_data_file = open(business_data_path, 'r')
review_data_file = open(review_data_path, 'r')

categories_list = ['Restaurants', 'Shopping', 'Arts', 'Health']
business_dic = {}
data_dict = {}
for category in categories_list:
    business_dic[category] = set()
    data_dict[category] = []


def get_categories(categories):
    return_list = []
    for category in categories_list:
        if category in categories:
            return_list.append(category)
    return return_list


for line in business_data_file.readlines():
    dic = json.loads(line)
    if 'categories' in dic.keys() and dic['categories'] is not None:
        category = get_categories(dic['categories'])
        if len(category) == 1:
            business_dic[category[0]].add(dic['business_id'])

# for category in categories_list:
for line in review_data_file.readlines():
    dic = json.loads(line)
    if 'business_id' in dic.keys() and dic['business_id'] is not None:
        for category in categories_list:
            if dic['business_id'] in business_dic[category]:
                if dic['text'] is not None and dic['stars'] is not None:
                    data_dict[category].append({'text': dic['text'], 'stars': dic['stars']})
                break

train_data_path = os.path.join('processed_data', "Health", 'train.json')
os.makedirs(Path(train_data_path).parent, exist_ok=True)
train_data_file = open(train_data_path, 'w')
data_list = data_dict["Health"]

sample_data_dict = dict()

for data in data_list:
    star = int(data["stars"])
    if star not in sample_data_dict:
        sample_data_dict[star] = []

    sample_data_dict[star].append(data)

data_list = []
star_keys = list(sample_data_dict.keys())
for star in star_keys:
    sample_data = sample_data_dict[star][:1000]
    random.shuffle(sample_data)
    data_list.extend(sample_data)

random.shuffle(data_list)
json.dump(data_list, train_data_file, indent=4)
train_data_file.close()
~~~

​	将数据集路径与名称和命名空间绑定。记住自己的数据集保存路径的路径。

~~~
flow table bind --namespace experiment --name slm_train --path path_to_save/train.json
~~~

##### 2. 配置文件修改

​	 test_fdkt_llmsuite.yaml 仅需修改数据与模型的实际路径

~~~
data:
  - file: path_to_save/train.json  #实际需要放置的路径
    table_name: slm_train
    namespace: experiment
    role: guest_0
  - file: path_to_save/train.json  #实际需要放置的路径
    table_name: slm_train
    namespace: experiment
    role: host_0
fdkt_lora_vs_zero_shot:
  pdss_lora:
    pretrained: "Sheared-LLaMa-1.3B"   # 模型放置的实际路径
    script: "./fdkt.py"
    conf: "./fdkt_config.yaml"

~~~

 	fdkt_config.yaml 仅需修改下模型的实际路径

~~~
# fdkt_config.yaml 开头片段
data:
  guest:
    namespace: experiment
    name: slm_train
  host:
    namespace: experiment
    name: slm_train

# 模型路径
pretrained_model_path: "Sheared-LLaMa-1.3B" # 模型放置的实际路径
~~~

##### 3. 运行

~~~
# 环境准备
cd /fate/
source fate/bin/init_env.sh

# 命令执行
fate_test llmsuite -i fate_llm/examples/fdkt/ --yes
~~~

**注意：** fdkt暂不支持数据评估，如您环境是torch=1.13.1，需升级至torch=2.3.1
##### 4. 问题定位

​	任务运行失败，报错，会在当前执行命令的目录下生成一个logs目录，找到对应的任务，检查stdout，或者exception.log检查报错原因。

