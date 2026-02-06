#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
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
INVALID_CHARACTERS = "".join([' ', '-', '.', '_', '~', '/', '\\', '*', '|', '#'])
LEAST_WORDS = 10


def filter_invalid_data(data_dict):
    sample_num = len(data_dict["inputs"])
    new_data_dict = dict(
        inputs=list(),
        labels=list()
    )
    for idx in range(sample_num):
        text = data_dict["inputs"][idx].strip(INVALID_CHARACTERS)
        if len(text.split()) < LEAST_WORDS:
            continue

        new_data_dict["inputs"].append(text)
        new_data_dict["labels"].append(data_dict["labels"][idx])

    return new_data_dict
