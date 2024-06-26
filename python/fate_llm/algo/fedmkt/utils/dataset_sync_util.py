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
import logging
import datasets
import torch
import torch.distributed as dist
from fate_llm.algo.fedmkt.utils.vars_define import (
    METRIC,
    PER_STEP_LOGITS,
    PER_STEP_INDICES,
)

logger = logging.getLogger(__name__)


def sync_dataset(dataset, local_rank, world_size, device):
    integer_keys_2d = ["input_ids", "attention_mask", "labels"]
    integer_keys_3d = [PER_STEP_INDICES]
    float_keys_3d = [PER_STEP_LOGITS]
    float_keys_1d = [METRIC]

    if local_rank == 0:
        for key in integer_keys_2d + integer_keys_3d + float_keys_3d + float_keys_1d:
            if key in integer_keys_2d or key in integer_keys_3d:
                dtype = torch.int32
            else:
                dtype = torch.float64

            values = dataset[key]
            v_tensor = torch.tensor(values, dtype=dtype).cuda(device)
            shape_tensor = torch.tensor(v_tensor.shape, dtype=torch.int32).cuda(device)
            shape_tensors = [shape_tensor for _ in range(world_size)]
            dist.scatter(shape_tensor, shape_tensors, async_op=False)

            v_tensors = [v_tensor for _ in range(world_size)]
            dist.scatter(v_tensor, v_tensors, async_op=False)

        return dataset

    else:
        data_dict = dict()
        for key in integer_keys_2d + integer_keys_3d + float_keys_3d + float_keys_1d:
            if key in integer_keys_2d or key in integer_keys_3d:
                dtype = torch.int32
            else:
                dtype = torch.float64

            if key in integer_keys_2d:
                shape_tensor = torch.tensor([0, 0], dtype=torch.int32).cuda(device)
            elif key in float_keys_3d or key in integer_keys_3d:
                shape_tensor = torch.tensor([0, 0, 0], dtype=torch.int32).cuda(device)
            else:
                shape_tensor = torch.tensor([0], dtype=torch.int32).cuda(device)

            dist.scatter(shape_tensor, src=0, async_op=False)
            v_tensor = torch.zeros(shape_tensor.tolist(), dtype=dtype).cuda(device)
            dist.scatter(v_tensor, src=0, async_op=False)
            data_dict[key] = v_tensor.tolist()

        return datasets.Dataset.from_dict(data_dict)
