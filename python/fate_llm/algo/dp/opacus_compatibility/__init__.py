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
from .grad_sample.embedding import compute_embedding_grad_sample
from .optimizers.optimizer import add_noise_wrapper


def add_layer_compatibility(opacus):
    replace_method = []
    for k, v in opacus.GradSampleModule.GRAD_SAMPLERS.items():
        if v.__name__ == "compute_embedding_grad_sample":
            replace_method.append(k)

    for k in replace_method:
        opacus.GradSampleModule.GRAD_SAMPLERS[k] = compute_embedding_grad_sample


def add_optimizer_compatibility(optimizer):
    add_noise_wrapper(optimizer)
