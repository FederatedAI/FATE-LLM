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
import types
from opacus.optimizers.optimizer import (
    _check_processed_flag,
    _generate_noise,
    _mark_as_processed
)


def add_noise(self):
    """
    Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
    """

    for p in self.params:
        _check_processed_flag(p.summed_grad)

        noise = _generate_noise(
            std=self.noise_multiplier * self.max_grad_norm,
            reference=p.summed_grad,
            generator=self.generator,
            secure_mode=self.secure_mode,
        )
        noise = noise.to(p.summed_grad.dtype)
        p.grad = (p.summed_grad + noise).view_as(p)

        _mark_as_processed(p.summed_grad)


def add_noise_wrapper(optimizer):
    optimizer.add_noise = types.MethodType(add_noise, optimizer)
