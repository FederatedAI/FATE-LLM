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
PER_STEP_LOGITS = "per_step_logits"
PER_STEP_INDICES = "per_step_indices"
METRIC = "metric_"

ALIGNED_OTHER_LOGITS = "aligned_other_logits"
ALIGNED_OTHER_INDICES = "aligned_other_indices"
ALIGNED_OTHER_METRIC = "aligned_other_metrice"

SELF_TARGET_DIST = "llm_target_distribution"
OTHER_TARGET_DIST = "slm_target_distribution"

INPUT_KEYS = {"input_ids", "attention_mask", "labels"}
