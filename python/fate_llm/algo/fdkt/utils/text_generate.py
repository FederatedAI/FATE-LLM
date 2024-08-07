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
from tqdm import tqdm
from typing import Any, Dict, List


def slm_text_generate(
    inference_inst,
    model,
    tokenizer,
    prompt_dict,
    seq_num_for_single_category,
    batch_size,
    use_cpu,
    generation_config
):
    generated_ret = dict(
        inputs=list(),
        labels=list(),
    )
    if inference_inst is not None:
        for label, prompt in prompt_dict.items():
            generated_sequences = inference_inst.inference([prompt] * seq_num_for_single_category, generation_config)
            for g in generated_sequences:
                generated_ret["inputs"].append(g)
                generated_ret["labels"].append(label)
    else:
        model.eval()
        for label, prompt_ids in prompt_dict.items():
            prompt_length = len(prompt_ids)
            batch_num = (seq_num_for_single_category + batch_size - 1) // batch_size
            for batch_idx in tqdm(range(batch_num)):
                if batch_idx + 1 == batch_num:
                    cur_batch_size = seq_num_for_single_category - batch_idx * batch_size
                else:
                    cur_batch_size = batch_size
                input_ids = prompt_ids.repeat(cur_batch_size, 1)

                if not use_cpu:
                    input_ids = input_ids.to(model.device)

                output_sequences = model.generate(
                    input_ids=input_ids,
                    **generation_config
                )
                output_sequences = output_sequences[:, prompt_length:]

                generated_sequences = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

                for g in generated_sequences:
                    generated_ret["inputs"].append(g)
                    generated_ret["labels"].append(label)

    return generated_ret


def general_text_generate(
    inference_inst,
    model,
    tokenizer,
    generation_config: Dict[Any, Any],
    prompts: List[str],
    batch_size,
    use_cpu: bool,
    prompt_max_length
):
    if inference_inst is not None:
        if prompt_max_length is not None:
            prompts = [prompt[:prompt_max_length] for prompt in prompts]
        generate_texts = inference_inst.inference(prompts, generation_config)
    else:
        model.eval()
        generate_texts = []
        batch_num = (len(prompts) + batch_size - 1) // batch_size
        for batch_idx in range(batch_num):
            batch_data = prompts[batch_idx * batch_size: (batch_idx + 1) * batch_size]

            inputs = tokenizer(batch_data, return_tensors="pt", padding="longest", truncation=True,
                               max_length=prompt_max_length)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            if not use_cpu:
                input_ids = input_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)

            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )

            batch_responses = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)

            generate_texts.extend(batch_responses)

    return generate_texts
