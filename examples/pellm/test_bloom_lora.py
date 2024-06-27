import time
from fate_client.pipeline.components.fate.reader import Reader
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate.homo_nn import HomoNN, get_config_of_seq2seq_runner
from fate_client.pipeline.components.fate.nn.algo_params import Seq2SeqTrainingArguments, FedAVGArguments
from fate_client.pipeline.components.fate.nn.loader import LLMModelLoader, LLMDatasetLoader, LLMDataFuncLoader
from peft import LoraConfig, TaskType
from fate_client.pipeline.utils import test_utils
import argparse
import yaml
from typing import Union, Dict


def main(config="../../config.yaml", param: Union[Dict, str] = None, namespace=""):
    if isinstance(config, str):
        config = test_utils.load_job_config(config)
    if isinstance(param, str):
        param = yaml.safe_load(param)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]
    pipeline = FateFlowPipeline().set_parties(guest=guest, host=host, arbiter=arbiter)

    reader_0 = Reader("reader_0", runtime_parties=dict(guest=guest, host=host))
    reader_0.guest.task_parameters(
        namespace=param["data"]["guest"]["namespace"],
        name=param["data"]["guest"]["name"]
    )
    reader_0.hosts[0].task_parameters(
        namespace=param["data"]["host"]["namespace"],
        name=param["data"]["host"]["name"]
    )

    lora_config = LoraConfig(**param["peft_config"])
    lora_config.target_modules = list(lora_config.target_modules)

    pretrained_model_path = param["pretrained_model_path"]
    model = LLMModelLoader(
        "pellm.bloom",
        "Bloom",
        pretrained_path=pretrained_model_path,
        peft_type="LoraConfig",
        peft_config=lora_config.to_dict(),
        trust_remote_code=True
    )

    tokenizer_params = dict(
        tokenizer_name_or_path=pretrained_model_path,
        trust_remote_code=True,
    )

    dataset = LLMDatasetLoader(
        "prompt_dataset",
        "PromptDataset",
        **tokenizer_params,
    )

    data_collator = LLMDataFuncLoader(
        "data_collator.cust_data_collator",
        "get_seq2seq_data_collator",
        **tokenizer_params,
    )

    conf = get_config_of_seq2seq_runner(
        algo='fedavg',
        model=model,
        dataset=dataset,
        data_collator=data_collator,
        training_args=Seq2SeqTrainingArguments(
            num_train_epochs=param["epoch"],
            per_device_train_batch_size=param["batch_size"],
            remove_unused_columns=False,
            predict_with_generate=False,
            deepspeed=param["ds_config"],
            learning_rate=param["lr"],
            use_cpu=False,  # this must be set as we will gpu
            fp16=True,
        ),
        fed_args=FedAVGArguments(),
        task_type='causal_lm',
        save_trainable_weights_only=True  # only save trainable weights
    )

    homo_nn_0 = HomoNN(
        'nn_0',
        runner_conf=conf,
        train_data=reader_0.outputs["output_data"],
        runner_module="homo_seq2seq_runner",
        runner_class="Seq2SeqRunner",
    )

    homo_nn_0.guest.conf.set("launcher_name", "deepspeed")  # tell schedule engine to run task with deepspeed
    homo_nn_0.hosts[0].conf.set("launcher_name", "deepspeed")  # tell schedule engine to run task with deepspeed

    pipeline.add_tasks([reader_0, homo_nn_0])
    pipeline.conf.set("task", dict(engine_run={"cores": 1}))  # the number of gpus of each party

    pipeline.compile()
    pipeline.fit()

    return pretrained_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LLMSUITE PIPELINE JOB")
    parser.add_argument("-c", "--config", type=str,
                        help="config file", default="../../config.yaml")
    parser.add_argument("-p", "--param", type=str,
                        help="config file for params", default="./bloom_lora_config.yaml")
    args = parser.parse_args()
    main(args.config, args.param)
