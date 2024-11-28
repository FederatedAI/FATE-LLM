import argparse
import yaml
from fate_client.pipeline.components.fate.reader import Reader
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate.homo_nn import HomoNN, get_conf_of_ot_runner
from fate_client.pipeline.components.fate.nn.algo_params import Seq2SeqTrainingArguments, FedAVGArguments
from fate_client.pipeline.components.fate.nn.loader import LLMModelLoader, LLMDatasetLoader, LLMDataFuncLoader
from fate_client.pipeline.components.fate.nn.torch import nn
from typing import Union, Dict

def main(config="../../config.yaml", param: Union[Dict, str] = None, namespace=""):
    if isinstance(config, str):
        config = test_utils.load_job_config(config)
    if isinstance(param, str):
        param = yaml.safe_load(param)
    # Load the configuration file
    parties = config.parties  
    guest = parties.guest[0]
    arbiter = parties.arbiter[0]  
    pretrained_model_path = param["pretrained_model_path"]
    
    # Create pipeline
    pipeline = FateFlowPipeline().set_parties(guest=guest, arbiter=arbiter)
    
    # Set up the data reader
    reader_0 = Reader("reader_0", runtime_parties=dict(guest=guest))
    reader_0.guest.task_parameters(
        namespace=param["data"]["guest"]["namespace"],
        name=param["data"]["guest"]["name"]
    )
    
    # Load the LLM model
    client_model = LLMModelLoader(
        module_name='offsite_tuning.gpt2', item_name='GPT2LMHeadSubModel',
        model_name_or_path=pretrained_model_path,
        emulator_layer_num=param["model_config"]["emulator_layer_num"],
        adapter_top_layer_num=param["model_config"]["adapter_top_layer_num"],
        adapter_bottom_layer_num=param["model_config"]["adapter_bottom_layer_num"]
    )
    
    server_model = LLMModelLoader(
        module_name='offsite_tuning.gpt2', item_name='GPT2LMHeadMainModel',
        model_name_or_path=pretrained_model_path,
        emulator_layer_num=param["model_config"]["emulator_layer_num"],
        adapter_top_layer_num=param["model_config"]["adapter_top_layer_num"],
        adapter_bottom_layer_num=param["model_config"]["adapter_bottom_layer_num"]
    )
    
    # Load the dataset and data processor
    dataset = LLMDatasetLoader(
        module_name='qa_dataset', item_name='QaDataset',
        tokenizer_name_or_path=pretrained_model_path,
        select_num=100
    )  
    
    data_collator = LLMDataFuncLoader(
        module_name='data_collator.cust_data_collator',
        item_name='get_seq2seq_data_collator',
        tokenizer_name_or_path=pretrained_model_path
    )    
    
    # DeepSpeed config
    ds_config = param["deepspeed_config"]
    # Training parameter settings
    train_args = Seq2SeqTrainingArguments(
        per_device_train_batch_size=param["training"]["batch_size"],
        learning_rate=param["training"]["lr"],
        disable_tqdm=param["training"]["disable_tqdm"],
        num_train_epochs=param["training"]["num_train_epochs"],
        logging_steps=param["training"]["logging_steps"],
        logging_strategy='steps',
        dataloader_num_workers=param["training"]["dataloader_num_workers"],
        use_cpu=False,
        deepspeed=ds_config,
        remove_unused_columns=False,
        fp16=True
    )
    
    # Set the configuration of the client and server models
    client_conf = get_conf_of_ot_runner(
        model=client_model,
        dataset=dataset,
        data_collator=data_collator,
        training_args=train_args,
        fed_args=FedAVGArguments(),
        aggregate_model=False,
    )
    
    server_conf = get_conf_of_ot_runner(
        model=server_model,
        dataset=dataset,
        data_collator=data_collator,
        training_args=train_args,
        fed_args=FedAVGArguments(),
        aggregate_model=False
    )
    
    # Set up the HomoNN component
    homo_nn_0 = HomoNN(
        'nn_0',
        train_data=reader_0.outputs["output_data"],
        runner_module="offsite_tuning_runner",
        runner_class="OTRunner"
    ) 
    homo_nn_0.guest.task_parameters(runner_conf=client_conf)
    homo_nn_0.arbiter.task_parameters(runner_conf=server_conf)
    
    # Using DeepSpeed
    homo_nn_0.guest.conf.set("launcher_name", "deepspeed")
    # Build a task pipeline
    pipeline.add_tasks([reader_0, homo_nn_0])
    pipeline.conf.set("task", dict(engine_run={"cores": 1}))
    pipeline.compile()
    pipeline.fit()
    return pretrained_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("LLMSUITE PIPELINE JOB")
    parser.add_argument("-c", "--config", type=str,
                        help="config file", default="./config.yaml")
    parser.add_argument("-p", "--param", type=str,
                        help="config file for params", default="./offsite_tuning_config.yaml")
    args = parser.parse_args()
    main(args.config, args.param)