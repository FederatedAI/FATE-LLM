import argparse
import yaml
from fate_client.pipeline.components.fate.reader import Reader
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate.homo_nn import HomoNN, get_conf_of_ot_runner
from fate_client.pipeline.components.fate.nn.algo_params import Seq2SeqTrainingArguments, FedAVGArguments
from fate_client.pipeline.components.fate.nn.loader import LLMModelLoader, LLMDatasetLoader, LLMDataFuncLoader
from fate_client.pipeline.components.fate.nn.torch.base import Sequential
from fate_client.pipeline.components.fate.nn.torch import nn

def load_params(file_path):
    """Load and parse the YAML params file."""
    with open(file_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def setup_pipeline(params):
    """Set up the pipeline using the provided parameters."""
    guest = params['pipeline']['guest']
    arbiter = params['pipeline']['arbiter']
    pretrained_model_path = params['paths']['pretrained_model_path']
    
    pipeline = FateFlowPipeline().set_parties(guest=guest, arbiter=arbiter)
    
    reader = Reader("reader_0", runtime_parties=dict(guest=guest))
    reader.guest.task_parameters(
        namespace=params['pipeline']['namespace'],
        name=params['pipeline']['name']
    )
    
    client_model = LLMModelLoader(
        module_name=params['models']['client']['module_name'],
        item_name=params['models']['client']['item_name'],
        model_name_or_path=pretrained_model_path,
        emulator_layer_num=params['models']['client']['emulator_layer_num'],
        adapter_top_layer_num=params['models']['client']['adapter_top_layer_num'],
        adapter_bottom_layer_num=params['models']['client']['adapter_bottom_layer_num']
    )
    
    server_model = LLMModelLoader(
        module_name=params['models']['server']['module_name'],
        item_name=params['models']['server']['item_name'],
        model_name_or_path=pretrained_model_path,
        emulator_layer_num=params['models']['server']['emulator_layer_num'],
        adapter_top_layer_num=params['models']['server']['adapter_top_layer_num'],
        adapter_bottom_layer_num=params['models']['server']['adapter_bottom_layer_num']
    )
    
    dataset = LLMDatasetLoader(
        module_name=params['dataset']['module_name'],
        item_name=params['dataset']['item_name'],
        tokenizer_name_or_path=params['dataset']['tokenizer_name_or_path'],
        select_num=params['dataset']['select_num']
    )
    
    data_collator = LLMDataFuncLoader(
        module_name=params['data_collator']['module_name'],
        item_name=params['data_collator']['item_name'],
        tokenizer_name_or_path=params['data_collator']['tokenizer_name_or_path']
    )
    
    train_args = Seq2SeqTrainingArguments(
        per_device_train_batch_size=params['training']['batch_size'],
        learning_rate=params['training']['learning_rate'],
        disable_tqdm=False,
        num_train_epochs=params['training']['num_train_epochs'],
        logging_steps=params['training']['logging_steps'],
        logging_strategy='steps',
        dataloader_num_workers=4,
        use_cpu=False,
        deepspeed=params['training']['deepspeed'],  # Add DeepSpeed config here
        remove_unused_columns=False,
        fp16=True
    )
    
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
    
    homo_nn = HomoNN(
        'nn_0',
        train_data=reader.outputs["output_data"],
        runner_module="offsite_tuning_runner",
        runner_class="OTRunner"
    )
    
    homo_nn.guest.task_parameters(runner_conf=client_conf)
    homo_nn.arbiter.task_parameters(runner_conf=server_conf)
    
    # If using Eggroll, you can add this line to submit your job
    homo_nn.guest.conf.set("launcher_name", "deepspeed")
    
    pipeline.add_tasks([reader, homo_nn])
    pipeline.conf.set("task", dict(engine_run=params['pipeline']['engine_run']))
    pipeline.compile()
    pipeline.fit()

def main(config_file, param_file):
    params = load_params(param_file)
    setup_pipeline(params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("LLMSUITE Offsite-tuning JOB")
    parser.add_argument("-c", "--config", type=str,
                        help="Path to config file", default="./config.yaml")
    parser.add_argument("-p", "--param", type=str,
                        help="Path to parameter file", default="./test_offsite_tuning_llmsuite.yaml")
    args = parser.parse_args()
    main(args.config, args.param)
