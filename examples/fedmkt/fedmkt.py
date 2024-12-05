from fate_client.pipeline.components.fate.homo_nn import HomoNN, get_config_of_fedmkt_runner
from fate_client.pipeline.components.fate.nn.algo_params import FedMKTTrainingArguments, FedAVGArguments
from fate_client.pipeline.components.fate.nn.loader import LLMModelLoader, LLMDatasetLoader, LLMDataFuncLoader
from peft import LoraConfig, TaskType
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate.reader import Reader
from transformers import AutoConfig
import argparse
import yaml
from typing import Union, Dict

def main(config="../../config.yaml", param: Union[Dict, str] = None, namespace=""):
    if isinstance(config, str):
        config = test_utils.load_job_config(config)
    if isinstance(param, str):
        param = yaml.safe_load(param)
        
    # load config
    parties = config.parties
    guest = parties.guest[0]  # replace with actual guest party ID
    host = parties.host[0]    # replace with actual host party ID
    arbiter = parties.arbiter[0]  # replace with actual arbiter party ID
    
    process_data_output_dir = param['paths']['process_data_output_dir']
    llm_pretrained_path = param['paths']['llm_pretrained_path']
    slm_0_pretrained_path = param['paths']['slm_0_pretrained_path']
    slm_1_pretrained_path = param['paths']['slm_1_pretrained_path']
    llm_slm_pairs = [
        (llm_pretrained_path, slm_0_pretrained_path),
        (llm_pretrained_path, slm_1_pretrained_path)
    ]
    vocab_mapping_directory = param['paths']['vocab_mapping_directory']

    slm_to_llm_vocab_mapping_paths = [
        vocab_mapping_directory + "/" + path for path in param['paths']['slm_to_llm_vocab_mapping_paths']
    ]
    llm_to_slm_vocab_mapping_paths = [
        vocab_mapping_directory + "/" + path for path in param['paths']['llm_to_slm_vocab_mapping_paths']
    ]
    slm_pretrained_paths = [slm_0_pretrained_path, slm_1_pretrained_path]
    slm_models = param['models']['slm_models']
    slm_lora_target_modules = [
        ["q_proj", "v_proj"],
        ["c_attn"]
    ]
    slm_models = [
        ("pellm.opt", "OPT"),
        ("pellm.gpt2", "GPT2CLM")
    ]
    
    def get_llm_conf():
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=param['lora_config']['llm']['r'],
            lora_alpha=param['lora_config']['llm']['lora_alpha'],
            lora_dropout=param['lora_config']['llm']['lora_dropout'],
            target_modules=param['lora_config']['llm']['target_modules']
        )
        lora_config.target_modules = list(lora_config.target_modules)

        llm_model = LLMModelLoader(
            "pellm.llama",
            "LLaMa",
            pretrained_path=llm_pretrained_path,
            peft_type="LoraConfig",
            peft_config=lora_config.to_dict(),
            torch_dtype="bfloat16"
        )

        pub_dataset = LLMDatasetLoader(
            "qa_dataset",
            "QaDataset",
            tokenizer_name_or_path=llm_pretrained_path,
            need_preprocess=True,
            dataset_name="arc_challenge",
            data_part="common",
            seq_max_len=512
        )

        training_args = FedMKTTrainingArguments(
            global_epochs=param['training']['llm']['global_epochs'],
            per_device_train_batch_size=param['training']['llm']['per_device_train_batch_size'],
            gradient_accumulation_steps=param['training']['llm']['gradient_accumulation_steps'],
            learning_rate=param['training']['llm']['learning_rate'],
            output_dir=param['training']['llm']['output_dir'],
            dataloader_num_workers=param['training']['llm']['dataloader_num_workers'],
            remove_unused_columns=param['training']['llm']['remove_unused_columns'],
            warmup_ratio=param['training']['llm']['warmup_ratio'],
            lr_scheduler_type=param['training']['llm']['lr_scheduler_type'],
            optim=param['training']['llm']['optim'],
            adam_beta1=param['training']['llm']['adam_beta1'],
            adam_beta2=param['training']['llm']['adam_beta2'],
            weight_decay=param['training']['llm']['weight_decay'],
            max_grad_norm=param['training']['llm']['max_grad_norm'],
            use_cpu=param['training']['llm']['use_cpu'],
            vocab_size=AutoConfig.from_pretrained(llm_pretrained_path).vocab_size,
        )

        fed_args = FedAVGArguments(
            aggregate_strategy='epoch',
            aggregate_freq=1
        )

        tokenizer = LLMDataFuncLoader(
            "tokenizers.cust_tokenizer",
            "get_tokenizer",
            tokenizer_name_or_path=llm_pretrained_path
        )

        slm_tokenizers = [
            LLMDataFuncLoader("tokenizers.cust_tokenizer", "get_tokenizer", tokenizer_name_or_path=path)
            for path in slm_pretrained_paths
        ]

        return get_config_of_fedmkt_runner(
            model=llm_model,
            training_args=training_args,
            fed_args=fed_args,
            pub_dataset=pub_dataset,
            tokenizer=tokenizer,
            slm_tokenizers=slm_tokenizers,
            slm_to_llm_vocab_mapping_paths=slm_to_llm_vocab_mapping_paths,
            pub_dataset_path=process_data_output_dir,
            save_trainable_weights_only=True,
        )
    
    def get_slm_conf(slm_idx):
        slm_pretrained_path = slm_pretrained_paths[slm_idx]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False, 
            r=param['lora_config']['slm'][slm_idx]['r'],
            lora_alpha=param['lora_config']['slm'][slm_idx]['lora_alpha'],
            lora_dropout=param['lora_config']['slm'][slm_idx]['lora_dropout'],
            target_modules=param['lora_config']['slm'][slm_idx]['target_modules']
        )
        lora_config.target_modules = list(lora_config.target_modules)
        llm_to_slm_vocab_mapping = llm_to_slm_vocab_mapping_paths[slm_idx]

        slm_model = LLMModelLoader(
            slm_models[slm_idx][0],
            slm_models[slm_idx][1],
            pretrained_path=slm_pretrained_path,
            peft_type="LoraConfig",
            peft_config=lora_config.to_dict(),
        )
        vocab_size = AutoConfig.from_pretrained(slm_pretrained_path).vocab_size

        pub_dataset = LLMDatasetLoader(
            "qa_dataset",
            "QaDataset",
            tokenizer_name_or_path=slm_pretrained_path,
            need_preprocess=True,
            dataset_name="arc_challenge",
            data_part="common",
            seq_max_len=512
        )

        priv_dataset = LLMDatasetLoader(
            "qa_dataset",
            "QaDataset",
            tokenizer_name_or_path=slm_pretrained_path,
            need_preprocess=True,
            dataset_name="arc_challenge",
            data_part="client_0",
            seq_max_len=512
        )

        training_args = FedMKTTrainingArguments(
            global_epochs=param['training']['slm']['global_epochs'],
            per_device_train_batch_size=param['training']['slm']['per_device_train_batch_size'],
            gradient_accumulation_steps=param['training']['slm']['gradient_accumulation_steps'],
            learning_rate=param['training']['slm']['learning_rate'] if slm_idx != 1 else 3e-4,
            output_dir=param['training']['slm']['output_dir'],
            dataloader_num_workers=param['training']['slm']['dataloader_num_workers'],
            remove_unused_columns=param['training']['slm']['remove_unused_columns'],
            warmup_ratio=param['training']['slm']['warmup_ratio'],
            lr_scheduler_type=param['training']['slm']['lr_scheduler_type'],
            optim=param['training']['slm']['optim'],
            adam_beta1=param['training']['slm']['adam_beta1'],
            adam_beta2=param['training']['slm']['adam_beta2'],
            weight_decay=param['training']['slm']['weight_decay'],
            max_grad_norm=param['training']['slm']['max_grad_norm'],
            use_cpu=param['training']['slm']['use_cpu'],
            vocab_size=vocab_size,
        )

        fed_args = FedAVGArguments(
            aggregate_strategy='epoch',
            aggregate_freq=1
        )

        tokenizer = LLMDataFuncLoader(
            "tokenizers.cust_tokenizer",
            "get_tokenizer",
            tokenizer_name_or_path=slm_pretrained_path
        )

        llm_tokenizer = LLMDataFuncLoader(
            "tokenizers.cust_tokenizer", 
            "get_tokenizer", 
            tokenizer_name_or_path=llm_pretrained_path
        )

        data_collator = LLMDataFuncLoader(
            module_name='data_collator.cust_data_collator',
            item_name='get_seq2seq_data_collator', 
            tokenizer_name_or_path=slm_pretrained_path
        )

        return get_config_of_fedmkt_runner(
            model=slm_model,
            training_args=training_args,
            fed_args=fed_args,
            pub_dataset=pub_dataset,
            priv_dataset=priv_dataset,
            tokenizer=tokenizer,
            llm_tokenizer=llm_tokenizer,
            llm_to_slm_vocab_mapping_path=llm_to_slm_vocab_mapping,
            pub_dataset_path=process_data_output_dir,
            save_trainable_weights_only=True,
            data_collator=data_collator
        )
        
    pipeline = FateFlowPipeline().set_parties(guest=guest, arbiter=arbiter, host=host)
    pipeline.bind_local_path(path=process_data_output_dir, namespace="experiment", name="arc_challenge")
    
    reader_0 = Reader("reader_0", runtime_parties=dict(guest=guest, host=host))
    reader_0.guest.task_parameters(
        namespace=param['data']['guest']['namespace'],
        name=param['data']['guest']['name']
    )
    reader_0.hosts[0].task_parameters(
        namespace=param['data']['host']['namespace'],
        name=param['data']['host']['name']
    )
    
    homo_nn_0 = HomoNN(
        'nn_0',
        train_data=reader_0.outputs["output_data"],
        runner_module="fedmkt_runner",
        runner_class="FedMKTRunner",
    )
    
    homo_nn_0.arbiter.task_parameters(
        runner_conf=get_llm_conf()
    )
    
    homo_nn_0.guest.task_parameters(
        runner_conf=get_slm_conf(slm_idx=0)
    )

    for idx in range(1):
        homo_nn_0.hosts[idx].task_parameters(
            runner_conf=get_slm_conf(slm_idx=idx + 1)
        )
    
    homo_nn_0.guest.conf.set("launcher_name", "deepspeed")  # tell scheduler engine to run task with deepspeed
    homo_nn_0.hosts[0].conf.set("launcher_name", "deepspeed")  # tell scheduler engine to run task with deepspeed
    homo_nn_0.arbiter.conf.set("launcher_name", "deepspeed")  # tell scheduler engine to run task with deepspeed
    
    pipeline.add_tasks([reader_0, homo_nn_0])
    pipeline.conf.set("task", dict(engine_run={"cores": 1}))  # the number of gpus of each party
    
    pipeline.compile()
    pipeline.fit()
    return llm_pretrained_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("LLMSUITE PIPELINE JOB")
    parser.add_argument("-c", "--config", type=str, help="config file", default="./config.yaml")
    parser.add_argument("-p", "--param", type=str, help="config file for params", default="./fedmkt_config.yaml")
    args = parser.parse_args()
    main(args.config, args.param)
