import yaml
from fate_client.pipeline.components.fate.homo_nn import HomoNN, get_config_of_fdkt_runner
from fate_client.pipeline.components.fate.nn.algo_params import FDKTTrainingArguments
from fate_client.pipeline.components.fate.nn.loader import LLMModelLoader, LLMDatasetLoader, LLMDataFuncLoader
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate.reader import Reader
from fate_client.pipeline.components.fate.nn.torch import nn, optim
from typing import Union, Dict
import argparse

def main(config="../../config.yaml", param: Union[Dict, str] = None, namespace=""):
    if isinstance(config, str):
        config = test_utils.load_job_config(config)
    if isinstance(param, str):
        param = yaml.safe_load(param)
    # Load the configuration file
    parties = config.parties
    guest = parties.guest[0]
    arbiter = parties.arbiter[0]
    
    pipeline = FateFlowPipeline().set_parties(guest=guest, arbiter=arbiter)
    pipeline.bind_local_path(path=param["datasets"]["slm_data_path"], namespace=param["data"]["guest"]["namespace"], 
                                                 name=param["data"]["guest"]["name"])
    
    def get_llm_conf():
        embedding_model = LLMModelLoader(
            "embedding_transformer.st_model",
            "SentenceTransformerModel",
            model_name_or_path=param['llm']['embedding_model_path']
        )
    
        dataset = LLMDatasetLoader(
            "flex_dataset",
            "FlexDataset",
            tokenizer_name_or_path=param['llm']['pretrained_path'],
            need_preprocess=True,
            dataset_name="yelp_review",
            data_part="train.json",
            load_from="json",
            few_shot_num_per_label=1,
        )
    
        training_args = FDKTTrainingArguments(
            sample_num_per_cluster=4,
            filter_prompt_max_length=2 ** 14,
            filter_generation_config=dict(
                max_tokens=3000,
            ),
            use_cpu=param['slm']['training_args']['use_cpu'],
            aug_generation_config=dict(
                max_tokens=3000,
                temperature=0.8,
                top_p=0.9,
            ),
            aug_prompt_num=200,
        )
    
        inference_inst_conf = dict(
            module_name="fate_llm.algo.fdkt.inference_inst",
            item_name="api_init",
            kwargs=dict(
                api_url=param['client']['api_url'],
                model_name=param['llm']['pretrained_path'],
                api_key=param['client']['api_key']
            )
        )
    
        return get_config_of_fdkt_runner(
            training_args=training_args,
            embedding_model=embedding_model,
            dataset=dataset,
            inference_inst_conf=inference_inst_conf,
        )


    def get_slm_conf():
        slm_model = LLMModelLoader(
            "hf_model",
            "HFAutoModelForCausalLM",
            pretrained_model_name_or_path=param['slm']['pretrained_path'],
            torch_dtype="bfloat16",
        )
    
        tokenizer = LLMDataFuncLoader(
            "tokenizers.cust_tokenizer",
            "get_tokenizer",
            tokenizer_name_or_path=param['slm']['pretrained_path'],
            pad_token_id=50256
        )
    
        training_args = FDKTTrainingArguments(
            use_cpu=param['slm']['training_args']['use_cpu'],
            device_id=1,
            num_train_epochs=param['slm']['training_args']['num_train_epochs'],
            per_device_train_batch_size=param['slm']['training_args']['per_device_train_batch_size'],
            slm_generation_batch_size=param['slm']['training_args']['slm_generation_batch_size'],
            seq_num_for_single_category=param['slm']['training_args']['seq_num_for_single_category'],
            slm_generation_config=param['slm']['training_args']['slm_generation_config'],
        )
    
        dataset = LLMDatasetLoader(
            "flex_dataset",
            "FlexDataset",
            tokenizer_name_or_path=param['slm']['pretrained_path'],
            need_preprocess=True,
            dataset_name="yelp_review",
            data_part="train",
            load_from="json",
            select_num=2000,
            few_shot_num_per_label=1,
        )
    
        optimizer = optim.Adam(lr=0.01)
    
        return get_config_of_fdkt_runner(
            model=slm_model,
            tokenizer=tokenizer,
            training_args=training_args,
            dataset=dataset,
            optimizer=optimizer,
            data_collator=LLMDataFuncLoader(
                "data_collator.cust_data_collator",
                "get_seq2seq_data_collator",
                label_pad_token_id=50256,
                tokenizer_name_or_path=param['slm']['pretrained_path'],
                pad_token_id=50256,
            ),
        )


    
    reader_0 = Reader("reader_0", runtime_parties=dict(guest=guest))
    reader_0.guest.task_parameters(
        namespace=param["data"]["guest"]["namespace"],
        name=param["data"]["guest"]["name"]
    )

    homo_nn_0 = HomoNN(
        'homo_nn_0',
        train_data=reader_0.outputs["output_data"],
        runner_module="fdkt_runner",
        runner_class="FDKTRunner",
    )

    homo_nn_0.arbiter.task_parameters(
        runner_conf=get_llm_conf()
    )

    homo_nn_0.guest.task_parameters(
        runner_conf=get_slm_conf()
    )

    pipeline.add_tasks([reader_0, homo_nn_0])
    pipeline.conf.set("task", dict(engine_run={"cores": 1}))

    pipeline.compile()
    pipeline.fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("FDKT JOB")
    parser.add_argument("-c", "--config", type=str, help="Path to config file", default="./config.yaml")
    parser.add_argument("-p", "--param", type=str, help="Path to parameter file", default="./fdkt_config.yaml")
    args = parser.parse_args()
    main(args.config, args.param)
