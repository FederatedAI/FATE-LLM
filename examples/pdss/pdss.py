import argparse
import yaml
from fate_client.pipeline.components.fate.reader import Reader
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate.nn.loader import Loader
from fate_client.pipeline.components.fate.homo_nn import HomoNN
from fate_client.pipeline.utils import test_utils
from typing import Union, Dict

def main(config="../../config.yaml", param: Union[Dict, str] = None, namespace=""):
    if isinstance(config, str):
        config = test_utils.load_job_config(config)
    if isinstance(param, str):
        param = yaml.safe_load(param)
    # 加载配置文件
    parties = config.parties
    guest = parties.guest[0]
    arbiter = parties.arbiter[0]  
    pretrained_model_path = param["pretrained_model_path"]
    
    # 创建流水线
    pipeline = FateFlowPipeline().set_parties(guest=guest, arbiter=arbiter)
    
    # 设置数据读取器
    reader_0 = Reader("reader_0", runtime_parties=dict(guest=guest))
    reader_0.guest.task_parameters(
        namespace=param["data"]["guest"]["namespace"],
        name=param["data"]["guest"]["name"]
    )

    # 模型加载配置
    model_conf = Loader(
        module_name='fate_llm.model_zoo.hf_model',
        item_name='HFAutoModelForCausalLM',
        pretrained_model_name_or_path=param["model"]["pretrained_model_name_or_path"]
    ).to_dict()

    # 数据处理器配置
    data_collator_conf = Loader(
        module_name='fate_llm.data.data_collator.pdss_collator',
        item_name='get_prefix_data_collator',
        tokenizer_name_or_path=param["data_collator"]["tokenizer_name_or_path"]
    ).to_dict()

    # 客户端推理初始化配置
    infer_init_conf_client = dict(
        module_name='fate_llm.algo.inferdpt.init.default_init',
        item_name='InferDPTAPIClientInit',
        kwargs=dict(
            api_url=param["inference"]["client"]["api_url"],
            model_name=param["inference"]["client"]["model_name"],
            api_key=param["inference"]["client"]["api_key"],
            inferdpt_kit_path=param["inference"]["client"]["inferdpt_kit_path"]
        )
    )

    # 服务器推理初始化配置
    infer_init_conf_server = dict(
        module_name='fate_llm.algo.inferdpt.init.default_init',
        item_name='InferDPTAPIServerInit',
        kwargs=dict(
            api_url=param["inference"]["server"]["api_url"],
            model_name=param["inference"]["server"]["model_name"],
            api_key=param["inference"]["server"]["api_key"]
        )
    )

    # 数据集配置
    dataset_conf = dict(
        module_name='fate_llm.dataset.pdss_dataset',
        item_name='PrefixDataset',
        kwargs=dict(
            tokenizer_path=param["dataset"]["tokenizer_path"],
            predict_input_template=param["dataset"]["predict_input_template"],
            predict_output_template=param["dataset"]["predict_output_template"],
            rationale_input_template=param["dataset"]["rationale_input_template"],
            rationale_output_template=param["dataset"]["rationale_output_template"],
            max_input_length=param["dataset"]["max_input_length"],
            max_target_length=param["dataset"]["max_target_length"],
            split_key=param["dataset"]["split_key"]
        )
    )

    # 编码和解码模板
    encoder_prompt = param["template"]["encoder_prompt"]
    decoder_prompt = param["template"]["decoder_prompt"]
    instruction_prompt = param["template"]["instruction_prompt"]

    # 推理参数
    remote_inference_kwargs = param["inference_params"]["remote"]
    local_inference_kwargs = param["inference_params"]["local"]

    # DeepSpeed 配置
    ds_config = param["deepspeed_config"]

    # 训练参数
    training_args_dict = dict(
        per_device_train_batch_size=param["training"]["batch_size"],
        gradient_accumulation_steps=param["training"]["gradient_accumulation_steps"],
        logging_steps=param["training"]["logging_steps"],
        max_steps=param["training"]["max_steps"],
        deepspeed=ds_config,
        fp16=True,
        log_level=param["training"]["log_level"]
    )

    # 设置客户端和服务器模型的配置
    client_conf = dict(
        model_conf=model_conf,
        dataset_conf=dataset_conf,
        training_args_conf=training_args_dict,
        data_collator_conf=data_collator_conf,
        mode=param["mode"],
        infer_inst_init_conf=infer_init_conf_client,
        encode_template=encoder_prompt,
        instruction_template=instruction_prompt,
        decode_template=decoder_prompt,
        remote_inference_kwargs=remote_inference_kwargs,
        local_inference_kwargs=local_inference_kwargs,
        perturb_doc_key='perturbed_doc',
        perturbed_response_key='perturbed_response',
        result_key='infer_result'
    )

    server_conf = dict(
        infer_inst_init_conf=infer_init_conf_server,
        mode=param["mode"]
    )

    # Initialize HomoNN component
    homo_nn_0 = HomoNN(
        'nn_0',
        train_data=reader_0.outputs["output_data"],
        runner_module=params['pipeline']['runner_module'],
        runner_class=params['pipeline']['runner_class']
    )

    homo_nn_0.guest.task_parameters(runner_conf=client_conf)
    homo_nn_0.arbiter.task_parameters(runner_conf=server_conf)

    homo_nn_0.guest.conf.set("launcher_name", "deepspeed")

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
                        help="config file for params", default="./pdss_config.yaml")
    args = parser.parse_args()
    main(args.config, args.param)