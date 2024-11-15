import argparse
import yaml
from fate_client.pipeline.components.fate.reader import Reader
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate.nn.loader import Loader
from fate_client.pipeline.components.fate.homo_nn import HomoNN
from fate_client.pipeline.utils import test_utils

def load_params(file_path):
    """Load and parse the YAML config file."""
    with open(file_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def setup_pipeline(params):
    """Set up the pipeline based on the provided parameters."""
    guest = params['pipeline']['guest']
    arbiter = params['pipeline']['arbiter']
    
    # Create the pipeline
    pipeline = FateFlowPipeline().set_parties(guest=guest, arbiter=arbiter)
    
    # Initialize Reader component
    reader_0 = Reader("reader_0", runtime_parties=dict(guest=guest))
    reader_0.guest.task_parameters(
        namespace=params['pipeline']['namespace'],
        name=params['pipeline']['name']
    )
    
    # Configuration loading from YAML
    model_conf = Loader(
        module_name=params['models']['module_name'],
        item_name=params['models']['item_name'],
        pretrained_model_name_or_path=params['models']['pretrained_model_path']
    ).to_dict()

    data_collator_conf = Loader(
        module_name=params['data_collator']['module_name'],
        item_name=params['data_collator']['item_name'],
        tokenizer_name_or_path=params['data_collator']['tokenizer_path']
    ).to_dict()

    infer_init_conf_client = {
        'module_name': params['infer_client']['module_name'],
        'item_name': params['infer_client']['item_name'],
        'kwargs': params['infer_client']['kwargs']
    }

    infer_init_conf_server = {
        'module_name': params['infer_server']['module_name'],
        'item_name': params['infer_server']['item_name'],
        'kwargs': params['infer_server']['kwargs']
    }

    dataset_conf = {
        'module_name': params['dataset']['module_name'],
        'item_name': params['dataset']['item_name'],
        'kwargs': params['dataset']['kwargs']
    }

    # Mode of operation
    mode = params['pipeline']['mode']

    client_conf = {
        'model_conf': model_conf,
        'dataset_conf': dataset_conf,
        'mode': mode,
        'infer_inst_init_conf': infer_init_conf_client
    }

    server_conf = {
        'infer_inst_init_conf': infer_init_conf_server,
        'mode': mode
    }

    # Initialize HomoNN component
    homo_nn_0 = HomoNN(
        'nn_0',
        train_data=reader_0.outputs["output_data"],
        runner_module=params['pipeline']['runner_module'],
        runner_class=params['pipeline']['runner_class']
    )

    homo_nn_0.guest.task_parameters(runner_conf=client_conf)
    homo_nn_0.arbiter.task_parameters(runner_conf=server_conf)

    # Add tasks to the pipeline
    pipeline.add_tasks([reader_0, homo_nn_0])
    pipeline.conf.set("task", dict(engine_run=params['pipeline']['engine_run']))
    pipeline.compile()
    pipeline.fit()

def main(config_file):
    params = load_params(config_file)
    setup_pipeline(params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("PDSS PIPELINE DEMO")
    parser.add_argument("--config", type=str, default="./pdss_config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(config_file=args.config)
