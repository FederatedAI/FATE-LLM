from fate_llm.dataset.input_output_dataset import InputOutputDataset
from transformers.trainer_pt_utils import LabelSmoother
from typing import List, Dict, Union, Literal
import logging
from jinja2 import Template
from transformers import AutoTokenizer



logger = logging.getLogger(__name__)


class PrefixDataset(InputOutputDataset):

    def __init__(self, 
                tokenizer_path,
                predict_template_input: str,
                predict_template_output: str,
                rationale_template_input: str,
                rationale_template_output: str,
                max_input_length: int = 256, 
                max_target_length: int = 256,
                load_from: Literal['jsonl', 'hf_load_from_disk', 'hf_load_dataset'] = 'hf_load_from_disk',
                split_key: str = None
                ):

        super().__init__(tokenizer_path, predict_template_input, predict_template_output, max_input_length, max_target_length, load_from, split_key)
        self.r_input_template = Template(rationale_template_input)
        self.r_output_template = Template(rationale_template_output)

    def load_rationale(self, result_list):
        for d, r in zip(self.dataset, result_list):
            d['rationale'] = r

    def get_str_item(self, i) -> dict:

        data_item = self.dataset[i]
        p_in = self.input_template.render(data_item)
        p_out = self.output_template.render(data_item)
        r_in = self.r_input_template.render(data_item)
        r_out = self.r_output_template.render(data_item)
        ret_dict = {
            'predict':{
                'input': p_in,
                'output': p_out
            },
            'rationale':{
                'input': r_in,
                'output': r_out
            }
        }
        return ret_dict
    
    def get_tokenized_item(self, i) -> dict:   

        str_item = self.get_str_item(i)
        ret_dict = {
            'predict': self._process_item(str_item['predict']),
            'rationale': self._process_item(str_item['rationale'])
        }

        return ret_dict
