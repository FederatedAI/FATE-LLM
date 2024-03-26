from datasets.arrow_dataset import Dataset
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from typing import List, Dict, Union


class PrefixDataset(Dataset):

    def __init__(self, 
                 dataset: Union[List[Dict[str, str]], Dataset], 
                 tokenizer, 
                 pad_token: int = -100, 
                 standard_q_prefix: str='predict:\nQuestion:', 
                 rationale_q_prefix: str='explain:\nQuestion:',
                 standard_a_prefix: str='\nAnswer:',
                 rationale_a_prefix: str='\nRationale:'
                 ):
        
        super(PrefixDataset, self).__init__()

        self.tokenizer = tokenizer
        self.dataset = dataset
        self.standard_q_prefix = standard_q_prefix
        self.rationale_q_prefix = rationale_q_prefix
        self.standard_a_prefix = standard_a_prefix
        self.rationale_a_prefix = rationale_a_prefix

        if pad_token is None:
            self.pad_token = LabelSmoother.ignore_index
        else:
            self.pad_token = pad_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> dict:
        item = self.get_tokenized_item(i)
        return item

    def get_rationale_generate_prompt(self, i) -> str:
        pass

    def get_tokenized_item(self, i) -> dict:
        pass
    

class QwPrefixDataset(PrefixDataset):

    def __init__(self, 
                dataset: List[Dict[str, str]] | Dataset, 
                tokenizer,
                max_input_length: int = 256, 
                pad_token: int = -100, 
                few_shot_str: str = None,
                standard_q_prefix: str = 'predict:\nQuestion:', rationale_q_prefix: str = 'explain:\nQuestion:', 
                standard_a_prefix: str = '\nAnswer:', rationale_a_prefix: str = '\nRationale:'
                ):
        
        super().__init__(dataset, tokenizer, pad_token, standard_q_prefix, rationale_q_prefix, standard_a_prefix, rationale_a_prefix)
        self.max_input_length = max_input_length
        self.few_shot_str = few_shot_str

    def get_rationale_generate_prompt(self, i) -> str:
        prompt = '\nQuestion:' + self.dataset[i]['input'] + '\nAnswer:' + self.dataset[i]['label'] + '\nRationale:'
        if self.few_shot_str:
            return self.few_shot_str + prompt
        return prompt
    
    def __getitem__(self, i) -> dict:

        data_item = self.dataset[i]
        pred_model_inputs = self.tokenizer.encode(self.standard_q_prefix + data_item['input'] + self.standard_a_prefix, max_length=self.max_input_length, truncation=True)
        pred_label = self.tokenizer.encode(data_item['label'] + '\n', max_length=self.max_input_length, truncation=True)
        pred_input_ids = pred_model_inputs + pred_label
        pred_labels = len(pred_model_inputs) * [self.pad_token] + pred_label

        expl_model_inputs = self.tokenizer.encode(self.standard_q_prefix + data_item['input'] + self.standard_a_prefix, max_length=self.max_input_length, truncation=True)
        expl_label = self.tokenizer.encode(data_item['rationale'] + '\n', max_length=self.max_input_length, truncation=True)

        expl_input_ids = expl_model_inputs + expl_label
        expl_labels = len(expl_model_inputs) * [self.pad_token] + expl_label

        return {
            'aux_labels': expl_labels,
            'labels': pred_labels,
            'input_ids': pred_input_ids,
            'expl_input_ids': expl_input_ids
        }
