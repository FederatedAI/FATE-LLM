from transformers import DataCollatorForSeq2Seq 
from transformers import AutoTokenizer
import pandas as pd

class PrefixDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        cot = super().__call__(list(features_df['predict']), return_tensors)
        label = super().__call__(list(features_df['rationale']), return_tensors)

        return {
            'predict': cot,
            'rationale': label
        }


def get_prefix_data_collator(tokenizer_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    data_collator = PrefixDataCollator(tokenizer)
    return data_collator
