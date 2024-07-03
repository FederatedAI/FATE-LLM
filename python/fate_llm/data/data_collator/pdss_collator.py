from transformers import DataCollatorForSeq2Seq 
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