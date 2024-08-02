from fate_llm.trainer.seq2seq_trainer import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer
import pandas as pd


class EDPrefixDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        a = super().__call__(list(features_df['encoder']), return_tensors)
        b = super().__call__(list(features_df['decoder']), return_tensors)

        return {
            'encoder': a,
            'decoder': b
        }


class EncoderDecoderPrefixTrainer(Seq2SeqTrainer):

    def __init__(self, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        out_a = model(**inputs['encoder'])
        out_b = model(**inputs['decoder'])
        loss = self.alpha * out_a.loss + (1. - self.alpha) * out_b.loss
        return (loss, {'out_a': out_a, 'out_b': out_b}) if return_outputs else loss
