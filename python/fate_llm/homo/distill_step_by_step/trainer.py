from torch import nn
from fate.ml.aggregator.base import Aggregator
from fate_llm.homo.fedavg import Seq2SeqTrainingArguments
from dataclasses import dataclass
from typing import List, Optional, Callable
from fate.arch import Context
from torch.utils.data import DataLoader, Dataset
from transformers.trainer_callback import TrainerCallback
from transformers import PreTrainedTokenizer
import logging
import torch
from fate_llm.dataset.dss_dataset import PrefixDataset
from transformers.modeling_utils import unwrap_model
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Dict, Any
from fate.ml.nn.trainer.trainer_base import HomoTrainerServer
from transformers import Seq2SeqTrainingArguments as Seq2SeqTrainer
from transformers.trainer_utils import EvalPrediction


logger = logging.getLogger(__name__)



class DSSTrainerClient(Seq2SeqTrainer):

    def __init__(self,
                ctx: Context,
                model: nn.Module,
                training_args: Seq2SeqTrainingArguments,
                train_set: Dataset,
                alpha: float = 0.8,
                annotate_only: bool = False,
                annotation_postprocess_func: Callable[[List[str]], List[str]] = None,
                val_set: Dataset = None,
                optimizer: torch.optim.Optimizer = None,
                data_collator: Callable = None,
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                tokenizer: Optional[PreTrainedTokenizer] = None,
                callbacks: Optional[List[TrainerCallback]] = [],
                compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        
        self.ctx = ctx
        self.annotate_only = annotate_only
        self.alpha = alpha
        self.train_set = train_set
        self.annotation_postprocess_func = annotation_postprocess_func

        Seq2SeqTrainer.__init__(
            self,
            model=model,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=val_set,
            data_collator=data_collator,
            optimizers=(optimizer, scheduler),
            tokenizer=tokenizer,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

    def fed_data_annotation(self) -> List[str]:
        
        assert isinstance(self.train_set, PrefixDataset), "train_set should be an instance of PrefixDataset"
        prompt_texts = []
        for i in range(len(self.train_set)):
            prompt_texts.append(self.train_set.get_rationale_generate_prompt(i))
        

    def train(self):

        if self.annotate_only:
            return  
        else:
            super().train()

    def compute_loss(self, model, inputs, return_outputs=False):
        pred_outputs = model(**inputs['pred'])
        expl_outputs = model(**inputs['expl'])

        loss = self.alpha * pred_outputs.loss + (1. - self.alpha) * expl_outputs.loss

        return (loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else loss


class DSSTraineServer(HomoTrainerServer):

    def __init__(self, ctx: Context, 
                 model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 generate_func: Callable[[PreTrainedModel, PreTrainedTokenizer, str]],
                 preproecess_dataset_func: Callable[[PreTrainedModel, PreTrainedTokenizer, str]] = None,
                 postprocess_annoated_func: Callable[[PreTrainedModel, PreTrainedTokenizer, str]] = None
                 ):
        
        super().__init__(ctx, local_mode=False)
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_to_annoate = None
        self.aggregator = None
        self.generate_func = generate_func
        self.preprocess_dataset_func = preproecess_dataset_func
        self.postprocess_annoated_func = postprocess_annoated_func

    def on_train_begin(self, ctx: Context, aggregator: Aggregator):
        dataset_to_annotate: List[str] = ctx.guest.get('dataset_to_process')
        self.dataset_to_annoate = dataset_to_annotate
        logger.info(f"Dataset to annotate, data num: {len(self.dataset_to_annoate)}")

    def train(self):

        if self.local_mode:
            logger.info("Local model is set, skip initializing fed setting & aggregator")
            return

        self.on_init_end(self.ctx, aggregator=self.aggregator)
        self.on_train_begin(self.ctx, aggregator=self.aggregator)

        self.dataset_to_annoate = self.preprocess_dataset_func(self.model, self.tokenizer, self.dataset_to_annoate)

        # data annotation
        generate_result = []
        for q in self.dataset_to_annoate:
            res = self.generate_func(self.model, self.tokenizer, q)
            generate_result.append(res)

        generate_result = self.preprocess_dataset_func(self.model, self.tokenizer, generate_result)

        if self.dataset_to_annoate is None:
            raise ValueError("No dataset to annotate")
        
        self.on_train_end(self.ctx, aggregator=self.aggregator)
    
    def predict(self):
        return super().predict()


if __name__ == '__main__':
    pass