#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import os
import pickle
import time
from torch import nn
from typing import List, Optional, Callable, Literal, Union
from fate.arch import Context
from torch.utils.data import DataLoader, Dataset
from transformers.trainer_callback import TrainerCallback
from transformers import PreTrainedTokenizer
import logging
import torch
import torch.distributed as dist
from fate_llm.dataset.pdss_dataset import PrefixDataset
from transformers.modeling_utils import unwrap_model
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Dict, Any
from transformers import Seq2SeqTrainingArguments 
from transformers.trainer_utils import EvalPrediction
from fate_llm.trainer.seq2seq_trainer import Seq2SeqTrainer, Seq2SeqTrainingArguments
from fate_llm.inference.inference_base import Inference
from fate_llm.algo.inferdpt.inferdpt import InferDPTClient, InferDPTServer
from fate_llm.algo.pdss.encoder_decoder.slm_encoder_decoder import SLMEncoderDecoderClient, SLMEncoderDecoderServer


logger = logging.getLogger(__name__)
_MODE = ['train_only', 'infer_only', 'infer_and_train']


# share obj between ranks in an easy way
def save_to(obj, filepath, filename='tmp.pkl'):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    path = filepath + filename
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    dist.barrier()
    os.remove(path)


def load(filepath, filename='tmp.pkl'):
    path = filepath + filename
    while not os.path.exists(path):
        time.sleep(0.1)  
    while True:
        try:
            with open(path, 'rb') as f:
                d = pickle.load(f)
                break
        except (EOFError, pickle.UnpicklingError):
            time.sleep(0.1) 

    dist.barrier()
    return d


class DSSTrainerClient(Seq2SeqTrainer):

    def __init__(self,
                model: nn.Module,
                training_args: Seq2SeqTrainingArguments,
                train_set: Dataset,
                val_set: Dataset = None,
                alpha: float = 0.5,
                optimizer: torch.optim.Optimizer = None,
                data_collator: Callable = None,
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                tokenizer: Optional[PreTrainedTokenizer] = None,
                callbacks: Optional[List[TrainerCallback]] = [],
                compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    ) -> None:

        self.alpha = alpha
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

    def compute_loss(self, model, inputs, return_outputs=False):

        label_outputs = model(**inputs['predict'])
        cot_outputs = model(**inputs['rationale'])
        loss = self.alpha * cot_outputs.loss + (1. - self.alpha) * label_outputs.loss
        return (loss, {'rationale_loss': cot_outputs, 'predict_loss': label_outputs}) if return_outputs else loss


class PDSSTrainerClient(DSSTrainerClient):

    def __init__(self,
        ctx: Context,
        training_args: Seq2SeqTrainingArguments,
        train_set: PrefixDataset,
        val_set: Dataset = None,
        model: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        data_collator: Callable = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        callbacks: Optional[List[TrainerCallback]] = [],
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        alpha: float = 0.5,
        mode: Literal['train_only', 'infer_only', 'infer_and_train'] = 'infer_and_train',
        infer_client: Union[SLMEncoderDecoderClient, InferDPTClient] = None,
        encode_template: str = None,
        instruction_template: str = None,
        decode_template: str = None,
        result_key: str = 'infer_result',
        verbose: bool = False,
        remote_inference_kwargs: dict = {},
        local_inference_kwargs: dict = {},
        tmp_data_share_path: str = None
    ) -> None:
        
        self.mode = mode
        self.infer_client = infer_client
        self.infer_result = None
        self.infer_predict_kwargs = {
            'encode_template': encode_template,
            'instruction_template': instruction_template,
            'decode_template': decode_template,
            'result_key': result_key,
            'verbose': verbose,
            'remote_inference_kwargs': remote_inference_kwargs,
            'local_inference_kwargs': local_inference_kwargs
        }
        self.infer_result = None
        self.tmp_data_share_path = tmp_data_share_path

        assert mode in _MODE, "mode should be one of {}".format(_MODE)
        if training_args.local_rank == 0:
            if mode == 'infer_only' or mode == 'infer_and_train':
                if self.infer_client is None:
                    raise ValueError('You must provide an inference instance for remote inference')

        if mode != 'infer_only':
            training_args.remove_unused_columns = False  # this parameter is neccessary
            DSSTrainerClient.__init__(
                self,
                model=model,
                training_args=training_args,
                train_set=train_set,
                val_set=val_set,
                data_collator=data_collator,
                optimizer=optimizer,
                scheduler=scheduler,
                tokenizer=tokenizer,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
                alpha=alpha
            )
        else:
            # skip trainer initialzation becuase training is not needed
            self.args = training_args
            self.train_dataset = train_set

    def infer(self) -> List[str]:        

        if self.args.local_rank == 0:  # other rank will skip federation step
            assert isinstance(self.train_dataset, PrefixDataset), "train_set should be an instance of PrefixDataset"
            dict_dataset = self.train_dataset.get_raw_dataset()
            infer_result = self.infer_client.inference(dict_dataset, **self.infer_predict_kwargs)
            self.infer_result = infer_result
            rationale_list = [i[self.infer_predict_kwargs['result_key']] for i in self.infer_result]
            self.train_dataset.load_rationale(rationale_list, key=self.infer_predict_kwargs['result_key'])
            logger.info('infer done')
            if self.mode == 'infer_and_train':
                if self.args.world_size > 1:  # sync dataset with other ranks
                    tmp_path = self.tmp_data_share_path if self.tmp_data_share_path is not None else self.args.output_dir
                    logger.info('scattering obj, save to temp path {}'.format(tmp_path))
                    save_to(rationale_list, tmp_path)

        if self.args.local_rank > 0:
            if self.mode == 'infer_and_train':
                # wait until infer is done
                tmp_path = self.tmp_data_share_path if self.tmp_data_share_path is not None else self.args.output_dir
                logger.info('waiting for obj, load frm temp path {}'.format(tmp_path))
                rationale_list = load(tmp_path)
                self.train_dataset.load_rationale(rationale_list)
                logger.info('Rationale loaded')

    def train(self):

        if self.mode == 'train_only':
            logger.info("Train only mode")
            super().train()
        elif self.mode == 'infer_only':
            logger.info("infer only mode, skip training")
            self.infer()
        elif self.mode == 'infer_and_train':
            logger.info("infer and train mode")
            self.infer()
            super().train() 

    def get_infer_result(self):
        return self.infer_result


class PDSSTraineServer(object):

    def __init__(self, ctx: Context, infer_server: Union[SLMEncoderDecoderServer, InferDPTServer]):
        super().__init__()
        self.ctx = ctx
        self.infer_server = infer_server

    def train(self):
        logger.info('Server side start inference')
        self.infer_server.inference()
        logger.info('Server inference done')


if __name__ == '__main__':
    pass
