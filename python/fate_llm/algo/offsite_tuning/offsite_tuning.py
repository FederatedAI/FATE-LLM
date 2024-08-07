from fate.ml.aggregator.base import Aggregator
from fate_llm.algo.fedavg.fedavg import Seq2SeqFedAVGClient, Seq2SeqFedAVGServer, Seq2SeqTrainingArguments
from fate.ml.nn.trainer.trainer_base import FedArguments, TrainingArguments
from typing import List, Optional, Callable, Tuple
from fate.arch import Context
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler
from transformers.trainer_callback import TrainerCallback
from torch.nn import Module
from transformers import TrainerState, TrainerControl, PreTrainedTokenizer
from fate_llm.model_zoo.offsite_tuning.offsite_tuning_model import OffsiteTuningBaseModel
import logging
import torch
import torch.distributed as dist
from transformers.modeling_utils import unwrap_model


logger = logging.getLogger(__name__)


class OffsiteTuningTrainerClient(Seq2SeqFedAVGClient):
    
    def __init__(
        self,
        ctx: Context,
        model: OffsiteTuningBaseModel,
        training_args: Seq2SeqTrainingArguments,
        fed_args: FedArguments,
        train_set: Dataset,
        val_set: Dataset = None,
        optimizer: Optimizer = None,
        scheduler: _LRScheduler = None,
        data_collator: Callable = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        callbacks: List[TrainerCallback] = [],
        compute_metrics: Callable = None,
        aggregate_model: bool = False,
        save_trainable_weights_only: bool = False,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        assert isinstance(model, OffsiteTuningBaseModel), "model must be the subclass of OffsiteTuningBaseModel"
        if aggregate_model == False and fed_args is None:
            fed_args = FedArguments()
        elif fed_args is None:
            raise ValueError("fed_args must be provided when aggregate_model is True")

        local_mode = True if not aggregate_model else False
            
        super().__init__(
            ctx,
            model,
            training_args,
            fed_args,
            train_set,
            val_set,
            optimizer,
            scheduler,
            data_collator,
            tokenizer,
            callbacks,
            compute_metrics,
            local_mode,
            save_trainable_weights_only,
            preprocess_logits_for_metrics
        )
        self._aggregate_model = aggregate_model


    def _share_model(self, model, args: Seq2SeqTrainingArguments, sync_trainable_only=True):

        if args.local_rank == 0:
            for p in model.parameters():
                if (not sync_trainable_only) or (sync_trainable_only and p.requires_grad):
                    scatter_list = [p.data for _ in range(args.world_size)]
                    dist.scatter(p.data, scatter_list, async_op=False)
        else:
            for p in model.parameters():
                if (not sync_trainable_only) or (sync_trainable_only and p.requires_grad):
                    dist.scatter(p.data, src=0, async_op=False)

    def on_train_begin(self, ctx: Context, aggregator: Aggregator, fed_args: FedArguments, 
                       args: TrainingArguments, model: Module = None, optimizer: Optimizer = None, scheduler: _LRScheduler = None, 
                       dataloader: Tuple[DataLoader]= None, control: TrainerControl= None, 
                       state: TrainerState = None, **kwargs):
        
        if args.local_rank == 0: # master
            logger.info('receving weights from server')
            parameters_to_get = ctx.arbiter.get('sub_model_para')
            model = unwrap_model(model)
            model.load_submodel_weights(parameters_to_get)
            logger.info('received submodel weigths from the server')
            if args.world_size > 1:
                self._share_model(model, args)
                logger.info('sharing model parameters done')
        else:
            if args.world_size > 1:
                model = unwrap_model(model)
                self._share_model(model, args)
                logger.info('sharing model parameters done')

    def on_federation(
        self,
        ctx: Context,
        aggregator,
        fed_args: FedArguments,
        args: TrainingArguments,
        model: Optional[OffsiteTuningBaseModel] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        dataloader: Optional[Tuple[DataLoader]] = None,
        control: Optional[TrainerControl] = None,
        state: Optional[TrainerState] = None,
        **kwargs,
    ):
        if self._aggregate_model:
            aggregator.model_aggregation(ctx, model)


    def on_train_end(self, ctx: Context, aggregator: Aggregator, fed_args: FedArguments, 
                    args: TrainingArguments, model: OffsiteTuningBaseModel = None, optimizer: Optimizer = None, scheduler: _LRScheduler = None, 
                    dataloader: Tuple[DataLoader]= None, control: TrainerControl= None, 
                    state: TrainerState = None, **kwargs):

        if args.local_rank == 0:
            if args.world_size > 1:
                model = unwrap_model(model)
            return_weights = model.get_submodel_weights(with_emulator=False)
            ctx.arbiter.put('trained_sub_model_para', return_weights)
            logger.info('weights sent back to the server')

    def init_aggregator(self, ctx: Context, fed_args: FedArguments):
        if self._aggregate_model:
            return super().init_aggregator(ctx, fed_args)
        else:
            return None


class OffsiteTuningTrainerServer(Seq2SeqFedAVGServer):
    
    def __init__(self, ctx: Context, model: OffsiteTuningBaseModel, aggregate_model=False) -> None:
        self._aggregate_model = aggregate_model
        super().__init__(ctx, local_mode=False)
        assert isinstance(model, OffsiteTuningBaseModel), "model must be the subclass of OffsiteTuningBaseModel"
        self.model = model

    def on_train_begin(self, ctx: Context, aggregator: Aggregator):
        logger.info('sending weights to clients')
        parameters_to_send = self.model.get_submodel_weights()
        ctx.guest.put('sub_model_para', parameters_to_send)
        if any(p.role=='host' for p in ctx.parties):
            ctx.hosts.put('sub_model_para', parameters_to_send)

    def on_train_end(self, ctx: Context, aggregator: Aggregator):
        parameters_to_get = ctx.guest.get('trained_sub_model_para')
        self.model.load_submodel_weights(parameters_to_get, with_emulator=False)
        logger.info('received trained submodel weigths from the client')

    def on_federation(self, ctx: Context, aggregator, agg_iter_idx: int):
        if self._aggregate_model:
            aggregator.model_aggregation(ctx)
        else:
            logger.info('skip aggregation')

    def init_aggregator(self, ctx):
        if self._aggregate_model:
            return super().init_aggregator(ctx)
        else:
            return None
        
    def train(self):

        if self._aggregate_model:
            super().train()
        else:
            # do nothing but send the submodel weights to the client
            # and then aggregate the weights from the client
            self.on_init_end(self.ctx, aggregator=self.aggregator)
            self.on_train_begin(self.ctx, aggregator=self.aggregator)
            self.on_train_end(self.ctx, aggregator=self.aggregator)

    def save_model(
        self,
        output_dir: Optional[str] = None,
        state_dict=None
    ):
        import torch
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.model.state_dict(), output_dir + '/pytorch_model.bin')
