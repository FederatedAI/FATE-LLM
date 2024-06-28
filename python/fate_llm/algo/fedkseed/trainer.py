import logging
from typing import Dict, Union, Any, Tuple
from typing import Optional, List, Callable

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, EvalPrediction, DataCollator
from transformers import Trainer, TrainingArguments
from transformers.optimization import get_scheduler, SchedulerType
from transformers.trainer_callback import TrainerCallback

from fate_llm.algo.fedkseed.args import KSeedTrainingArguments
from fate_llm.algo.fedkseed.optimizer import KSeedZerothOrderOptimizer
from fate_llm.algo.fedkseed.pytorch_utils import get_optimizer_parameters_grouped_with_decay

logger = logging.getLogger(__name__)


class KSeedZOExtendedTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        training_args: TrainingArguments = None,
        kseed_args: "KSeedTrainingArguments" = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.kseed_args = kseed_args
        self._kseed_optimizer = None

        self._seed_candidates = None
        self._seed_probabilities = None

    def configure_seed_candidates(self, seed_candidates: torch.LongTensor, seed_probabilities: torch.FloatTensor):
        self._seed_candidates = seed_candidates
        self._seed_probabilities = seed_probabilities

    def get_directional_derivative_history(self):
        """
        hook to get the directional derivative history
        """
        if KSeedZOExtendedTrainer.k_seed_zo_mode(self.kseed_args):
            if self._kseed_optimizer is None:
                raise ValueError("KSeedZerothOrderOptimizer is not configured")
            return self._kseed_optimizer.directional_derivative_history
        else:
            raise ValueError("KSeedZerothOrderOptimizer is not configured")

    @staticmethod
    def k_seed_zo_mode(args):
        return hasattr(args, "zo_optim") and args.zo_optim

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        hook to do the step with KSeedZerothOrderOptimizer
        """
        if KSeedZOExtendedTrainer.k_seed_zo_mode(self.kseed_args):
            if self._kseed_optimizer is None:
                raise ValueError("KSeedZerothOrderOptimizer is not configured")

            model.eval()
            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                # zeroth order optimization needs forward pass twice in an optimization step,
                # so we need to wrap the forward pass in a closure
                def closure() -> torch.FloatTensor:
                    with torch.no_grad():
                        return self.compute_loss(model, inputs, return_outputs=False).detach()

            # we don't use step() method of KSeedZerothOrderOptimizer here
            # because `Trainer` wraps the optimizer that is subclass of `torch.optim.Optimizer` and
            # returns nothing from the step method
            with torch.no_grad():
                loss = self._kseed_optimizer.kseed_zeroth_order_step(closure=closure)
                return loss.detach()
        else:
            return super().training_step(model, inputs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        hook to add KSeedZerothOrderOptimizer
        """
        if KSeedZOExtendedTrainer.k_seed_zo_mode(self.kseed_args):

            if self._seed_candidates is None or self._seed_probabilities is None:
                raise ValueError("Seed candidates and probabilities are not configured.")

            optimizer_grouped_parameters = get_optimizer_parameters_grouped_with_decay(
                self.model, self.args.weight_decay
            )
            self.optimizer = KSeedZerothOrderOptimizer(
                optimizer_grouped_parameters,
                seed_candidates=self._seed_candidates,
                seed_probabilities=self._seed_probabilities,
                lr=self.args.learning_rate,
                eps=self.kseed_args.eps,
                weight_decay=self.args.weight_decay,
                grad_clip=self.kseed_args.grad_clip,
            )
            # we need to keep the reference to the original optimizer to use it in training_step
            self._kseed_optimizer = self.optimizer
            # if we use learning rate scheduler, we may need to preserve all updates instead of the aggregated one
            self.lr_scheduler = get_scheduler(
                name=SchedulerType.CONSTANT,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            super().create_optimizer_and_scheduler(num_training_steps)
