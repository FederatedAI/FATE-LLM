from dataclasses import dataclass, field

from transformers import TrainingArguments
from transformers.utils import add_start_docstrings


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class KSeedTrainingArguments(TrainingArguments):
    """
    TrainingArguments is the subset of the arguments we use in our example scripts, they are the arguments that

    Parameters:
        optim: optional, default is KSeedZO
            The optimizer to use.
        eps: optional, default is 0.0005
            Epsilon value for KSeedZerothOrderOptimizer.
        grad_clip: optional, default is -100.0
            Gradient clip value for KSeedZerothOrderOptimizer.
    """

    zo_optim: bool = field(
        default=True,
        metadata={"help": "Whether to use KSeedZerothOrderOptimizer. This suppress `optim` argument when True."},
    )
    k: int = field(
        default=4096,
        metadata={"help": "The number of seed candidates to use. This suppress `seed_candidates` argument when > 1."},
    )
    eps: float = field(default=0.0005, metadata={"help": "Epsilon value for KSeedZerothOrderOptimizer."})
    grad_clip: float = field(default=-100.0, metadata={"help": "Gradient clip value for KSeedZerothOrderOptimizer."})
