## FedKSeed

The Algorithm is based on the paper: [Federated Full-Parameter Tuning of Billion-Sized Language Models
with Communication Cost under 18 Kilobytes](https://arxiv.org/pdf/2312.06353.pdf) and the code is adaptor
from the https://github.com/alibaba/FederatedScope/tree/FedKSeed.
We refactor the code to make it more compatible with (transformers/PyTorch) framework 
and integrate it into the FATE-LLM framework.

The main works include:
1. An KSeedZerothOrderOptimizer class that can be used to optimize model along given direction that generated with random seed.
2. An KSeedZOExtendedTrainer subclass of Trainer from transformers that can be used to train large language models with KSeedZerothOrderOptimizer.
3. Trainers for federated learning with large language models.