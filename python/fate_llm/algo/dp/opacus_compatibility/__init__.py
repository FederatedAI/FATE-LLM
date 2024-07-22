from .grad_sample.embedding import compute_embedding_grad_sample
from .optimizers.optimizer import add_noise_wrapper


def add_layer_compatibility(opacus):
    replace_method = []
    for k, v in opacus.GradSampleModule.GRAD_SAMPLERS.items():
        if v.__name__ == "compute_embedding_grad_sample":
            replace_method.append(k)

    for k in replace_method:
        opacus.GradSampleModule.GRAD_SAMPLERS[k] = compute_embedding_grad_sample


def add_optimizer_compatibility(optimizer):
    add_noise_wrapper(optimizer)
