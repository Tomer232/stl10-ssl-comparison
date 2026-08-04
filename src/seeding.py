"""Reproducibility helpers, mirroring Lior's notebook so both arms are seeded identically.

Lior's cell 4 defines reset_seed / seeded_generator and his cell 2 turns on deterministic
cuDNN. We reproduce both. If only one arm were deterministic, any accuracy gap between the
generative and contrastive latent spaces could just be run-to-run variance, and the whole
comparison would be unfalsifiable.

The functions here take the seed explicitly rather than reading a module global, because
the fairness protocol calls for >= 3 seeds and the caller needs to vary it.
"""

import os
import random

import numpy as np
import torch


def reset_seed(seed):
    """Reseed every random source the pipeline touches.

    Same four calls, in the same order, as Lior's reset_seed. Called at the start of every
    training run and before every stochastic evaluation step, not just once at import --
    that is what makes a rerun of a single stage reproducible on its own.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_deterministic():
    """Force deterministic kernels, matching Lior's cell 2.

    cudnn.benchmark = False stops cuDNN from autotuning algorithms per input shape (the
    autotuner's choice varies between runs, and different algorithms give different
    floating-point rounding). use_deterministic_algorithms then makes PyTorch raise instead
    of silently falling back to a nondeterministic kernel, so a determinism regression
    fails loudly rather than quietly polluting the results.

    Costs some throughput. Worth it: the two arms have to be compared on equal footing.
    """
    # Deterministic cuBLAS needs this set before the CUDA context is created, otherwise
    # torch raises on the first matmul under use_deterministic_algorithms. Lior's Colab
    # runtime had it preset; on the lab server it does not, so we set it here and callers
    # should call enable_deterministic() before touching the GPU.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seeded_generator(seed):
    """A CPU torch.Generator for anything that takes an explicit generator.

    Shuffling and random splits are given their own generator rather than drawing from the
    global RNG, so inserting or removing an unrelated random call elsewhere in the pipeline
    does not shift which images land in which fold.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
