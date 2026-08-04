"""Optimizer and learning-rate schedule, mirroring Lior's notebook cell 18 exactly.

Budget parity is one of the fairness constraints: both arms train for 100 epochs at batch
256 with the *same* optimizer and the *same* schedule, so the only thing that differs
between them is the pretraining objective (ELBO vs NT-Xent). Any drift here -- a different
warmup shape, a different weight-decay split -- would show up as an accuracy difference
that has nothing to do with the research question.

The one deliberate change from Lior's version is that the constants are arguments instead
of module globals, since the beta ablation reruns this with the same numbers many times and
implicit globals make that hard to audit. The arithmetic is character-for-character his.

Device-agnostic: the optimizer is built from whatever parameters the model already holds,
so it follows the model to CPU or CUDA without this module knowing which.
"""

import math

import torch


def build_optimizer(model, weight_decay, learning_rate=0.0):
    """AdamW with weight decay applied only to multi-dimensional parameters.

    The ndim == 1 test is Lior's, and it is the standard trick: every BatchNorm weight and
    bias, and every conv/linear bias, is a 1-D tensor, so this one condition excludes all of
    them in a single pass. Decaying a BatchNorm scale toward zero fights the normalization
    it is supposed to provide, and decaying biases just adds noise -- neither has the
    overfitting story that motivates weight decay on a weight matrix.

    Group order (decay first, no-decay second) matches his so the two optimizers consume
    RNG and step state in the same order.

    Initialized at lr = 0.0 on purpose: the schedule below sets the learning rate explicitly
    before every single step, so any value here would be overwritten and a nonzero one would
    only be misleading.
    """
    decay_parameters = []
    no_decay_parameters = []

    for parameter in model.parameters():
        if parameter.ndim == 1:
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0}
        ],
        lr=learning_rate,
    )


def learning_rate_at_step(step, total_steps, warmup_steps, max_lr, min_lr):
    """Linear warmup then cosine decay, evaluated per OPTIMIZER STEP, not per epoch.

    Per-step matters: with ~390 steps per epoch, a per-epoch schedule would hold the
    learning rate flat across each epoch and produce a visibly different training curve from
    Lior's, breaking budget parity even though the epoch count matched.

    Warmup uses (step + 1) / warmup_steps so the very first step already takes a nonzero
    step rather than a wasted zero-length one, and so the warmup ends exactly at max_lr on
    step warmup_steps - 1.

    The cosine denominator is max(total_steps - warmup_steps - 1, 1): the -1 makes the final
    step land exactly on min_lr instead of just short of it, and the max(..., 1) guards the
    degenerate case where warmup covers the whole run and the denominator would otherwise be
    zero or negative.
    """
    if step < warmup_steps:
        warmup_progress = (step + 1) / warmup_steps
        return max_lr * warmup_progress

    cosine_denominator = max(total_steps - warmup_steps - 1, 1)
    # Clamped at 1.0 so an extra step past total_steps returns min_lr rather than climbing
    # back up the far side of the cosine.
    cosine_progress = min((step - warmup_steps) / cosine_denominator, 1.0)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
    return min_lr + (max_lr - min_lr) * cosine_factor


def set_learning_rate(optimizer, learning_rate):
    """Write one learning rate into every parameter group.

    Both groups share a schedule -- they differ only in weight decay -- so this is a plain
    loop rather than a torch LRScheduler. Hand-setting it also keeps the schedule readable
    and loggable: the training loop records the first and last learning rate of each epoch.
    """
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
