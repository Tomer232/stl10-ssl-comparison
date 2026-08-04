"""CNN classifier trained on the pseudo-labels produced by label propagation.

This is the last stage of Tomer's arm: VAE -> frozen mu -> KNN graph -> label
propagation -> *this*. Label propagation hands us a label for (almost) every one
of the 105k graph nodes; this module trains an SE-ResNet on those pseudo-labels
and scores it on the 8000 test images.

    KNOWN CONFOUND -- READ THIS BEFORE COMPARING ANY NUMBER FROM HERE
    TO LIOR'S SIMCLR ARM.

    Lior's downstream evaluation is a LINEAR PROBE on frozen 512-d features.
    Ours, in this file, is a full CNN trained on ~105k pseudo-labelled images.
    Those are not the same estimator and they do not have the same capacity, so
    a straight "our accuracy vs his accuracy" comparison measures two things at
    once: the quality of the latent space (the actual research question) and the
    capacity of the classifier sitting on top of it (a confound). A CNN can win
    this table while the generative latent space is the worse one, and it can
    lose it while being the better one.

    The honest apples-to-apples row is `src/probe.py` -- the same hand-rolled
    linear probe, on OUR embeddings, under the same 10-fold protocol. The report
    must LEAD with that row. The number this module produces is a second,
    clearly-labelled result answering a different and narrower question: how far
    does the full pseudo-labelling pipeline get end to end, given that it can
    train on 20x more (noisily labelled) images than the probe ever sees.

    Do not quietly merge the two into one column.

Every knob that affects budget parity -- optimizer, weight-decay split, warmup
plus cosine schedule -- is imported from `src/optim.py` rather than re-derived
here, so this classifier trains on the same schedule as the VAE pretraining and
as Lior's encoder.

Device-agnostic: the device is taken from the model's parameters or passed in
explicitly, so the same code path runs on the laptop CPU for tests and on the
lab 5090 for real.
"""

import numpy as np
import torch
import torch.nn as nn

try:
    # How the notebooks and scripts import the project: src/ itself on sys.path.
    import data
    import metrics
    import optim
    import trunk
except ImportError:
    # How a test runner at the repo root imports it.
    from src import data, metrics, optim, trunk


NUMBER_OF_CLASSES = 10

# Label propagation marks a node it could never reach -- one sitting in a
# connected component with no labeled seed -- with this sentinel. Such a node has
# no pseudo-label at all, only the arbitrary argmax of an all-zero row, and it
# must never contribute a gradient.
UNLABELED_SENTINEL = -1


class PseudoLabelClassifier(nn.Module):
    """The shared SE-ResNet trunk plus a linear classification head.

    Deliberately the *same* `SEResNetEncoder` as the VAE and as Lior's SimCLR
    arm. Swapping in a different backbone here would mean the end-to-end number
    reflects a different architecture as well as a different pretraining
    objective, and the trunk is the one thing the fairness protocol pins down.

    `freeze_encoder` supports two genuinely different experiments and both are
    worth running:
      * frozen -- the head is a linear classifier on fixed features, which
        isolates how much class structure the VAE's trunk already encodes;
      * unfrozen -- the whole network fine-tunes on the pseudo-labels, which is
        the realistic "use the pipeline for its intended purpose" setting.
    """

    def __init__(self, num_classes=NUMBER_OF_CLASSES, freeze_encoder=False):
        super().__init__()

        self.encoder = trunk.SEResNetEncoder()
        # Same reason as in src/vae.py: without this the trunk starts from
        # PyTorch's default kaiming-uniform rather than Lior's kaiming-normal
        # fan_out / xavier scheme. It is overwritten wholesale by
        # `from_pretrained_encoder`, so it only matters for the random-init
        # control run -- which is exactly the run where the two arms have to
        # start from the same distribution for the comparison to mean anything.
        trunk.initialize_encoder_weights(self.encoder)

        self.classification_head = nn.Linear(trunk.ENCODER_DIM, num_classes)
        self.encoder_is_frozen = False

        if freeze_encoder:
            self.set_encoder_frozen(True)

    def set_encoder_frozen(self, frozen):
        """Freeze or unfreeze the trunk.

        Turning off `requires_grad` is only half the job. A BatchNorm layer keeps
        updating its running mean and variance in training mode even with no
        gradients flowing, so a "frozen" encoder would still drift and produce
        different features at the end of training than at the start. `train()`
        below therefore forces the encoder to stay in eval mode while it is
        frozen, which is what actually makes the features fixed.
        """
        self.encoder_is_frozen = bool(frozen)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not self.encoder_is_frozen
        if self.encoder_is_frozen:
            self.encoder.eval()
        return self

    def train(self, mode=True):
        """Standard train/eval toggle, except a frozen encoder stays in eval."""
        super().train(mode)
        if self.encoder_is_frozen:
            self.encoder.eval()
        return self

    def forward(self, images):
        return self.classification_head(self.encoder(images))

    @classmethod
    def from_pretrained_encoder(cls, state_dict, num_classes=NUMBER_OF_CLASSES,
                                freeze_encoder=False, strict=True):
        """Build a classifier whose trunk starts from an already-trained encoder.

        The intended input is the VAE run's trained trunk: the classifier should
        not start from scratch when we have just spent 100 epochs learning a
        representation. It also accepts Lior's checkpoint unchanged, which is a
        useful sanity check -- `strict=True` will only succeed if the two trunks
        really are the same network, so a successful load is itself evidence for
        the trunk-parity claim in the report.

        Accepts either a bare encoder state dict or a wrapper state dict whose
        keys are prefixed with "encoder." (which is what saving the whole VAE
        model produces). The prefix is stripped rather than requiring callers to
        do it, because getting that wrong silently gives a randomly-initialized
        trunk with `strict=False` -- the kind of bug that costs a day.
        """
        model = cls(num_classes=num_classes, freeze_encoder=False)

        if any(key.startswith("encoder.") for key in state_dict):
            encoder_state = {
                key[len("encoder."):]: value
                for key, value in state_dict.items()
                if key.startswith("encoder.")
            }
        else:
            encoder_state = state_dict

        model.encoder.load_state_dict(encoder_state, strict=strict)

        if freeze_encoder:
            model.set_encoder_frozen(True)

        return model


def _infer_device(model, device):
    """Use the caller's device if given, otherwise follow the model's parameters."""
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def _usable_positions(pseudo_labels, num_classes, ignore_label):
    """Positions of the samples that may enter the loss.

    Filtering ONCE, up front, rather than masking inside the training loop. Two
    reasons, and both matter:
      * a masked-out sample that still reaches the loss can only be neutralised
        by multiplying its term by zero, and a single NaN in an ignored row would
        propagate through that multiplication anyway;
      * the schedule in optim.learning_rate_at_step is defined over the total
        number of optimizer steps, and that count has to be derived from the
        number of samples we ACTUALLY train on. Sampling from the filtered index
        keeps the epoch length honest and every batch full.
    """
    label_array = np.asarray(pseudo_labels).reshape(-1)
    usable = np.flatnonzero(label_array != ignore_label).astype(np.int64)

    if usable.size == 0:
        raise ValueError(
            "every pseudo-label is the " + str(ignore_label) + " sentinel -- label "
            "propagation reached no node, so there is nothing to train on"
        )

    kept = label_array[usable]
    if kept.min() < 0 or kept.max() >= num_classes:
        raise ValueError(
            "pseudo-labels must lie in [0, " + str(num_classes) + ") or equal the "
            + str(ignore_label) + " sentinel, got range [" + str(int(kept.min()))
            + ", " + str(int(kept.max())) + "]"
        )

    return usable


def _gather_image_batch(images, positions):
    """Index into either a numpy array (possibly a memmap view) or a torch tensor."""
    if isinstance(images, torch.Tensor):
        return images[torch.as_tensor(positions, device=images.device)]
    return images[positions]


def train_classifier(model, images, pseudo_labels, normalization_mean,
                     normalization_standard_deviation, device=None, sample_weights=None,
                     epochs=100, batch_size=256, max_learning_rate=5e-4,
                     min_learning_rate=1e-5, warmup_epochs=10, weight_decay=1e-4,
                     seed=42, num_classes=NUMBER_OF_CLASSES,
                     ignore_label=UNLABELED_SENTINEL, epoch_callback=None):
    """Train `model` on pseudo-labelled images with cross-entropy. Returns a history list.

    `images` is raw uint8 [N, 3, 96, 96] -- a numpy array, a memmap view of the
    unlabeled split, or a uint8 tensor already resident in VRAM (which is the
    plan on the server: 2.76 GB fits, and it removes the DataLoader entirely).
    Normalization is applied per batch by `data.normalize_batch`, with the
    constants computed from the unlabeled split, so the classifier sees exactly
    the same pixel distribution as the VAE and as Lior's encoder.

    TWO PROPERTIES SPECIFIC TO TRAINING ON PROPAGATED LABELS:

    1. `ignore_label` (-1 by default) marks nodes label propagation could not
       reach. They are dropped before the first epoch and never sampled, so they
       cannot contribute a gradient. Training on them would be training on noise
       -- their "label" is the argmax of an all-zero row.

    2. `sample_weights` is the optional per-sample confidence from label
       propagation (typically the winning class's mass in the converged F matrix).
       A node two hops from a single seed is a much weaker supervision signal
       than a seed itself, and weighting the loss by confidence is the cheapest
       way to say so. The batch loss is
           sum_i w_i * cross_entropy_i / sum_i w_i,
       i.e. normalized by the WEIGHT sum rather than the sample count. That keeps
       the gradient magnitude independent of how confident a given batch happens
       to be; dividing by the count instead would make low-confidence batches take
       systematically smaller steps and would silently interact with the learning
       rate schedule.

    The optimizer and the per-step warmup+cosine schedule come from src/optim.py
    unchanged, which is what keeps this run inside the project's budget-parity
    story.
    """
    device = _infer_device(model, device)
    model.to(device)

    usable = _usable_positions(pseudo_labels, num_classes, ignore_label)
    number_of_samples = usable.size

    label_tensor = torch.as_tensor(
        np.asarray(pseudo_labels).reshape(-1)[usable], dtype=torch.int64, device=device)

    if sample_weights is None:
        weight_tensor = None
    else:
        weight_array = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if weight_array.shape[0] != np.asarray(pseudo_labels).reshape(-1).shape[0]:
            raise ValueError(
                "sample_weights must have one entry per pseudo-label, got "
                + str(weight_array.shape[0])
            )
        weight_tensor = torch.as_tensor(
            weight_array[usable], dtype=torch.float32, device=device)
        if float(weight_tensor.min().item()) < 0.0:
            raise ValueError("sample_weights must be non-negative")

    steps_per_epoch = max(number_of_samples // batch_size, 1)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(steps_per_epoch * warmup_epochs, 1)

    optimizer = optim.build_optimizer(model, weight_decay)

    # One generator for the whole run rather than one per epoch, so the sequence
    # of shuffles is reproducible from `seed` alone and does not shift if the
    # epoch count changes.
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)

    history = []
    global_step = 0

    for epoch in range(epochs):
        model.train()

        permutation = torch.randperm(number_of_samples, generator=shuffle_generator).numpy()

        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_seen = 0
        first_learning_rate = None
        last_learning_rate = None

        for step in range(steps_per_epoch):
            # Dropping the ragged tail keeps every step the same batch size, which
            # keeps the loss scale (and therefore the schedule) uniform.
            batch_slots = permutation[step * batch_size:(step + 1) * batch_size]
            if batch_slots.size == 0:
                continue

            batch_positions = usable[batch_slots]

            learning_rate = optim.learning_rate_at_step(
                global_step, total_steps, warmup_steps, max_learning_rate, min_learning_rate)
            optim.set_learning_rate(optimizer, learning_rate)
            if first_learning_rate is None:
                first_learning_rate = learning_rate
            last_learning_rate = learning_rate

            batch_images = data.normalize_batch(
                _gather_image_batch(images, batch_positions),
                normalization_mean,
                normalization_standard_deviation,
                device,
            )
            batch_labels = label_tensor[torch.as_tensor(batch_slots, device=device)]

            logits = model(batch_images)

            if weight_tensor is None:
                loss = torch.nn.functional.cross_entropy(logits, batch_labels)
            else:
                batch_weights = weight_tensor[torch.as_tensor(batch_slots, device=device)]
                per_sample_loss = torch.nn.functional.cross_entropy(
                    logits, batch_labels, reduction="none")
                # Clamped denominator: a batch that happens to be entirely
                # zero-confidence would otherwise divide by zero and NaN the run.
                loss = (per_sample_loss * batch_weights).sum() / torch.clamp(
                    batch_weights.sum(), min=1e-8)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_count = batch_labels.numel()
            epoch_loss_sum += float(loss.item()) * batch_count
            epoch_correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
            epoch_seen += batch_count
            global_step += 1

        epoch_record = {
            "epoch": epoch,
            "loss": epoch_loss_sum / max(epoch_seen, 1),
            "pseudo_label_accuracy": epoch_correct / max(epoch_seen, 1),
            "first_learning_rate": first_learning_rate,
            "last_learning_rate": last_learning_rate,
            "samples": epoch_seen,
        }
        history.append(epoch_record)

        # Lets the caller print, checkpoint or early-stop without this module
        # taking a dependency on how the run is orchestrated.
        if epoch_callback is not None:
            epoch_callback(epoch_record, model)

    return history


@torch.inference_mode()
def predict_probabilities(model, images, normalization_mean, normalization_standard_deviation,
                          device=None, batch_size=512):
    """Class probabilities for every image, [N, num_classes] float64 on CPU.

    Returned in float64 on CPU because the only consumer is the metrics module,
    which does all of its arithmetic there. `inference_mode` plus `model.eval()`
    is the deterministic path: BatchNorm uses its running statistics and nothing
    updates.

    Default batch_size is FEATURE_BATCH_SIZE (512) from Lior's notebook -- the
    same batching he uses for feature extraction.
    """
    device = _infer_device(model, device)
    model.to(device)
    model.eval()

    number_of_images = len(images)
    probability_batches = []

    for start in range(0, number_of_images, batch_size):
        end = min(start + batch_size, number_of_images)
        batch_images = data.normalize_batch(
            images[start:end], normalization_mean, normalization_standard_deviation, device)
        logits = model(batch_images)
        probability_batches.append(torch.softmax(logits, dim=1).to(torch.float64).cpu())

    return torch.cat(probability_batches, dim=0)


def evaluate(model, images, labels, normalization_mean, normalization_standard_deviation,
             device=None, batch_size=512, num_classes=NUMBER_OF_CLASSES):
    """Score the classifier. Returns the [num_classes, num_classes] confusion matrix.

    The confusion matrix rather than a dict of scalars, because that is the one
    primitive `src/metrics.py` is built around: accuracy, macro-F1 and per-class
    precision/recall are all read off the same table, so returning the table lets
    the caller compute whichever of them the report needs without re-running the
    model. Use `predict_probabilities` alongside it when the log-loss row is
    wanted too.

    THIS IS A TEST-SET FUNCTION AND THE TEST SET IS TOUCHED EXACTLY ONCE, at the
    very end, by both arms. Model selection happens on the development split.
    """
    probabilities = predict_probabilities(
        model, images, normalization_mean, normalization_standard_deviation,
        device=device, batch_size=batch_size)

    predictions = probabilities.argmax(dim=1)
    return metrics.confusion_matrix(labels, predictions, num_classes)
