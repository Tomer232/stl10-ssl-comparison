"""On-GPU augmentation for the VAE arm, written with plain torch ops.

Two constraints shaped this module.

1. Build-from-0 bans torchvision, so ``RandomHorizontalFlip`` and
   ``RandomResizedCrop`` are hand-rolled here. They are not hard -- the point of
   the exercise is that a crop is a slice and a resize is
   ``torch.nn.functional.interpolate``.

2. Per CLAUDE.md the 100k unlabeled images (2.76 GB as uint8) live resident in
   VRAM and the DataLoader is skipped entirely. A torchvision transform pipeline
   is per-image and CPU-bound, which is exactly what that plan is avoiding: with
   the images already on the GPU, eight CPU workers copying tensors back and
   forth would be the bottleneck. Everything below is batched and runs wherever
   its input tensor already is.

INTENDED CALL ORDER
-------------------
    uint8 slice on GPU  ->  data.normalize_batch(...)  ->  augment_batch(...)

i.e. normalize first, augment second. The crop resamples with bilinear
interpolation and therefore needs a float tensor anyway, and normalizing after
the crop would compute statistics over resampled pixels rather than the fixed
constants both arms share.

THE AUGMENTED VIEW IS ALSO THE RECONSTRUCTION TARGET
----------------------------------------------------
The training loop must feed ``augment_batch``'s output to the VAE *and* use that
same tensor as the target in ``vae_loss``. Reconstructing the un-augmented
original from an augmented input would turn the model into a denoising/
de-cropping autoencoder, which is a different objective from the ELBO.

=============================================================================
WHY THIS POLICY IS LIGHT, AND WHY THAT IS NOT A FAIRNESS PROBLEM
=============================================================================
SimCLR *needs* aggressive augmentation: its loss is "pull two views of the same
image together, push other images apart", so the two views have to be hard
enough to make that non-trivial. Strip the colour jitter and grayscale out of
SimCLR and it learns colour histograms.

A VAE has no second view and no such pressure. Its loss compares the
reconstruction to a target, so every augmentation applied changes the target
itself. Aggressive augmentation would spend the decoder's capacity modelling the
augmentation distribution -- a heavily colour-jittered target is a *different
image* to reconstruct, and asking one 512-d latent to cover all of them
increases the posterior variance without adding any semantic structure. Gaussian
blur and grayscale in particular destroy exactly the high-frequency and colour
detail the reconstruction term is scored on, so the model would be penalised for
information it was deliberately denied.

So the policy here is flip + mild crop: enough to stop the decoder from
memorising pixel positions, not enough to change what the target *is*.

CLAUDE.md lists augmentations under "legitimately different, so document rather
than force-match", alongside batch size and optimizer schedule. This block is
that documentation. What must match between the arms -- and does -- is the
trunk, the embedding dimensionality and tap point, the splits, the seed, the
normalization constants and the 100-epoch budget. Copying SimCLR's augmentation
policy into a generative objective would not make the comparison fairer; it
would handicap one arm with a transform that only makes sense for the other.
"""

import math

import torch
import torch.nn.functional as functional


IMAGE_SIZE = 96

# The light policy described above. Kept as a plain dict so a script or notebook
# can copy it, override one key and log the result next to the accuracy numbers.
#
#   horizontal_flip_probability  0.5   left-right flip is label-preserving for
#                                      all ten STL-10 classes (no text, no
#                                      handedness), so it is free variation.
#   crop_scale                 (0.8, 1.0)  keeps at least 80% of the image area.
#                                      SimCLR's default lower bound is 0.08 --
#                                      an order of magnitude more aggressive --
#                                      which routinely crops to a patch of
#                                      background. As a reconstruction target
#                                      that is close to unlearnable.
#   crop_ratio                 (0.9, 1.1)  gentle aspect-ratio wobble; SimCLR
#                                      uses (3/4, 4/3).
#   output_size                96    native STL-10 resolution, unchanged, so the
#                                      encoder always sees the same input shape
#                                      as in Lior's arm.
LIGHT_AUGMENTATION_POLICY = {
    "horizontal_flip_probability": 0.5,
    "crop_scale": (0.8, 1.0),
    "crop_ratio": (0.9, 1.1),
    "output_size": IMAGE_SIZE,
}


def sample_uniform(shape, low, high, reference, generator):
    """Uniform draw in [low, high), returned on ``reference``'s device.

    Exists because ``torch.rand`` refuses a CPU generator with a CUDA ``device``
    argument. The seeding helper in ``seeding.seeded_generator`` hands out CPU
    generators, and the batches live on the GPU, so every draw is made on the
    generator's own device and then moved. The tensors involved are one float per
    image, so the transfer is negligible -- and doing it this way means a run is
    reproducible from a single CPU seed regardless of which device it trains on,
    which is what lets the CPU tests and the 5090 run the same code path.

    ``generator=None`` falls back to the global RNG that ``seeding.reset_seed``
    has already seeded.
    """
    if generator is None:
        values = torch.rand(shape, device=reference.device, dtype=torch.float32)
    else:
        values = torch.rand(
            shape, generator=generator, device=generator.device, dtype=torch.float32
        ).to(reference.device)

    return low + (high - low) * values


def random_horizontal_flip(batch, probability, generator=None):
    """Flip each image left-right independently with the given probability.

    Fully vectorized: the whole batch is flipped once, and ``torch.where`` picks
    per image between the flipped and the original copy. That costs one extra
    buffer the size of the batch but avoids any per-image indexing, which on GPU
    is much cheaper than it sounds compared to a loop of 256 kernel launches.

    Works on any dtype (uint8 included) since a flip only reorders elements.
    """
    if probability <= 0.0:
        return batch

    draw = sample_uniform((batch.shape[0],), 0.0, 1.0, batch, generator)
    should_flip = (draw < probability).view(-1, 1, 1, 1)

    # dim 3 is W in [B, C, H, W]; flipping H would turn images upside down,
    # which is not a transformation STL-10 photographs are invariant to.
    return torch.where(should_flip, torch.flip(batch, dims=(3,)), batch)


def random_resized_crop(batch, output_size, scale, ratio, generator=None):
    """Per-image random crop of a random area/aspect, resized to output_size.

    ``scale`` is a (low, high) fraction of the image *area*; ``ratio`` is a
    (low, high) width/height aspect. Aspect is drawn log-uniformly so that
    squashing and stretching are equally likely -- drawn uniformly, ratios above
    1 would occupy more of the range than their reciprocals.

    Requires a floating-point batch: bilinear interpolation is undefined on
    uint8 and ``interpolate`` would raise a much less helpful error.

    VECTORIZATION, AND WHERE IT STOPS
    ---------------------------------
    All the sampling -- areas, aspects, crop sizes, top-left corners -- is done
    in whole-batch tensor ops, no loop. THE RESIZE ITSELF FALLS BACK TO A PYTHON
    LOOP OVER THE BATCH. That is a real, acknowledged fallback:
    ``interpolate`` applies one output size to one input tensor, and here every
    image has a *different* crop window, so there is no single call that
    expresses the batch. The vectorized alternative is to build a per-image
    affine grid and call ``grid_sample``, which is one kernel for the whole
    batch; it is not used because its sampling grid conventions
    (``align_corners``, half-pixel offsets) differ subtly from ``interpolate``'s
    and would make our crop quietly non-equivalent to the standard
    RandomResizedCrop everyone else's numbers are produced with. With a batch of
    256 the loop is ~256 slice+interpolate pairs per step, which is small next to
    a twelve-block SE-ResNet forward and backward pass.

    The crop boxes are pulled to host memory with a single ``.tolist()`` each
    rather than ``.item()`` inside the loop: on CUDA every ``.item()`` is a
    device synchronisation, so per-image ones would serialise the whole step.
    """
    if not batch.is_floating_point():
        raise ValueError(
            "random_resized_crop needs a float batch (bilinear resampling is "
            "undefined on uint8); normalize with data.normalize_batch first, got %r"
            % (batch.dtype,)
        )

    batch_size, _, height, width = batch.shape
    image_area = float(height * width)

    target_area = sample_uniform((batch_size,), scale[0], scale[1], batch, generator) * image_area
    log_aspect = sample_uniform(
        (batch_size,), math.log(ratio[0]), math.log(ratio[1]), batch, generator
    )
    aspect = torch.exp(log_aspect)

    # area = w * h and aspect = w / h  =>  w = sqrt(area * aspect),
    #                                      h = sqrt(area / aspect).
    # torchvision retries the draw when the box does not fit and only then falls
    # back to a centre crop; we clamp instead. With scale bounded below at 0.8
    # the box essentially always fits, so the two behaviours coincide on this
    # policy, and clamping keeps the function branch-free and deterministic in
    # the number of random draws it consumes -- which matters for reproducibility,
    # since a rejection loop makes the RNG stream depend on the rejected samples.
    crop_width = torch.sqrt(target_area * aspect).round().long().clamp(1, width)
    crop_height = torch.sqrt(target_area / aspect).round().long().clamp(1, height)

    # Uniform top-left corner over the valid range, inclusive of both ends.
    top_fraction = sample_uniform((batch_size,), 0.0, 1.0, batch, generator)
    left_fraction = sample_uniform((batch_size,), 0.0, 1.0, batch, generator)
    top = (top_fraction * (height - crop_height + 1).float()).long().clamp(min=0)
    left = (left_fraction * (width - crop_width + 1).float()).long().clamp(min=0)
    top = torch.minimum(top, height - crop_height)
    left = torch.minimum(left, width - crop_width)

    # One host transfer per box tensor, outside the loop. See docstring.
    top_values = top.tolist()
    left_values = left.tolist()
    crop_height_values = crop_height.tolist()
    crop_width_values = crop_width.tolist()

    resized_crops = []
    for index in range(batch_size):
        row_start = top_values[index]
        column_start = left_values[index]
        window = batch[
            index : index + 1,
            :,
            row_start : row_start + crop_height_values[index],
            column_start : column_start + crop_width_values[index],
        ]
        resized_crops.append(
            functional.interpolate(
                window,
                size=(output_size, output_size),
                mode="bilinear",
                align_corners=False,
            )
        )

    return torch.cat(resized_crops, dim=0)


def augment_batch(batch, config=None, generator=None):
    """Apply the light VAE policy: horizontal flip, then mild random resized crop.

    ``config`` is a dict with the keys of ``LIGHT_AUGMENTATION_POLICY``, which is
    used when it is omitted. ``generator`` is an optional ``torch.Generator``
    (CPU is fine, see ``sample_uniform``); omit it to draw from the global RNG
    that ``seeding.reset_seed`` seeded.

    Flip before crop, not after: both orders give the same distribution of
    outputs, but flipping the full image first is one op on the whole batch,
    whereas flipping after the crop would have to happen inside the per-image
    resize loop.

    The module docstring explains at length why this policy is deliberately
    lighter than Lior's SimCLR policy, and why that is a documented difference
    between the arms rather than a fairness violation. Read it before "fixing"
    this to match his.
    """
    if config is None:
        config = LIGHT_AUGMENTATION_POLICY

    augmented = random_horizontal_flip(
        batch, config["horizontal_flip_probability"], generator
    )
    augmented = random_resized_crop(
        augmented,
        config["output_size"],
        config["crop_scale"],
        config["crop_ratio"],
        generator,
    )

    return augmented
