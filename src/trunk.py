"""SE-ResNet trunk, reimplemented from scratch for Tomer's VAE arm.

Why this file exists at all: the lecturer's "build from 0" rule forbids
``torchvision.models``, so the backbone has to be hand-written. That constraint
turns out to be a gift for the experiment. Both arms of the comparison (Lior's
SimCLR and our VAE) must share the *same* trunk, otherwise the pretraining
objective is no longer the only variable and the whole two-arm comparison
collapses into "which architecture is better".

So this module is a deliberate, line-for-line reimplementation of the
``SEResNetEncoder`` defined in cell 12 of ``simclr_stl10_abilation_f.ipynb``:
same attribute names, same submodule order, same shapes, same lack of biases.
The practical test of "same" is that

    encoder = SEResNetEncoder()
    encoder.load_state_dict(torch.load(lior_checkpoint)["encoder_state"], strict=True)

succeeds. ``strict=True`` compares the full set of parameter/buffer keys and
every tensor shape, so if it loads, the two trunks are the same network. That
is the strongest cheap proof of trunk parity we can produce, and it is why the
attribute names below are copied verbatim rather than "improved".

Only ``torch`` is imported here — no torchvision, no timm, no sklearn.
"""

import torch
import torch.nn as nn


# Architecture constants, mirrored from cell 4 of Lior's notebook. They are
# duplicated here rather than imported so that this module stands alone and so
# that a diff against the notebook is a one-screen visual check.
STEM_WIDTH = 48
STAGE_DEPTHS = 2, 3, 4, 3
STAGE_WIDTHS = 64, 128, 256, 384
ENCODER_DIM = 512
SE_REDUCTION = 16

# Shape of the feature map that reaches ``final_norm``, for a 96x96 input.
# The VAE decoder has to mirror this exactly, so it is exported rather than
# left as a magic number in the decoder module. Both values are specific to
# IMAGE_SIZE = 96; see the spatial trace in SEResNetEncoder's docstring.
FINAL_FEATURE_MAP_CHANNELS = STAGE_WIDTHS[-1]
FINAL_FEATURE_MAP_SIZE = 6


class SqueezeExcitation(nn.Module):
    """Channel-attention gate: pool to one value per channel, learn a rescale.

    The two 1x1 convolutions keep their biases (PyTorch's default), which
    matters for state-dict compatibility -- ``scale.0.bias`` and
    ``scale.2.bias`` are real keys in Lior's checkpoint.
    """

    def __init__(self, channels, reduction):
        super().__init__()

        # The floor of 16 stops the bottleneck from collapsing to 4 channels in
        # the narrow early stages (64 // 16 = 4 would throw away most of the
        # channel statistics).
        hidden_channels = max(channels // reduction, 16)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.scale = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        channel_weights = self.scale(self.pool(x))
        return x * channel_weights


class PreActivationSEBlock(nn.Module):
    """Pre-activation residual block (norm -> act -> conv) with an SE gate.

    Pre-activation means the normalization and ReLU sit *before* each
    convolution, so the residual branch is added to an un-normalized signal and
    gradients reach early layers through a clean identity path.
    """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.activation1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            padding_mode="zeros",
            bias=False,
        )

        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            bias=False,
        )

        self.se = SqueezeExcitation(out_channels, SE_REDUCTION)

        # A projection shortcut is only needed when the block changes the
        # spatial size or the channel count; otherwise the identity is used and
        # there is no ``downsample`` entry in the state dict at all.
        needs_projection = stride != 1 or in_channels != out_channels
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if needs_projection
            else None
        )

    def forward(self, x):
        preactivated = self.activation1(self.norm1(x))

        # DELIBERATE, DO NOT "FIX": the projection shortcut consumes
        # ``preactivated``, not the raw ``x``. Textbook pre-activation ResNets
        # do exactly this (the shortcut conv is treated as part of the block's
        # first pre-activation), but it reads like a bug at a glance and the
        # tempting "correction" to ``self.downsample(x)`` would silently make
        # our trunk a different network from Lior's while still loading his
        # weights -- the worst possible failure mode for a parity argument.
        # The identity branch, in contrast, carries the raw ``x``.
        shortcut = x if self.downsample is None else self.downsample(preactivated)

        residual = self.conv1(preactivated)
        residual = self.norm2(residual)
        residual = self.activation2(residual)
        residual = self.conv2(residual)
        residual = self.se(residual)

        return shortcut + residual


class SEResNetEncoder(nn.Module):
    """The shared trunk. Maps a [B, 3, 96, 96] batch to a [B, 512] embedding.

    Spatial trace for the 96x96 native STL-10 input (asserted here because the
    VAE decoder is built to mirror it):

        input                        96 x 96 x 3
        stem conv1, stride 2   ->    48 x 48 x 48
        stem conv2, stride 1   ->    48 x 48 x 64
        stage 0 (depth 2)      ->    48 x 48 x 64     all strides 1
        stage 1 (depth 3)      ->    24 x 24 x 128    first block stride 2
        stage 2 (depth 4)      ->    12 x 12 x 256    first block stride 2
        stage 3 (depth 3)      ->     6 x  6 x 384    first block stride 2
        output_projection 1x1  ->     6 x  6 x 512
        global average pool    ->     1 x  1 x 512
        flatten(1)             ->    [B, 512]

    So the feature map entering ``final_norm`` is 6 x 6 x 384, i.e.
    ``FINAL_FEATURE_MAP_SIZE`` x ``FINAL_FEATURE_MAP_SIZE`` x
    ``FINAL_FEATURE_MAP_CHANNELS``. The VAE decoder must start from that shape
    and undo exactly these three stride-2 stages plus the stride-2 stem, a
    total downsampling factor of 16.

    Two design choices are worth flagging because they differ from a stock
    ResNet and both are kept for parity:

    * The stem is two 3x3 convolutions with **no max-pool**. STL-10 at 96x96 is
      small enough that the aggressive 7x7-stride-2-plus-pool ImageNet stem
      would throw away too much resolution before any residual block runs.
    * There is **no normalization or activation after the second stem conv**.
      The first ``PreActivationSEBlock`` opens with ``norm1``/``activation1``,
      so adding them in the stem would apply them twice.

    The 512-d output is the embedding tap point for both arms: Lior probes it
    linearly after discarding his projection head, and we feed it to the VAE's
    mu/logvar heads. Same tensor, same dimensionality, which is the fairness
    constraint the whole comparison rests on.
    """

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                STEM_WIDTH,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="zeros",
                bias=False,
            ),
            nn.BatchNorm2d(STEM_WIDTH),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                STEM_WIDTH,
                STAGE_WIDTHS[0],
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
                bias=False,
            ),
        )

        stages = []
        in_channels = STAGE_WIDTHS[0]

        for stage_index, (depth, out_channels) in enumerate(zip(STAGE_DEPTHS, STAGE_WIDTHS)):
            blocks = []

            for block_index in range(depth):
                # Stage 0 keeps the stem's 48x48 resolution; every later stage
                # halves it in its first block only.
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(PreActivationSEBlock(in_channels, out_channels, stride))
                in_channels = out_channels

            stages.append(nn.Sequential(*blocks))

        # Nesting Sequentials (one per stage) rather than flattening all twelve
        # blocks into a single Sequential is what produces the checkpoint keys
        # "stages.<stage_index>.<block_index>.<param>". Flattening would still
        # give an identical network but a different state dict, so it stays.
        self.stages = nn.Sequential(*stages)
        self.final_norm = nn.BatchNorm2d(in_channels)
        self.final_activation = nn.ReLU(inplace=True)
        self.output_projection = nn.Conv2d(in_channels, ENCODER_DIM, kernel_size=1, bias=False)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        # Because the blocks are pre-activation, the last block's output is
        # un-normalized; this trailing norm/ReLU pair is what makes the pooled
        # embedding comparable across images.
        x = self.final_norm(x)
        x = self.final_activation(x)
        x = self.output_projection(x)
        x = self.global_pool(x)
        return x.flatten(1)


def initialize_encoder_weights(encoder):
    """Apply Lior's initialization scheme to a freshly built encoder.

    Ported from ``initialize_model_weights`` in cell 12. His version also
    handles the SimCLR projection head; we have none, so those branches are
    dropped, but every module the two arms share is treated identically.

    The scheme, and why each layer gets what it gets:

    * SE **reduce** conv (``scale[0]``): kaiming normal, ``fan_in``. It is
      followed by a ReLU, and fan_in keeps the *forward* activation variance
      stable, which is what matters for a 1x1 gate computed on pooled values.
    * SE **expand** conv (``scale[2]``) and ``output_projection``: xavier
      uniform. Both feed something that is not a ReLU -- a sigmoid gate and the
      global average pool respectively -- so the kaiming ReLU gain would be
      wrong. Xavier balances forward and backward variance instead.
    * Every other Conv2d: kaiming normal, ``fan_out``, the standard ResNet
      choice; fan_out preserves gradient variance on the backward pass through
      a deep residual stack.
    * BatchNorm: weight 1, bias 0, i.e. start as the identity transform.

    Ids are collected first because ``model.modules()`` yields the SE convs
    interleaved with the ordinary ones and there is no way to tell them apart
    from the module alone -- identity comparison is the cheap, unambiguous way
    to special-case exactly those tensors.

    Note: this reproduces Lior's *rules*, not his random draw. His RNG stream
    also advances through the projection head, so the actual sampled values
    differ. That is fine -- parity of the trunk is established by the state
    dict, and the VAE is trained from its own initialization anyway.
    """

    squeeze_excitation_reduce_ids = set()
    squeeze_excitation_expand_ids = set()

    for module in encoder.modules():
        if isinstance(module, SqueezeExcitation):
            squeeze_excitation_reduce_ids.add(id(module.scale[0]))
            squeeze_excitation_expand_ids.add(id(module.scale[2]))

    output_projection_id = id(encoder.output_projection)

    for module in encoder.modules():
        module_id = id(module)

        if isinstance(module, nn.Conv2d):
            if module_id in squeeze_excitation_reduce_ids:
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            elif module_id in squeeze_excitation_expand_ids or module_id == output_projection_id:
                nn.init.xavier_uniform_(module.weight, gain=1.0)
            else:
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def parameter_count(model):
    """Total number of parameter elements in a model.

    Used to report trunk size in the writeup and, more usefully, as a one-line
    sanity check that our trunk and Lior's have the same capacity. Counts all
    parameters, frozen or not, because the point is architectural size -- when
    the encoder is frozen for embedding extraction, a trainable-only count
    would read zero.
    """

    return sum(parameter.numel() for parameter in model.parameters())
