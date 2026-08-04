"""The generative arm: a beta-VAE whose encoder is Lior's trunk, unchanged.

This module is the whole point of the two-arm comparison. Lior pretrains
``SEResNetEncoder`` with NT-Xent (contrastive); we pretrain the *same class*,
built the *same way*, with the ELBO (generative). Nothing about the backbone,
the embedding dimensionality or the tap point differs -- the encoder is imported
from ``trunk.py`` rather than re-declared here precisely so that no accidental
divergence can creep in. The pretraining objective is therefore the only
variable, which is the claim the entire report rests on.

What this file adds on top of the shared trunk:

    x -> SEResNetEncoder -> 512-d feature -> to_mu     -> mu      (512-d)
                                          -> to_logvar -> logvar  (512-d)
         mu, logvar -> reparameterize -> z -> Decoder -> reconstruction

Only ``torch`` is imported. No torchvision, no sklearn -- build-from-0.
"""

import torch
import torch.nn as nn

try:  # repo root on sys.path (notebooks and scripts import as ``src.vae``)
    from src.trunk import (
        ENCODER_DIM,
        FINAL_FEATURE_MAP_CHANNELS,
        FINAL_FEATURE_MAP_SIZE,
        STAGE_WIDTHS,
        STEM_WIDTH,
        SEResNetEncoder,
        initialize_encoder_weights,
    )
except ImportError:  # src/ itself on sys.path (running a module directly from src/)
    from trunk import (
        ENCODER_DIM,
        FINAL_FEATURE_MAP_CHANNELS,
        FINAL_FEATURE_MAP_SIZE,
        STAGE_WIDTHS,
        STEM_WIDTH,
        SEResNetEncoder,
        initialize_encoder_weights,
    )


# LATENT_DIM must equal ENCODER_DIM, and it is derived rather than typed so the
# two cannot drift. Lior's embedding is the 512-d pooled trunk feature with the
# projection head discarded; our mu -- the thing the KNN graph is built on -- has
# to be 512-d as well. A narrower latent would hand the contrastive arm a
# dimensionality advantage that has nothing to do with generative vs contrastive,
# and that is the single highest-risk fairness item in the project.
LATENT_DIM = ENCODER_DIM

# Decoder channel widths, one per upsampling stage, mirroring the encoder's
# widths in reverse. The encoder walks 96(3) -> 48(48) -> 48(64) -> 24(128) ->
# 12(256) -> 6(384); the decoder walks 6(384) -> 12(256) -> 24(128) -> 48(64) ->
# 96(48) and then projects to 3 channels. The trailing 48 is STEM_WIDTH, so even
# the stem has a mirror image.
DECODER_STAGE_WIDTHS = (STAGE_WIDTHS[2], STAGE_WIDTHS[1], STAGE_WIDTHS[0], STEM_WIDTH)

# exp(logvar) is the variance, and an unconstrained linear head can easily emit
# +40 in the first few hundred steps while the encoder is still unscaled --
# exp(40) is ~2.4e17, which overflows float16 outright and makes the KL term inf
# in float32. Clamping the *logvar* rather than the variance keeps the gradient
# finite instead of NaN. The range is wide enough to be inactive on a healthy
# run: exp(-10) ~ 4.5e-5 and exp(10) ~ 2.2e4 both sit far outside anything a
# converged posterior produces, so the clamp is a guard rail, not a modelling
# choice.
LOGVAR_MINIMUM = -10.0
LOGVAR_MAXIMUM = 10.0


class Decoder(nn.Module):
    """512-d latent -> [B, 3, 96, 96], mirroring the encoder's spatial trace.

    Spatial trace, read against ``SEResNetEncoder``'s docstring backwards:

        z                              [B, 512]
        latent_projection        ->    [B, 384 * 6 * 6]
        reshape                  ->     6 x  6 x 384
        stage 0 (up + conv)      ->    12 x 12 x 256
        stage 1 (up + conv)      ->    24 x 24 x 128
        stage 2 (up + conv)      ->    48 x 48 x  64
        stage 3 (up + conv)      ->    96 x 96 x  48
        output_projection 3x3    ->    96 x 96 x   3

    Four doublings undo the encoder's total downsampling factor of 16 (a
    stride-2 stem plus three stride-2 stages), landing back on the native
    96x96 STL-10 resolution.

    WHY NEAREST-UPSAMPLE + CONV3X3 INSTEAD OF ConvTranspose2d
    ---------------------------------------------------------
    A transposed convolution with kernel 3 (or 4) and stride 2 covers output
    pixels an uneven number of times, and the resulting periodic intensity
    pattern is the familiar checkerboard artifact. For an ordinary image
    generator that is a cosmetic complaint; here it is a methodological one.
    The reconstruction gradient is the *only* signal shaping the latent space,
    so if the decoder can lower its loss by emitting a fixed high-frequency
    grid, the encoder is pushed to encode whatever helps that grid line up --
    high-frequency, texture-like structure rather than object identity. The
    KNN graph is then built on exactly those directions, and label propagation
    ends up diffusing labels along "these two images have similar checkerboard
    phase" edges. Separating the resize (parameter-free nearest upsample) from
    the filtering (a stride-1 conv, which touches every output pixel the same
    number of times) removes the artifact by construction.

    THE FINAL CONV HAS NO ACTIVATION, ON PURPOSE
    --------------------------------------------
    The reconstruction target is a *normalized* image -- ``normalize_batch``
    subtracts the unlabeled-split channel means and divides by the channel
    standard deviations, which puts pixels roughly in [-2.5, +2.5] with both
    signs. A sigmoid or tanh head would clip the target's range and make the
    MSE unable to reach zero, so the output stays unbounded.
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        self.latent_dim = latent_dim
        self.initial_channels = FINAL_FEATURE_MAP_CHANNELS
        self.initial_size = FINAL_FEATURE_MAP_SIZE

        # One Linear back to the 6x6x384 feature map the encoder's global pool
        # collapsed. This is the only place the decoder can undo the pooling:
        # the pooled 512-d vector has no spatial extent at all, so the map has
        # to be re-synthesised rather than un-pooled.
        self.latent_projection = nn.Linear(
            latent_dim, self.initial_channels * self.initial_size * self.initial_size
        )

        stages = []
        in_channels = self.initial_channels

        for out_channels in DECODER_STAGE_WIDTHS:
            stages.append(
                nn.Sequential(
                    # Parameter-free resize; "nearest" rather than "bilinear"
                    # so the upsample contributes no smoothing of its own and
                    # every bit of filtering is learned by the conv below it.
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        padding_mode="zeros",
                        bias=False,
                    ),
                    # bias=False above because the BatchNorm immediately after
                    # has its own shift, which would make the conv bias
                    # redundant and unidentifiable.
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = out_channels

        self.stages = nn.Sequential(*stages)

        # Keeps its bias (PyTorch default): with no BatchNorm after it, this is
        # the only way the decoder can represent a nonzero mean per channel,
        # and a normalized target is not centred at zero per image.
        self.output_projection = nn.Conv2d(
            in_channels, 3, kernel_size=3, stride=1, padding=1, padding_mode="zeros"
        )

    def forward(self, z):
        x = self.latent_projection(z)
        x = x.view(-1, self.initial_channels, self.initial_size, self.initial_size)
        x = self.stages(x)
        return self.output_projection(x)


class VariationalAutoencoder(nn.Module):
    """SEResNetEncoder + Gaussian posterior head + mirrored decoder.

    ============================================================================
    DOWNSTREAM CODE MUST USE ``mu``, NEVER A SAMPLE FROM THE POSTERIOR.
    ============================================================================
    Everything after pretraining -- embedding extraction, L2 normalization, the
    KNN graph, label propagation, the K sweep, the linear probe, the CNN on
    pseudo-labels -- reads ``mu`` and only ``mu``. ``mu`` is a deterministic
    function of the image, so the 105k x 512 embedding matrix, and therefore the
    graph built from it, is byte-identical every time it is recomputed.

    A sample ``z = mu + sigma * epsilon`` is not. Re-extracting embeddings with
    sampling on would perturb every pairwise cosine similarity, which reshuffles
    the top-K neighbours near each tie, which changes the graph's edge set, which
    changes which labels propagate where. Accuracy would then wobble between
    runs for reasons unrelated to the research question, and the beta ablation --
    whose whole job is to attribute changes in LP accuracy to beta -- would be
    measuring noise. ``reparameterize`` below therefore returns ``mu`` unchanged
    whenever the module is in eval mode, so the correct behaviour is the default
    for any code that remembered to call ``.eval()``.

    Sampling is used during *training* only, where it is what makes this a VAE
    rather than a plain autoencoder: the noise is what forces nearby latents to
    decode to similar images, which is exactly the local smoothness label
    propagation assumes.
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        # Imported and instantiated as-is. Do not subclass, wrap or "improve"
        # it: parity with Lior's arm is checked by loading his checkpoint into
        # this exact class with strict=True.
        self.encoder = SEResNetEncoder()

        # Lior's initialization scheme (his cell 12), applied to the trunk only.
        # Without this call the trunk starts from PyTorch's defaults -- kaiming
        # UNIFORM with a = sqrt(5) for every conv -- while his starts from kaiming
        # NORMAL fan_out, with xavier on the SE expand convs and the output
        # projection. Two arms whose shared trunk is drawn from different
        # distributions are not running "the same trunk with a different
        # objective", which is the claim the whole comparison rests on. The
        # decoder and the mu/logvar heads keep PyTorch's defaults: they have no
        # counterpart in his arm, so there is nothing to match them to.
        initialize_encoder_weights(self.encoder)

        self.latent_dim = latent_dim
        self.to_mu = nn.Linear(ENCODER_DIM, latent_dim)
        self.to_logvar = nn.Linear(ENCODER_DIM, latent_dim)

        self.decoder = Decoder(latent_dim)

    def encode(self, x):
        """[B, 3, 96, 96] -> (mu, logvar), both [B, latent_dim]."""
        features = self.encoder(x)
        mu = self.to_mu(features)
        # See LOGVAR_MINIMUM/LOGVAR_MAXIMUM: exp() of an unclamped logvar
        # overflows early in training, before the encoder has settled.
        logvar = self.to_logvar(features).clamp(LOGVAR_MINIMUM, LOGVAR_MAXIMUM)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample z = mu + sigma * epsilon in train mode; return mu in eval mode.

        The reparameterization trick moves the randomness into ``epsilon``, an
        input that carries no parameters, so the gradient of the loss flows
        through ``mu`` and ``logvar`` analytically. Sampling z directly from
        N(mu, sigma^2) would give no gradient path at all.

        ``torch.randn_like`` inherits device and dtype from ``std``, which is how
        this stays CPU/CUDA agnostic without a device argument.
        """
        if not self.training:
            # Determinism guarantee described in the class docstring.
            return mu

        standard_deviation = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(standard_deviation)
        return mu + standard_deviation * epsilon

    def decode(self, z):
        """[B, latent_dim] -> reconstruction [B, 3, 96, 96], unbounded."""
        return self.decoder(z)

    def forward(self, x):
        """-> (reconstruction, mu, logvar).

        All three come back because the loss needs all three and the training
        loop logs the KL term separately from the reconstruction term.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


def vae_loss(reconstruction, target, mu, logvar, beta):
    """Beta-weighted negative ELBO.

    Returns ``(total, reconstruction_term, kl_term)``.

    Reconstruction term: squared error **summed over the 3 * 96 * 96 pixels** and
    then averaged over the batch. Summing over pixels rather than averaging over
    them is what keeps the two terms on a comparable scale -- the KL is a sum
    over 512 latent dimensions, so a per-pixel *mean* reconstruction would be
    ~27648x smaller and even beta = 0.001 would drown it. Averaging over the
    batch (rather than summing) keeps the loss magnitude independent of batch
    size, so the learning rate inherited from Lior's schedule still means the
    same thing.

    KL term: the standard closed form for KL(N(mu, sigma^2) || N(0, I)),

        -0.5 * sum_over_latent_dims(1 + logvar - mu^2 - exp(logvar))

    averaged over the batch. Closed form rather than a Monte-Carlo estimate
    because both distributions are Gaussian, which makes the integral exact and
    the gradient much lower variance.

    ``kl_term`` is returned UNWEIGHTED -- it is the raw KL in nats, not
    ``beta * kl``. That is deliberate: the posterior-collapse signal we have to
    be able to see is the KL itself falling toward 0, meaning every posterior has
    become the prior and ``mu`` carries no information about the image. Logging
    ``beta * kl`` during the beta warmup would confound "the KL is shrinking"
    with "beta is still ramping up", and posterior collapse would hide inside the
    schedule. The training loop logs all three columns for exactly this reason.
    """
    squared_error = (reconstruction - target) ** 2
    reconstruction_term = squared_error.flatten(1).sum(dim=1).mean()

    kl_per_sample = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    kl_term = kl_per_sample.mean()

    total = reconstruction_term + beta * kl_term

    return total, reconstruction_term, kl_term


def beta_at_epoch(epoch, beta_max, warmup_epochs):
    """Linear KL warmup: 0 at epoch 0, ``beta_max`` from ``warmup_epochs`` on.

    WHY THE WARMUP EXISTS
    ---------------------
    At the start of training the decoder is random, so it cannot reconstruct
    anything and the reconstruction term gives the encoder only a weak, noisy
    gradient. The KL term, in contrast, has an immediately available minimum:
    drive every ``mu`` to 0 and every ``logvar`` to 0, i.e. make the posterior
    exactly the prior. If beta is at full strength from step one, that is the
    fastest loss reduction available and the model takes it -- the posterior
    collapses, ``mu`` becomes constant across images, every pairwise cosine
    similarity becomes ~1, and the KNN graph degenerates into noise before any
    useful representation has formed. Ramping beta from 0 gives the decoder time
    to become worth reconstructing through, so that by the time the KL pressure
    arrives there is a real reconstruction gradient to push back with.

    ``epoch`` is 0-indexed, so epoch 0 runs as a plain autoencoder (beta exactly
    0) and epoch ``warmup_epochs`` is the first epoch at full ``beta_max``. Note
    this differs from ``optim.learning_rate_at_step``, which uses
    ``(step + 1) / warmup_steps``: there a zero learning rate would waste a step
    entirely, whereas here a zero beta is a meaningful and desirable first epoch.

    ``warmup_epochs <= 0`` disables the warmup and returns ``beta_max``
    immediately, which is also the sane reading of "no warmup". With
    ``beta_max = 0`` (the plain-autoencoder control arm in BETA_GRID) this
    returns 0 everywhere, as it should.
    """
    if warmup_epochs <= 0:
        return beta_max

    warmup_progress = min(epoch / warmup_epochs, 1.0)
    return beta_max * warmup_progress
