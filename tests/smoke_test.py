"""Executable CPU smoke test for the whole VAE + label-propagation pipeline.

Run it with:

    py -3.12 tests/smoke_test.py

WHY A PLAIN SCRIPT AND NOT PYTEST
---------------------------------
Everything else in this project is written under the "build from 0" rule, and a
test suite that needs `pip install pytest` on the lab server is one more thing to
go wrong on a machine where disk is scarce. `assert` plus a small runner is all a
smoke test needs, and it means the file runs identically on the laptop (CPU) and
on the 5090 box without any harness at all.

WHAT THIS FILE IS FOR
---------------------
It is not a correctness proof of the science. It is the answer to "has any of
this code ever actually executed". It exercises every module in `src/` on tiny
synthetic data, on CPU, in a few seconds, and it checks the handful of things
that are silently catastrophic when wrong:

  * the STL-10 column-major transpose (a wrong orientation trains fine and makes
    every number in the report quietly wrong);
  * the chunked KNN search against a brute-force full-matrix topk, which is the
    correctness proof for the memory optimisation the report claims;
  * the label-propagation re-clamp (without it the labeled rows diffuse away and
    the fixed point stops depending on the labels at all);
  * the -1 unreachable sentinel (without it every unreachable node is silently
    pseudo-labeled class 0);
  * the hand-rolled metrics against values computed by hand on paper.

Every test runs even if an earlier one fails, so one run reports every defect
rather than only the first.
"""

import os
import shutil
import sys
import tempfile
import traceback

import numpy
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import augment
from src import classifier
from src import config
from src import data
from src import knn
from src import labelprop
from src import metrics
from src import optim
from src import probe
from src import seeding
from src import trunk
from src import unionfind
from src import vae


DEVICE = torch.device("cpu")

REGISTERED_TESTS = []


def smoke_test(function):
    """Collect a test function. Plain decorator so the file stays dependency-free."""
    REGISTERED_TESTS.append(function)
    return function


# ---------------------------------------------------------------------------
# Trunk
# ---------------------------------------------------------------------------


@smoke_test
def test_encoder_output_shape_and_spatial_trace():
    """[2, 3, 96, 96] -> [2, 512], with a 6x6 map entering the global pool.

    The 6x6 is load-bearing: `vae.Decoder` starts from
    FINAL_FEATURE_MAP_SIZE and undoes exactly four factor-2 upsamples. If the
    encoder's downsampling factor ever drifted, the decoder would still build and
    would still run -- it would just reconstruct the wrong resolution.
    """
    torch.manual_seed(0)
    encoder = trunk.SEResNetEncoder().eval()

    captured = {}
    encoder.final_norm.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__(
            "into_final_norm", tuple(inputs[0].shape)))
    encoder.global_pool.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__(
            "into_global_pool", tuple(inputs[0].shape)))

    with torch.no_grad():
        embeddings = encoder(torch.randn(2, 3, 96, 96))

    assert tuple(embeddings.shape) == (2, 512), tuple(embeddings.shape)
    assert embeddings.dtype == torch.float32
    assert torch.isfinite(embeddings).all()

    assert captured["into_final_norm"] == (2, 384, 6, 6), captured["into_final_norm"]
    assert captured["into_global_pool"] == (2, 512, 6, 6), captured["into_global_pool"]

    assert trunk.FINAL_FEATURE_MAP_SIZE == 6
    assert trunk.FINAL_FEATURE_MAP_CHANNELS == 384
    assert trunk.ENCODER_DIM == config.ENCODER_DIM == 512


@smoke_test
def test_encoder_matches_config_constants():
    """The trunk module and the config module must not drift apart."""
    assert tuple(trunk.STAGE_DEPTHS) == tuple(config.STAGE_DEPTHS)
    assert tuple(trunk.STAGE_WIDTHS) == tuple(config.STAGE_WIDTHS)
    assert trunk.STEM_WIDTH == config.STEM_WIDTH
    assert trunk.SE_REDUCTION == config.SE_REDUCTION

    encoder = trunk.SEResNetEncoder()
    assert len(encoder.stages) == len(config.STAGE_DEPTHS)
    for stage_index, depth in enumerate(config.STAGE_DEPTHS):
        assert len(encoder.stages[stage_index]) == depth

    assert trunk.parameter_count(encoder) > 1000000


@smoke_test
def test_encoder_initialization_runs():
    """`initialize_encoder_weights` must touch every module without raising."""
    encoder = trunk.SEResNetEncoder()
    trunk.initialize_encoder_weights(encoder)

    for module in encoder.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert torch.equal(module.bias, torch.zeros_like(module.bias))
        if isinstance(module, torch.nn.Conv2d):
            assert torch.isfinite(module.weight).all()


@smoke_test
def test_encoder_state_dict_has_expected_keys():
    """A cheap guard on the checkpoint key names the parity argument relies on."""
    keys = set(trunk.SEResNetEncoder().state_dict().keys())
    for expected in ("stem.0.weight", "stages.0.0.conv1.weight",
                     "stages.0.0.se.scale.0.weight", "final_norm.weight",
                     "output_projection.weight"):
        assert expected in keys, expected
    # Stage 0 keeps 64 channels at stride 1, so its first block needs no
    # projection shortcut and must NOT have a downsample entry.
    assert "stages.0.0.downsample.weight" not in keys
    # Stage 1 changes both stride and width, so it must.
    assert "stages.1.0.downsample.weight" in keys


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------


@smoke_test
def test_vae_forward_shapes():
    torch.manual_seed(0)
    model = vae.VariationalAutoencoder().eval()

    with torch.no_grad():
        reconstruction, mu, logvar = model(torch.randn(2, 3, 96, 96))

    assert tuple(reconstruction.shape) == (2, 3, 96, 96), tuple(reconstruction.shape)
    assert tuple(mu.shape) == (2, 512), tuple(mu.shape)
    assert tuple(logvar.shape) == (2, 512), tuple(logvar.shape)
    assert torch.isfinite(reconstruction).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(logvar).all()
    assert vae.LATENT_DIM == config.LATENT_DIM == 512


@smoke_test
def test_vae_eval_mode_returns_mu_exactly():
    """Determinism guarantee: no sampling outside training mode.

    Everything downstream (the graph, the sweep, the ablation) reads mu, and it
    has to be byte-identical between re-extractions or the graph changes for
    reasons unrelated to the research question.
    """
    torch.manual_seed(0)
    model = vae.VariationalAutoencoder()

    mu = torch.randn(4, 512)
    logvar = torch.zeros(4, 512)

    model.eval()
    assert torch.equal(model.reparameterize(mu, logvar), mu)

    model.train()
    sampled = model.reparameterize(mu, logvar)
    assert not torch.equal(sampled, mu), "training mode must actually sample"


@smoke_test
def test_vae_logvar_is_clamped():
    """An unclamped exp(logvar) overflows in the first few hundred steps."""
    model = vae.VariationalAutoencoder().eval()
    with torch.no_grad():
        model.to_logvar.weight.zero_()
        model.to_logvar.bias.fill_(1000.0)
        _, logvar = model.encode(torch.randn(2, 3, 96, 96))
    assert float(logvar.max().item()) == vae.LOGVAR_MAXIMUM


@smoke_test
def test_vae_loss_three_finite_scalars():
    torch.manual_seed(0)
    reconstruction = torch.randn(4, 3, 96, 96)
    target = torch.randn(4, 3, 96, 96)
    mu = torch.randn(4, 512)
    logvar = torch.randn(4, 512).clamp(-2.0, 2.0)

    total, reconstruction_term, kl_term = vae.vae_loss(
        reconstruction, target, mu, logvar, beta=0.01)

    for name, value in (("total", total), ("reconstruction", reconstruction_term),
                        ("kl", kl_term)):
        assert isinstance(value, torch.Tensor), name
        assert value.dim() == 0, (name, value.shape)
        assert torch.isfinite(value), name

    # The reported KL must be the RAW KL, not beta * KL -- otherwise posterior
    # collapse hides inside the warmup schedule.
    # RELATIVE tolerance, not absolute: the reconstruction term is a sum over
    # 3 * 96 * 96 pixels, so it lands around 5.5e4 and one float32 ulp there is
    # already ~4e-3. An absolute 1e-3 check would fail on correct arithmetic.
    recomposed = float(reconstruction_term.item()) + 0.01 * float(kl_term.item())
    assert abs(float(total.item()) - recomposed) <= 1e-6 * abs(float(total.item()))

    # Reconstruction is summed over pixels, averaged over the batch.
    expected_reconstruction = ((reconstruction - target) ** 2).sum() / 4.0
    assert torch.allclose(reconstruction_term, expected_reconstruction, rtol=1e-5)

    # KL of N(0, I) against N(0, I) is exactly 0.
    zero_kl = vae.vae_loss(reconstruction, target,
                           torch.zeros(4, 512), torch.zeros(4, 512), beta=1.0)[2]
    assert abs(float(zero_kl.item())) < 1e-6, float(zero_kl.item())


@smoke_test
def test_beta_warmup_ramp():
    assert vae.beta_at_epoch(0, 0.01, 10) == 0.0
    assert abs(vae.beta_at_epoch(5, 0.01, 10) - 0.005) < 1e-12
    assert abs(vae.beta_at_epoch(10, 0.01, 10) - 0.01) < 1e-12
    assert abs(vae.beta_at_epoch(99, 0.01, 10) - 0.01) < 1e-12
    # No warmup means full beta immediately.
    assert vae.beta_at_epoch(0, 0.01, 0) == 0.01
    # The plain-autoencoder control arm stays at 0 everywhere.
    assert vae.beta_at_epoch(0, 0.0, 10) == 0.0
    assert vae.beta_at_epoch(50, 0.0, 10) == 0.0
    # Monotone non-decreasing over the warmup.
    ramp = [vae.beta_at_epoch(epoch, 0.1, 10) for epoch in range(15)]
    assert all(later >= earlier for earlier, later in zip(ramp, ramp[1:]))
    assert max(ramp) <= 0.1 + 1e-12


# ---------------------------------------------------------------------------
# STL-10 reader
# ---------------------------------------------------------------------------


def _asymmetric_reference_images(number_of_images):
    """Images whose value pattern differs under an H/W transpose.

    `3 * row + column` is not equal to `3 * column + row`, so if the
    column-major fix in `data.load_images` were dropped, the arrays below would
    not merely look different -- they would compare unequal, which is exactly
    the check we want.
    """
    rows = numpy.arange(data.IMAGE_SIZE).reshape(data.IMAGE_SIZE, 1)
    columns = numpy.arange(data.IMAGE_SIZE).reshape(1, data.IMAGE_SIZE)
    base = (3 * rows + columns) % 256

    images = numpy.zeros(
        (number_of_images, data.NUMBER_OF_CHANNELS, data.IMAGE_SIZE, data.IMAGE_SIZE),
        dtype=numpy.uint8)
    for image_index in range(number_of_images):
        for channel in range(data.NUMBER_OF_CHANNELS):
            images[image_index, channel] = (base + 7 * image_index + 29 * channel) % 256
    return images


def _write_fake_stl10(directory, number_of_images):
    """Write a tiny stl10_binary/ in the REAL on-disk layout.

    STL-10 stores each image column-major, so the byte stream reshaped to
    [N, 3, 96, 96] is [N, C, W, H]. To fabricate that from a reference array in
    [N, C, H, W] we transpose the last two axes on the way OUT -- the mirror of
    what the loader does on the way in.
    """
    binary_directory = os.path.join(directory, "stl10_binary")
    os.makedirs(binary_directory, exist_ok=True)

    images = _asymmetric_reference_images(number_of_images)
    on_disk = numpy.ascontiguousarray(numpy.transpose(images, (0, 1, 3, 2)))
    on_disk.tofile(os.path.join(binary_directory, "train_X.bin"))

    # STL-10 labels are 1-indexed uint8.
    labels = (numpy.arange(number_of_images) % 10).astype(numpy.uint8) + 1
    labels.tofile(os.path.join(binary_directory, "train_y.bin"))

    generator = numpy.random.default_rng(0)
    with open(os.path.join(binary_directory, "fold_indices.txt"), "w",
              encoding="utf-8") as handle:
        for _ in range(10):
            fold = generator.permutation(5000)[:1000]
            handle.write(" ".join(str(int(value)) for value in fold) + "\n")

    with open(os.path.join(binary_directory, "class_names.txt"), "w",
              encoding="utf-8") as handle:
        handle.write("\n".join("class_%d" % index for index in range(10)) + "\n")

    return binary_directory, images, labels.astype(numpy.int64) - 1


@smoke_test
def test_stl10_reader_orientation():
    """THE transpose test. A wrong orientation is invisible until the report is wrong."""
    directory = tempfile.mkdtemp(prefix="stl10_smoke_")
    try:
        # 12 images so all ten 1-indexed label values appear in the fake file and
        # the 1..10 -> 0..9 conversion is genuinely exercised at both ends.
        binary_directory, expected_images, expected_labels = _write_fake_stl10(directory, 12)

        loaded_images = data.load_images(os.path.join(binary_directory, "train_X.bin"))
        assert loaded_images.dtype == numpy.uint8
        assert loaded_images.shape == expected_images.shape, loaded_images.shape
        assert numpy.array_equal(loaded_images, expected_images), (
            "orientation is wrong -- the column-major transpose in data.load_images "
            "is missing or applied on the wrong axes")

        # The transposed version must NOT match, which is what proves the test
        # would actually catch a dropped transpose.
        assert not numpy.array_equal(
            loaded_images, numpy.transpose(expected_images, (0, 1, 3, 2)))

        loaded_labels = data.load_labels(os.path.join(binary_directory, "train_y.bin"))
        assert loaded_labels.dtype == numpy.int64
        assert numpy.array_equal(loaded_labels, expected_labels)
        assert int(loaded_labels.min()) == 0 and int(loaded_labels.max()) == 9

        memory_mapped = data.load_images_memory_mapped(
            os.path.join(binary_directory, "train_X.bin"))
        assert numpy.array_equal(numpy.asarray(memory_mapped), expected_images)
        assert not memory_mapped.flags["C_CONTIGUOUS"], (
            "the memmap view should be a stride swap, not a copy")

        # resolve_binary_directory must accept both the parent and the archive.
        assert data.resolve_binary_directory(directory) == binary_directory
        assert data.resolve_binary_directory(binary_directory) == binary_directory

        assert data.load_class_names(directory) == ["class_%d" % i for i in range(10)]

        folds = data.load_official_folds(directory)
        assert folds.shape == (10, 1000), folds.shape
        assert folds.dtype == numpy.int64
        assert int(folds.min()) >= 0 and int(folds.max()) < 5000

        # A truncated download must be caught, not silently trained on.
        try:
            data.load_split(directory, "train")
        except ValueError:
            pass
        else:
            raise AssertionError("load_split accepted a 12-image 'train' split")
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@smoke_test
def test_channel_statistics_match_direct_computation():
    generator = numpy.random.default_rng(7)
    images = generator.integers(0, 256, size=(11, 3, 96, 96), dtype=numpy.uint8)

    mean, standard_deviation = data.compute_channel_statistics(images, chunk_size=4)

    reference = images.astype(numpy.float64) / 255.0
    expected_mean = reference.mean(axis=(0, 2, 3))
    expected_standard_deviation = reference.std(axis=(0, 2, 3))

    assert len(mean) == 3 and len(standard_deviation) == 3
    assert numpy.allclose(mean, expected_mean, atol=1e-12), (mean, expected_mean)
    assert numpy.allclose(standard_deviation, expected_standard_deviation, atol=1e-9), (
        standard_deviation, expected_standard_deviation)

    # Chunking must not change the answer.
    mean_other_chunk, std_other_chunk = data.compute_channel_statistics(images, chunk_size=11)
    assert numpy.allclose(mean, mean_other_chunk, atol=1e-12)
    assert numpy.allclose(standard_deviation, std_other_chunk, atol=1e-12)

    # A constant image has zero variance and must not come back as NaN.
    constant = numpy.full((3, 3, 96, 96), 200, dtype=numpy.uint8)
    _, constant_std = data.compute_channel_statistics(constant, chunk_size=2)
    assert all(value >= 0.0 for value in constant_std)
    assert not any(numpy.isnan(constant_std))


@smoke_test
def test_normalize_batch():
    images = numpy.full((2, 3, 4, 4), 128, dtype=numpy.uint8)
    mean = [0.5, 0.25, 0.0]
    standard_deviation = [0.5, 1.0, 2.0]

    normalized = data.normalize_batch(images, mean, standard_deviation, DEVICE)

    assert normalized.dtype == torch.float32
    assert tuple(normalized.shape) == (2, 3, 4, 4)
    scaled = 128.0 / 255.0
    for channel in range(3):
        expected = (scaled - mean[channel]) / standard_deviation[channel]
        assert abs(float(normalized[0, channel].mean().item()) - expected) < 1e-5

    # A uint8 TENSOR input must not be mutated. This is the server's hot path:
    # the 2.76 GB unlabeled split lives resident on the device and every batch is
    # a slice of it, so an in-place normalization that wrote back into the slice
    # would corrupt the dataset progressively over an epoch.
    resident = torch.full((2, 3, 4, 4), 128, dtype=torch.uint8)
    before = resident.clone()
    data.normalize_batch(resident, mean, standard_deviation, DEVICE)
    assert torch.equal(resident, before), "normalize_batch mutated its uint8 input"

    # Same guarantee for a float tensor. `.to(float32)` is a no-op on float32
    # input, so without an explicit copy the in-place sub_/div_ would alias and
    # rewrite the caller's buffer.
    float_resident = torch.full((2, 3, 4, 4), 128.0)
    float_before = float_resident.clone()
    normalized_float = data.normalize_batch(
        float_resident, mean, standard_deviation, DEVICE)
    assert torch.equal(float_resident, float_before), \
        "normalize_batch mutated its float input in place"
    assert normalized_float.data_ptr() != float_resident.data_ptr(), \
        "normalize_batch returned an alias of its input"

    # A non-contiguous memmap-style view (what the column-major fix produces)
    # must be accepted rather than rejected by torch.from_numpy.
    stride_swapped = numpy.transpose(
        numpy.full((2, 3, 4, 5), 200, dtype=numpy.uint8), (0, 1, 3, 2))
    assert not stride_swapped.flags["C_CONTIGUOUS"]
    assert tuple(data.normalize_batch(
        stride_swapped, mean, standard_deviation, DEVICE).shape) == (2, 3, 5, 4)


@smoke_test
def test_stratified_split_preserves_proportions_and_is_deterministic():
    labels = numpy.concatenate([
        numpy.full(60, 0), numpy.full(30, 1), numpy.full(10, 2)]).astype(numpy.int64)

    train_positions, validation_positions = data.stratified_split(labels, 0.2, seed=42)

    assert len(train_positions) + len(validation_positions) == len(labels)
    assert len(numpy.intersect1d(train_positions, validation_positions)) == 0
    assert numpy.array_equal(
        numpy.sort(numpy.concatenate([train_positions, validation_positions])),
        numpy.arange(len(labels)))
    assert numpy.array_equal(train_positions, numpy.sort(train_positions))
    assert numpy.array_equal(validation_positions, numpy.sort(validation_positions))

    for class_id, expected_validation in ((0, 12), (1, 6), (2, 2)):
        assert int((labels[validation_positions] == class_id).sum()) == expected_validation
        assert int((labels[train_positions] == class_id).sum()) == \
            int((labels == class_id).sum()) - expected_validation

    # Deterministic across calls...
    repeat_train, repeat_validation = data.stratified_split(labels, 0.2, seed=42)
    assert numpy.array_equal(train_positions, repeat_train)
    assert numpy.array_equal(validation_positions, repeat_validation)

    # ...and genuinely seed-dependent.
    other_train, other_validation = data.stratified_split(labels, 0.2, seed=43)
    assert not numpy.array_equal(validation_positions, other_validation)
    assert len(other_train) + len(other_validation) == len(labels)

    # No class may vanish from either half, even at an extreme fraction.
    tiny_train, tiny_validation = data.stratified_split(labels, 0.001, seed=1)
    for class_id in (0, 1, 2):
        assert int((labels[tiny_validation] == class_id).sum()) >= 1
        assert int((labels[tiny_train] == class_id).sum()) >= 1


@smoke_test
def test_split_file_round_trip():
    """build_and_save_splits -> load_splits must survive the .npz round trip.

    `load_split` is monkeypatched because the real one insists on 5000 images and
    a genuine train_X.bin is 138 MB -- not something a smoke test should write.
    The npz round trip is what is under test here, not the reader.
    """
    directory = tempfile.mkdtemp(prefix="splits_smoke_")
    original_load_split = data.load_split
    try:
        binary_directory, _, _ = _write_fake_stl10(directory, 4)
        fake_labels = (numpy.arange(5000) % 10).astype(numpy.int64)

        def fake_load_split(data_root, split, memory_map_unlabeled=True):
            assert split == "train"
            return None, fake_labels

        data.load_split = fake_load_split

        output_path = os.path.join(directory, "splits")
        built = data.build_and_save_splits(directory, output_path, seed=42,
                                           validation_fraction=0.20)
        loaded = data.load_splits(output_path + ".npz")

        assert loaded["seed"] == 42
        assert abs(loaded["validation_fraction"] - 0.20) < 1e-12
        assert numpy.array_equal(loaded["official_folds"], built["official_folds"])
        assert numpy.array_equal(loaded["train_labels"], fake_labels)
        assert len(loaded["development_splits"]) == 10

        for fold_index in range(10):
            train_positions, validation_positions = loaded["development_splits"][fold_index]
            assert len(train_positions) + len(validation_positions) == 1000
            assert len(numpy.intersect1d(train_positions, validation_positions)) == 0
            assert int(validation_positions.max()) < 1000
            assert numpy.array_equal(
                train_positions, built["development_splits"][fold_index][0])
    finally:
        data.load_split = original_load_split
        shutil.rmtree(directory, ignore_errors=True)


# ---------------------------------------------------------------------------
# KNN graph
# ---------------------------------------------------------------------------


def _brute_force_topk(embeddings, k):
    """Reference implementation: build the WHOLE similarity matrix, then topk.

    This is the thing `build_knn_graph` claims to be equivalent to. It is only
    possible at N = 50; at N = 105000 it would be 44 GB, which is the entire
    reason the chunked version exists.
    """
    normalized = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-12)
    similarity = normalized @ normalized.t()
    similarity.fill_diagonal_(float("-inf"))
    return torch.topk(similarity, k, dim=1, largest=True, sorted=True)


@smoke_test
def test_l2_normalize():
    generator = torch.Generator().manual_seed(0)
    embeddings = torch.randn(20, 8, generator=generator) * 5.0

    normalized = knn.l2_normalize(embeddings)
    assert torch.allclose(normalized.norm(dim=1), torch.ones(20), atol=1e-6)

    # Idempotent, so build_knn_graph can call it unconditionally.
    assert torch.allclose(knn.l2_normalize(normalized), normalized, atol=1e-7)

    # A zero row must stay zero rather than becoming NaN and poisoning topk.
    with_zero = embeddings.clone()
    with_zero[3] = 0.0
    normalized_with_zero = knn.l2_normalize(with_zero)
    assert torch.isfinite(normalized_with_zero).all()
    assert float(normalized_with_zero[3].abs().max().item()) == 0.0

    # numpy input is accepted (cached embeddings are fp16 numpy on disk).
    from_numpy = knn.l2_normalize(embeddings.numpy().astype(numpy.float16))
    assert from_numpy.dtype == torch.float32
    assert from_numpy.device.type == "cpu"


@smoke_test
def test_chunked_knn_matches_brute_force_exactly():
    """THE correctness proof for the chunking, at N=50, k=5, chunk_rows=7.

    chunk_rows=7 does not divide 50, so the last chunk is ragged -- the case an
    off-by-one in the diagonal mask would break.
    """
    generator = torch.Generator().manual_seed(1234)
    embeddings = torch.randn(50, 16, generator=generator)

    reference_similarities, reference_indices = _brute_force_topk(embeddings, 5)

    indices, similarities = knn.build_knn_graph(embeddings, k=5, chunk_rows=7)

    assert tuple(indices.shape) == (50, 5), tuple(indices.shape)
    assert indices.dtype == torch.int64
    assert similarities.dtype == torch.float32
    assert torch.equal(indices, reference_indices), "chunked KNN indices differ from brute force"
    assert torch.allclose(similarities, reference_similarities, atol=1e-6)

    # No node may be its own neighbour.
    self_index = torch.arange(50).reshape(-1, 1)
    assert not bool((indices == self_index).any()), "the diagonal mask leaked a self-loop"

    # Sorted by DECREASING similarity -- heat_kernel_weights relies on it.
    assert bool((similarities[:, :-1] >= similarities[:, 1:]).all())

    # Every chunk size, including 1, N and > N, must give the same indices.
    for chunk_rows in (1, 2, 3, 7, 13, 49, 50, 64, 1024):
        chunk_indices, chunk_similarities = knn.build_knn_graph(
            embeddings, k=5, chunk_rows=chunk_rows)
        assert torch.equal(chunk_indices, reference_indices), chunk_rows
        assert torch.allclose(chunk_similarities, reference_similarities, atol=1e-6), chunk_rows

    # And numpy fp16 input follows the same path.
    numpy_indices, _ = knn.build_knn_graph(
        embeddings.numpy().astype(numpy.float32), k=5, chunk_rows=7)
    assert torch.equal(numpy_indices, reference_indices)

    # Guard rails.
    for bad_k in (0, 50):
        try:
            knn.build_knn_graph(embeddings, k=bad_k, chunk_rows=7)
        except ValueError:
            pass
        else:
            raise AssertionError("build_knn_graph accepted k=%d at N=50" % bad_k)


@smoke_test
def test_heat_kernel_weights():
    similarities = torch.tensor([[1.0, 0.5, 0.0], [0.8, 0.6, 0.4]])

    weights, bandwidth = knn.heat_kernel_weights(similarities, sigma=0.5)
    assert abs(bandwidth - 0.5) < 1e-12
    assert torch.allclose(weights, torch.exp((similarities - 1.0) / 0.5), atol=1e-7)
    # Identical vectors get exactly weight 1, and weights never exceed 1.
    assert abs(float(weights[0, 0].item()) - 1.0) < 1e-7
    assert float(weights.max().item()) <= 1.0 + 1e-7

    # Self-tuning bandwidth = mean cosine distance to the FURTHEST kept neighbour.
    _, auto_bandwidth = knn.heat_kernel_weights(similarities, sigma=None)
    expected = float(((1.0 - 0.0) + (1.0 - 0.4)) / 2.0)
    assert abs(auto_bandwidth - expected) < 1e-6, (auto_bandwidth, expected)

    # A fully collapsed latent must not divide by zero.
    collapsed = torch.ones(4, 3)
    collapsed_weights, collapsed_bandwidth = knn.heat_kernel_weights(collapsed, sigma=None)
    assert collapsed_bandwidth > 0.0
    assert torch.isfinite(collapsed_weights).all()


@smoke_test
def test_symmetrize_union_and_row_normalize():
    # A deliberately one-directional graph: node 0 lists 1, node 1 lists 2,
    # node 2 lists 1. Union must recover every reverse edge.
    neighbor_indices = torch.tensor([[1], [2], [1]])
    weights = torch.tensor([[0.25], [0.75], [0.5]])

    symmetric = knn.symmetrize_union(neighbor_indices, weights, num_nodes=3)
    dense = symmetric.to_dense()

    assert tuple(dense.shape) == (3, 3)
    assert torch.allclose(dense, dense.t(), atol=0.0), "W is not symmetric"
    assert float(dense[0, 1].item()) == 0.25 and float(dense[1, 0].item()) == 0.25
    # Both directions exist between 1 and 2 with weights 0.75 and 0.5 -- the
    # union rule keeps the MAX, not the sum and not the mean.
    assert abs(float(dense[1, 2].item()) - 0.75) < 1e-7, float(dense[1, 2].item())
    assert abs(float(dense[2, 1].item()) - 0.75) < 1e-7
    assert float(dense[0, 0].item()) == 0.0

    propagation = knn.row_normalize(symmetric)
    propagation_dense = propagation.to_dense()
    row_sums = propagation_dense.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(3), atol=1e-6), row_sums

    # symmetrize_union requires one row per node (neighbor_indices is always
    # [N, K]), so it rejects a mismatch rather than silently padding.
    try:
        knn.symmetrize_union(neighbor_indices, weights, num_nodes=6)
    except ValueError:
        pass
    else:
        raise AssertionError("symmetrize_union accepted 3 rows for 6 nodes")


@smoke_test
def test_symmetrize_union_matches_a_dense_max_reference():
    """The sort-based max-reduce must equal a dense `max(W, W.T)`.

    `coalesce()` reduces duplicate (row, col) entries by SUM, so the max has to
    be done by hand; this is the check that the hand-rolled version is right. A
    dense reference is only affordable because N is 40 here -- at N = 105000 it
    would be the 44 GB matrix the whole module exists to avoid.
    """
    generator = torch.Generator().manual_seed(99)
    number_of_nodes, neighbors_per_node = 40, 4
    embeddings = torch.randn(number_of_nodes, 12, generator=generator)

    indices, similarities = knn.build_knn_graph(
        embeddings, k=neighbors_per_node, chunk_rows=9)
    weights, _ = knn.heat_kernel_weights(similarities, sigma=None)

    directed = torch.zeros(number_of_nodes, number_of_nodes)
    for row in range(number_of_nodes):
        for column_position in range(neighbors_per_node):
            directed[row, int(indices[row, column_position].item())] = \
                float(weights[row, column_position].item())
    reference = torch.maximum(directed, directed.t())

    produced = knn.symmetrize_union(indices, weights, number_of_nodes).to_dense()

    assert torch.allclose(produced, reference, atol=1e-6), \
        float((produced - reference).abs().max().item())
    assert torch.allclose(produced, produced.t(), atol=0.0)
    assert float(torch.diagonal(produced).abs().max().item()) == 0.0

    propagation = knn.row_normalize(
        knn.symmetrize_union(indices, weights, number_of_nodes)).to_dense()
    assert torch.isfinite(propagation).all()
    assert torch.allclose(propagation.sum(dim=1), torch.ones(number_of_nodes), atol=1e-6)

    reference_propagation = reference / reference.sum(dim=1, keepdim=True)
    assert torch.allclose(propagation, reference_propagation, atol=1e-6)


@smoke_test
def test_row_normalize_handles_empty_and_zero_weight_rows():
    """Isolated and fully-underflowed rows must give all-zero rows, never NaN.

    A single NaN row poisons every node in its component within two iterations
    of `F <- P F`, so this branch is worth pinning even though the KNN graph
    itself always gives every node K out-edges.
    """
    number_of_nodes = 5
    # (row, column) pairs, one per value. Nodes 0, 1 and 2 carry real weights;
    # node 3's only edge underflowed to exactly 0.0; node 4 has no entries at all.
    # Note the pairs are all distinct -- coalesce() reduces duplicates by SUM, so
    # repeating a pair here would silently add the weights instead of listing them.
    indices = torch.tensor([[0, 0, 1, 2, 3],
                            [1, 2, 0, 0, 0]])
    values = torch.tensor([1.0, 3.0, 1.0, 3.0, 0.0])
    sparse_weights = torch.sparse_coo_tensor(
        indices, values, (number_of_nodes, number_of_nodes)).coalesce()

    propagation = knn.row_normalize(sparse_weights).to_dense()

    assert torch.isfinite(propagation).all(), "row_normalize produced NaN/inf"
    row_sums = propagation.sum(dim=1)
    assert abs(float(row_sums[0].item()) - 1.0) < 1e-6
    assert abs(float(row_sums[1].item()) - 1.0) < 1e-6
    assert abs(float(row_sums[2].item()) - 1.0) < 1e-6
    assert float(row_sums[3].item()) == 0.0, "an all-zero-weight row must stay zero"
    assert float(row_sums[4].item()) == 0.0, "an edgeless row must stay zero"

    # Every row of P sums to exactly 1 or exactly 0 -- nothing in between.
    for value in row_sums.tolist():
        assert abs(value - 1.0) < 1e-6 or value == 0.0, value

    # Proportions are preserved: node 0's weights were 1 (to node 1) and 3 (to node 2).
    assert abs(float(propagation[0, 1].item()) - 0.25) < 1e-6, float(propagation[0, 1].item())
    assert abs(float(propagation[0, 2].item()) - 0.75) < 1e-6, float(propagation[0, 2].item())

    # An entirely empty graph must round-trip rather than raise.
    empty = torch.sparse_coo_tensor(
        torch.empty((2, 0), dtype=torch.int64), torch.empty(0), (3, 3)).coalesce()
    assert float(knn.row_normalize(empty).to_dense().abs().sum().item()) == 0.0


# ---------------------------------------------------------------------------
# Label propagation
# ---------------------------------------------------------------------------


@smoke_test
def test_one_hot_and_initial_label_matrix():
    one_hot = labelprop.one_hot_matrix([0, 2, 1], num_classes=3, device=DEVICE)
    assert torch.equal(one_hot, torch.tensor([[1.0, 0.0, 0.0],
                                              [0.0, 0.0, 1.0],
                                              [0.0, 1.0, 0.0]]))

    initial_F = labelprop.initial_label_matrix(
        5, 3, labeled_indices=[1, 4], labeled_labels=[2, 0], device=DEVICE)
    assert tuple(initial_F.shape) == (5, 3)
    # Unlabeled rows are EXACTLY zero, not uniform -- that is what makes an
    # unreachable node detectable at the end.
    assert float(initial_F[0].abs().sum().item()) == 0.0
    assert float(initial_F[1, 2].item()) == 1.0
    assert float(initial_F[4, 0].item()) == 1.0

    # The -1 sentinel must never be seeded into the label matrix.
    try:
        labelprop.one_hot_matrix([-1, 0], num_classes=3, device=DEVICE)
    except ValueError:
        pass
    else:
        raise AssertionError("one_hot_matrix accepted the -1 sentinel")


@smoke_test
def test_label_propagation_on_two_clusters():
    """Two cliques plus an isolated node, one seed per clique.

    Checks the three properties that make or break the method: the clusters are
    recovered, the labeled rows are STILL exactly one-hot at the end (which is
    only true if the re-clamp runs every step), and the node in the unseeded
    component comes back as -1 rather than silently as class 0.
    """
    adjacency = torch.zeros(6, 6)
    for i in (0, 1, 2):
        for j in (0, 1, 2):
            if i != j:
                adjacency[i, j] = 1.0
    for i in (3, 4):
        for j in (3, 4):
            if i != j:
                adjacency[i, j] = 1.0
    # Node 5 is isolated: no edges at all.

    row_sums = adjacency.sum(dim=1, keepdim=True)
    propagation = torch.where(row_sums > 0, adjacency / row_sums.clamp(min=1e-12),
                              torch.zeros_like(adjacency))

    labeled_indices = [0, 3]
    labeled_labels = [0, 1]
    initial_F = labelprop.initial_label_matrix(
        6, 2, labeled_indices, labeled_labels, DEVICE)
    clamp_target = initial_F[torch.tensor(labeled_indices)]

    final_F, iterations, delta = labelprop.propagate(
        propagation, initial_F, labeled_indices, clamp_target,
        max_iterations=200, tolerance=1e-9)

    assert iterations >= 1
    assert delta < 1e-9, ("propagation did not converge", delta)
    assert torch.isfinite(final_F).all()

    pseudo_labels, confidence = labelprop.pseudo_labels_from(final_F)

    assert int(pseudo_labels[0].item()) == 0
    assert int(pseudo_labels[1].item()) == 0, "cluster A not recovered"
    assert int(pseudo_labels[2].item()) == 0, "cluster A not recovered"
    assert int(pseudo_labels[3].item()) == 1
    assert int(pseudo_labels[4].item()) == 1, "cluster B not recovered"
    assert int(pseudo_labels[5].item()) == -1, "unreachable node was not marked -1"
    assert float(confidence[5].item()) == 0.0

    # THE RE-CLAMP PROOF: labeled rows are still exactly one-hot.
    assert torch.equal(final_F[0], torch.tensor([1.0, 0.0]))
    assert torch.equal(final_F[3], torch.tensor([0.0, 1.0]))

    # And the unreachable row never received any mass.
    assert float(final_F[5].abs().sum().item()) == 0.0

    # Without the re-clamp the seeds would bleed away; confirm the clamp is not a
    # no-op by checking a seed's neighbours actually moved.
    assert float(final_F[1].sum().item()) > 0.0

    summary = labelprop.propagation_summary(final_F, pseudo_labels)
    assert summary["num_nodes"] == 6
    assert summary["unreachable_count"] == 1
    assert abs(summary["unreachable_fraction"] - 1.0 / 6.0) < 1e-12
    assert summary["class_counts"] == [3, 2]
    assert 0.0 <= summary["confidence_mean"] <= 1.0


@smoke_test
def test_propagate_does_not_alias_or_mutate_initial_F():
    """REGRESSION. `propagate` must work on its own copy of F0.

    `initial_F.to(device, dtype)` returns the SAME tensor when both already
    match, which is the normal case. Without a clone, `current_F` is the
    caller's `initial_F`, so:
      * the caller's F0 is silently overwritten by the clamp, and
      * a `labeled_onehot` that is a VIEW of F0 makes the clamp an index_put_
        with overlapping source and destination, which
        `torch.use_deterministic_algorithms(True)` -- on in every script --
        rejects with a RuntimeError.

    The second symptom is the dangerous one: it would surface hours into a sweep
    on the server, not here.
    """
    propagation = torch.zeros(6, 6)
    propagation[2, 0] = 1.0
    propagation[4, 3] = 1.0

    initial_F = labelprop.initial_label_matrix(6, 2, [0, 3], [0, 1], DEVICE)
    snapshot = initial_F.clone()

    # A SLICE clamp target is a view of initial_F -- the aliasing case.
    final_F, _, _ = labelprop.propagate(
        propagation, initial_F, [0, 1], initial_F[:2],
        max_iterations=5, tolerance=1e-9)

    assert torch.equal(initial_F, snapshot), "propagate mutated the caller's initial_F"
    assert final_F.data_ptr() != initial_F.data_ptr()

    # And the same must hold under the determinism mode the scripts enable.
    torch.use_deterministic_algorithms(True)
    try:
        labelprop.propagate(propagation, initial_F, [0, 1], initial_F[:2],
                            max_iterations=5, tolerance=1e-9)
    finally:
        torch.use_deterministic_algorithms(False)

    assert torch.equal(initial_F, snapshot)


@smoke_test
def test_label_propagation_end_to_end_from_embeddings():
    """The real pipeline: embeddings -> P -> propagate -> pseudo-labels.

    Three well-separated clusters, only two of which carry a seed, so the third
    must come back entirely as -1. This is the pipeline-level version of the
    "unreachable node" check and it also exercises build_propagation_matrix,
    unionfind and the coverage metric together.
    """
    generator = torch.Generator().manual_seed(11)
    points_per_cluster = 6
    latent_dimension = 16

    clusters = []
    for cluster_index in range(3):
        center = torch.zeros(latent_dimension)
        center[cluster_index] = 5.0
        clusters.append(center + 0.3 * torch.randn(
            points_per_cluster, latent_dimension, generator=generator))
    embeddings = torch.cat(clusters, dim=0)
    number_of_nodes = embeddings.shape[0]

    propagation, neighbor_indices, neighbor_similarities, bandwidth = \
        knn.build_propagation_matrix(embeddings, k=3, chunk_rows=5)

    assert bandwidth > 0.0
    assert tuple(neighbor_indices.shape) == (number_of_nodes, 3)
    assert propagation.is_sparse
    dense_propagation = propagation.to_dense()
    assert torch.isfinite(dense_propagation).all()
    non_empty = dense_propagation.sum(dim=1) > 0
    assert torch.allclose(dense_propagation.sum(dim=1)[non_empty],
                          torch.ones(int(non_empty.sum().item())), atol=1e-6)

    # Every neighbour must be inside the node's own cluster at this separation.
    true_cluster = torch.arange(number_of_nodes) // points_per_cluster
    assert bool((true_cluster[neighbor_indices] == true_cluster.reshape(-1, 1)).all()), \
        "clusters are not separated enough for this test to mean anything"

    component_ids, number_of_components = unionfind.connected_components(
        neighbor_indices, number_of_nodes)
    assert number_of_components == 3, number_of_components

    labeled_indices = [0, points_per_cluster]  # one seed in cluster 0 and cluster 1
    labeled_labels = [0, 1]
    initial_F = labelprop.initial_label_matrix(
        number_of_nodes, 2, labeled_indices, labeled_labels, DEVICE)
    clamp_target = initial_F[torch.tensor(labeled_indices)]

    final_F, _, _ = labelprop.propagate(
        propagation, initial_F, labeled_indices, clamp_target,
        max_iterations=config.LP_MAX_ITERATIONS, tolerance=config.LP_TOLERANCE)

    pseudo_labels, _ = labelprop.pseudo_labels_from(final_F)

    for node in range(points_per_cluster):
        assert int(pseudo_labels[node].item()) == 0, node
    for node in range(points_per_cluster, 2 * points_per_cluster):
        assert int(pseudo_labels[node].item()) == 1, node
    for node in range(2 * points_per_cluster, 3 * points_per_cluster):
        assert int(pseudo_labels[node].item()) == -1, (node, "unseeded component")

    labeled_mask = numpy.zeros(number_of_nodes, dtype=bool)
    labeled_mask[labeled_indices] = True
    coverage = metrics.component_label_coverage(component_ids, labeled_mask)
    assert abs(coverage - 2.0 / 3.0) < 1e-9, coverage

    purity = metrics.edge_purity(neighbor_indices, true_cluster)
    assert abs(purity - 1.0) < 1e-9, purity


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@smoke_test
def test_metrics_against_hand_computed_values():
    """Worked by hand on paper; see the arithmetic in the comments.

        true = [0, 0, 1, 1, 2, 2, 2]
        pred = [0, 1, 1, 1, 2, 2, 0]

        confusion   [[1, 1, 0],
                     [0, 2, 0],
                     [1, 0, 2]]

        accuracy    (1 + 2 + 2) / 7 = 5/7
        precision   [1/2, 2/3, 1]        (column sums 2, 3, 2)
        recall      [1/2, 1,   2/3]      (row sums    2, 2, 3)
        f1          [0.5, 0.8, 0.8]
        macro f1    2.1 / 3 = 0.7
    """
    true_labels = [0, 0, 1, 1, 2, 2, 2]
    predicted_labels = [0, 1, 1, 1, 2, 2, 0]

    confusion = metrics.confusion_matrix(true_labels, predicted_labels, 3)
    expected = torch.tensor([[1, 1, 0], [0, 2, 0], [1, 0, 2]], dtype=confusion.dtype)
    assert torch.equal(confusion, expected), confusion

    assert abs(metrics.accuracy(confusion) - 5.0 / 7.0) < 1e-12

    precision, recall, f1 = metrics.per_class_precision_recall_f1(confusion)
    assert torch.allclose(precision, torch.tensor([0.5, 2.0 / 3.0, 1.0], dtype=torch.float64))
    assert torch.allclose(recall, torch.tensor([0.5, 1.0, 2.0 / 3.0], dtype=torch.float64))
    assert torch.allclose(f1, torch.tensor([0.5, 0.8, 0.8], dtype=torch.float64))

    assert abs(metrics.macro_f1(confusion) - 0.7) < 1e-12
    assert abs(metrics.macro_precision(confusion) - (0.5 + 2.0 / 3.0 + 1.0) / 3.0) < 1e-12

    # A class that never appears keeps its all-zero row/column (minlength) and
    # contributes 0 to the macro average, matching sklearn's zero_division=0.
    padded = metrics.confusion_matrix(true_labels, predicted_labels, 5)
    assert tuple(padded.shape) == (5, 5)
    assert abs(metrics.accuracy(padded) - 5.0 / 7.0) < 1e-12
    assert abs(metrics.macro_f1(padded) - 2.1 / 5.0) < 1e-12

    # The -1 sentinel must be refused rather than folded into class 0.
    try:
        metrics.confusion_matrix([-1, 0], [0, 0], 3)
    except ValueError:
        pass
    else:
        raise AssertionError("confusion_matrix accepted the -1 sentinel")


@smoke_test
def test_cross_entropy_loss():
    probabilities = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=torch.float64)
    labels = [0, 1]
    expected = float(-(numpy.log(0.7) + numpy.log(0.8)) / 2.0)
    assert abs(metrics.cross_entropy_loss(probabilities, labels) - expected) < 1e-9

    # A zero probability on the true class must clip, not return inf.
    certain = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    value = metrics.cross_entropy_loss(certain, [1, 0])
    assert numpy.isfinite(value) and value > 0.0


@smoke_test
def test_edge_purity_and_coverage():
    neighbor_indices = torch.tensor([[1], [0], [3], [2]])

    assert abs(metrics.edge_purity(neighbor_indices, [0, 0, 1, 1]) - 1.0) < 1e-12
    assert abs(metrics.edge_purity(neighbor_indices, [0, 1, 1, 1]) - 0.5) < 1e-12

    # Only edges with two known endpoints count.
    mask = numpy.array([True, True, False, False])
    assert abs(metrics.edge_purity(neighbor_indices, [0, 0, 1, 1], valid_mask=mask) - 1.0) < 1e-12

    component_ids = numpy.array([0, 0, 1, 1])
    assert abs(metrics.component_label_coverage(
        component_ids, numpy.array([True, False, False, False])) - 0.5) < 1e-12
    assert abs(metrics.component_label_coverage(
        component_ids, numpy.array([True, False, False, True])) - 1.0) < 1e-12
    assert abs(metrics.component_label_coverage(
        component_ids, numpy.zeros(4, dtype=bool)) - 0.0) < 1e-12


# ---------------------------------------------------------------------------
# Union-find
# ---------------------------------------------------------------------------


@smoke_test
def test_union_find_and_connected_components():
    disjoint_set = unionfind.UnionFind(6)
    assert disjoint_set.number_of_components == 6
    assert disjoint_set.union(0, 1) is True
    assert disjoint_set.union(1, 2) is True
    assert disjoint_set.union(0, 2) is False, "a redundant union must be a no-op"
    assert disjoint_set.union(3, 4) is True
    assert disjoint_set.number_of_components == 3  # {0,1,2}, {3,4}, {5}

    assert disjoint_set.find(0) == disjoint_set.find(2)
    assert disjoint_set.find(0) != disjoint_set.find(3)
    assert disjoint_set.component_id(1) == disjoint_set.find(1)

    sizes = disjoint_set.component_sizes()
    assert sorted(sizes.values()) == [1, 2, 3], sizes

    # 6 nodes, three components: {0,1}, {2,3}, {4,5}.
    neighbor_indices = numpy.array([[1], [0], [3], [2], [5], [4]])
    component_ids, number_of_components = unionfind.connected_components(neighbor_indices, 6)

    assert number_of_components == 3, number_of_components
    assert component_ids.dtype == numpy.int64
    assert component_ids.shape == (6,)
    # Contiguous 0..M-1 relabelling, which metrics.component_label_coverage needs.
    assert sorted(set(component_ids.tolist())) == [0, 1, 2]
    assert component_ids[0] == component_ids[1]
    assert component_ids[2] == component_ids[3]
    assert component_ids[4] == component_ids[5]
    assert component_ids[0] != component_ids[2]

    # A long chain must not blow the recursion limit (find is iterative).
    chain = numpy.arange(1, 2001).reshape(-1, 1)
    chain = numpy.concatenate([chain, numpy.array([[1999]])], axis=0)
    chain_ids, chain_components = unionfind.connected_components(chain, 2001)
    assert chain_components == 1, chain_components
    assert len(set(chain_ids.tolist())) == 1

    # A torch tensor input must work too (that is what build_knn_graph returns).
    torch_ids, torch_components = unionfind.connected_components(
        torch.tensor(neighbor_indices), 6)
    assert torch_components == 3
    assert numpy.array_equal(torch_ids, component_ids)


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------


@smoke_test
def test_standardizer():
    features = torch.tensor([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]], dtype=torch.float64)
    standardizer = probe.Standardizer().fit(features)

    assert torch.allclose(standardizer.mean, torch.tensor([3.0, 10.0], dtype=torch.float64))
    # ddof = 0, matching StandardScaler.
    assert abs(float(standardizer.standard_deviation[0].item())
               - float(numpy.std([1.0, 3.0, 5.0]))) < 1e-12
    # The constant column gets scale 1.0, not a division by ~0.
    assert float(standardizer.standard_deviation[1].item()) == 1.0

    transformed = standardizer.transform(features)
    assert torch.isfinite(transformed).all()
    assert abs(float(transformed[:, 0].mean().item())) < 1e-12
    assert float(transformed[:, 1].abs().max().item()) == 0.0


@smoke_test
def test_logistic_regression_separates_a_trivial_toy_set():
    generator = numpy.random.default_rng(3)
    per_class = 40
    class_a = generator.normal(loc=[-4.0, -4.0], scale=0.3, size=(per_class, 2))
    class_b = generator.normal(loc=[4.0, 4.0], scale=0.3, size=(per_class, 2))

    features = numpy.concatenate([class_a, class_b], axis=0)
    labels = numpy.concatenate([numpy.zeros(per_class), numpy.ones(per_class)]).astype(numpy.int64)

    weights, bias = probe.fit_logistic_regression(
        features, labels, C=1.0, max_iterations=200, num_classes=2)

    assert tuple(weights.shape) == (2, 2), tuple(weights.shape)
    assert tuple(bias.shape) == (2,)
    assert torch.isfinite(weights).all() and torch.isfinite(bias).all()

    predictions = probe.predict(features, weights, bias)
    assert torch.equal(predictions, torch.as_tensor(labels)), "trivially separable set missed"

    probabilities = probe.predict_proba(features, weights, bias)
    assert tuple(probabilities.shape) == (2 * per_class, 2)
    assert torch.allclose(probabilities.sum(dim=1),
                          torch.ones(2 * per_class, dtype=torch.float64), atol=1e-10)

    scores = probe.evaluate_probe(features, labels, weights, bias, num_classes=2)
    assert abs(scores["accuracy"] - 1.0) < 1e-12
    assert abs(scores["f1_macro"] - 1.0) < 1e-12
    assert scores["log_loss"] >= 0.0

    # The -1 sentinel must be refused rather than fitted on.
    try:
        probe.fit_logistic_regression(features, labels - 1, C=1.0, num_classes=2)
    except ValueError:
        pass
    else:
        raise AssertionError("fit_logistic_regression accepted the -1 sentinel")


@smoke_test
def test_stratified_k_fold_and_c_selection():
    labels = numpy.concatenate([numpy.zeros(25), numpy.ones(15)]).astype(numpy.int64)
    splits = probe.stratified_k_fold(labels, num_splits=5, seed=42)

    assert len(splits) == 5
    covered = []
    for train_positions, validation_positions in splits:
        assert len(numpy.intersect1d(train_positions, validation_positions)) == 0
        assert len(train_positions) + len(validation_positions) == len(labels)
        # Stratified: every class present in every validation fold.
        assert int((labels[validation_positions] == 0).sum()) >= 1
        assert int((labels[validation_positions] == 1).sum()) >= 1
        covered.append(validation_positions)

    all_validation = numpy.sort(numpy.concatenate(covered))
    assert numpy.array_equal(all_validation, numpy.arange(len(labels))), \
        "the k folds must partition the data exactly once"

    repeat = probe.stratified_k_fold(labels, num_splits=5, seed=42)
    for (_, first), (_, second) in zip(splits, repeat):
        assert numpy.array_equal(first, second), "stratified_k_fold is not deterministic"

    # Selection order is Lior's: LOG LOSS ascending is the PRIMARY criterion,
    # then accuracy descending, then macro-F1 descending, then the smaller C.
    # Log loss first is not cosmetic -- a barely-regularized large C routinely
    # wins on accuracy while being over-confident and losing on log loss, so
    # ranking on accuracy first would systematically pick a larger C than his arm.
    rows = [{"c": 0.1, "log_loss_mean": 0.50, "accuracy_mean": 0.80, "f1_macro_mean": 0.80},
            {"c": 10.0, "log_loss_mean": 0.90, "accuracy_mean": 0.95, "f1_macro_mean": 0.95}]
    assert probe.select_best_c(rows) == 0.1, "log loss must beat accuracy"

    # Full tie on log loss -> higher accuracy wins.
    tied = [{"c": 0.1, "log_loss_mean": 0.5, "accuracy_mean": 0.80, "f1_macro_mean": 0.80},
            {"c": 1.0, "log_loss_mean": 0.5, "accuracy_mean": 0.90, "f1_macro_mean": 0.80}]
    assert probe.select_best_c(tied) == 1.0

    # Full tie everywhere -> the SMALLER (more regularized) C wins.
    identical = [{"c": 10.0, "log_loss_mean": 0.5, "accuracy_mean": 0.9, "f1_macro_mean": 0.9},
                 {"c": 0.1, "log_loss_mean": 0.5, "accuracy_mean": 0.9, "f1_macro_mean": 0.9}]
    assert probe.select_best_c(identical) == 0.1

    try:
        probe.select_best_c([])
    except ValueError:
        pass
    else:
        raise AssertionError("select_best_c accepted an empty result list")


@smoke_test
def test_select_best_c_across_folds():
    """One global C for all 10 folds -- the protocol Lior's arm actually uses.

    Picking a C per fold would give our arm ten chances to find a good
    regularization strength where his arm gets one, which is a quietly more
    permissive protocol and not a fair row in the comparison table.
    """
    fold_one = [{"c": 0.1, "log_loss_mean": 0.9, "accuracy_mean": 0.7, "f1_macro_mean": 0.7},
                {"c": 1.0, "log_loss_mean": 0.5, "accuracy_mean": 0.8, "f1_macro_mean": 0.8}]
    fold_two = [{"c": 0.1, "log_loss_mean": 0.1, "accuracy_mean": 0.9, "f1_macro_mean": 0.9},
                {"c": 1.0, "log_loss_mean": 0.7, "accuracy_mean": 0.6, "f1_macro_mean": 0.6}]

    selected_c, aggregate_rows = probe.select_best_c_across_folds([fold_one, fold_two])

    assert len(aggregate_rows) == 2
    by_c = {row["c"]: row for row in aggregate_rows}
    assert abs(by_c[0.1]["log_loss_mean"] - 0.5) < 1e-12   # mean(0.9, 0.1)
    assert abs(by_c[1.0]["log_loss_mean"] - 0.6) < 1e-12   # mean(0.5, 0.7)
    assert abs(by_c[0.1]["accuracy_mean"] - 0.8) < 1e-12
    # 0.1 wins on the AVERAGED log loss even though it loses fold one outright.
    assert selected_c == 0.1, selected_c

    try:
        probe.select_best_c_across_folds([])
    except ValueError:
        pass
    else:
        raise AssertionError("select_best_c_across_folds accepted an empty list")


@smoke_test
def test_cross_validate_c_runs():
    generator = numpy.random.default_rng(5)
    per_class = 25
    features = numpy.concatenate([
        generator.normal(loc=[-3.0, 0.0], scale=1.0, size=(per_class, 2)),
        generator.normal(loc=[3.0, 0.0], scale=1.0, size=(per_class, 2)),
    ], axis=0)
    labels = numpy.concatenate([numpy.zeros(per_class), numpy.ones(per_class)]).astype(numpy.int64)

    rows = probe.cross_validate_c(
        features, labels, c_grid=(0.1, 1.0), num_splits=5, seed=0,
        max_iterations=100, num_classes=2)

    assert len(rows) == 2
    for row in rows:
        assert set(row) == {"c", "log_loss_mean", "accuracy_mean", "f1_macro_mean"}
        assert 0.0 <= row["accuracy_mean"] <= 1.0
        assert numpy.isfinite(row["log_loss_mean"])
        assert row["accuracy_mean"] > 0.8, row

    best_c = probe.select_best_c(rows)
    assert best_c in (0.1, 1.0)


# ---------------------------------------------------------------------------
# Optimizer schedule, augmentation, seeding
# ---------------------------------------------------------------------------


@smoke_test
def test_learning_rate_schedule():
    total_steps, warmup_steps = 100, 10
    max_lr, min_lr = 5e-4, 1e-5

    first = optim.learning_rate_at_step(0, total_steps, warmup_steps, max_lr, min_lr)
    assert first > 0.0, "step 0 must not waste a zero-length step"
    assert abs(first - max_lr / warmup_steps) < 1e-15

    peak = optim.learning_rate_at_step(warmup_steps - 1, total_steps, warmup_steps, max_lr, min_lr)
    assert abs(peak - max_lr) < 1e-15, "warmup must end exactly at max_lr"

    schedule = [optim.learning_rate_at_step(step, total_steps, warmup_steps, max_lr, min_lr)
                for step in range(total_steps)]
    assert all(later >= earlier for earlier, later in zip(schedule[:warmup_steps],
                                                         schedule[1:warmup_steps]))
    assert all(later <= earlier + 1e-15 for earlier, later in zip(schedule[warmup_steps:],
                                                                 schedule[warmup_steps + 1:]))
    assert abs(schedule[-1] - min_lr) < 1e-12, "the last step must land exactly on min_lr"
    # Past the end it must stay at min_lr rather than climbing the far side.
    assert abs(optim.learning_rate_at_step(
        total_steps + 50, total_steps, warmup_steps, max_lr, min_lr) - min_lr) < 1e-12

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    optimizer = optim.build_optimizer(model, weight_decay=1e-4)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    # Only the 2-D Linear weight is decayed; the bias and both BN vectors are not.
    assert all(parameter.ndim > 1 for parameter in optimizer.param_groups[0]["params"])
    assert all(parameter.ndim == 1 for parameter in optimizer.param_groups[1]["params"])

    optim.set_learning_rate(optimizer, 0.123)
    assert all(group["lr"] == 0.123 for group in optimizer.param_groups)


@smoke_test
def test_augmentation():
    generator = torch.Generator().manual_seed(0)
    batch = torch.randn(8, 3, 96, 96)

    flipped = augment.random_horizontal_flip(batch, probability=1.0, generator=generator)
    assert torch.equal(flipped, torch.flip(batch, dims=(3,)))
    assert torch.equal(augment.random_horizontal_flip(batch, probability=0.0), batch)

    cropped = augment.random_resized_crop(
        batch, output_size=96, scale=(0.8, 1.0), ratio=(0.9, 1.1), generator=generator)
    assert tuple(cropped.shape) == (8, 3, 96, 96)
    assert torch.isfinite(cropped).all()

    augmented = augment.augment_batch(batch, generator=torch.Generator().manual_seed(1))
    assert tuple(augmented.shape) == (8, 3, 96, 96)
    assert torch.isfinite(augmented).all()

    # Reproducible from the generator seed alone.
    again = augment.augment_batch(batch, generator=torch.Generator().manual_seed(1))
    assert torch.equal(augmented, again)

    # uint8 must be refused: bilinear resampling on uint8 is undefined.
    try:
        augment.random_resized_crop(
            torch.zeros(2, 3, 96, 96, dtype=torch.uint8), 96, (0.8, 1.0), (0.9, 1.1))
    except ValueError:
        pass
    else:
        raise AssertionError("random_resized_crop accepted a uint8 batch")


@smoke_test
def test_seeding_is_reproducible():
    seeding.reset_seed(42)
    first = torch.randn(5)
    seeding.reset_seed(42)
    assert torch.equal(first, torch.randn(5))

    generator_a = seeding.seeded_generator(7)
    generator_b = seeding.seeded_generator(7)
    assert torch.equal(torch.randn(3, generator=generator_a),
                       torch.randn(3, generator=generator_b))


# ---------------------------------------------------------------------------
# Downstream classifier (the pipeline glue)
# ---------------------------------------------------------------------------


@smoke_test
def test_classifier_trains_one_epoch_on_pseudo_labels():
    """The end of the pipeline: pseudo-labels -> CNN -> confusion matrix.

    Tiny on purpose (8 images, 2 steps). The point is that the glue runs at all:
    the -1 filter, the confidence weighting, per-batch normalization, the
    schedule, and the evaluation path.
    """
    seeding.reset_seed(0)

    generator = numpy.random.default_rng(0)
    images = generator.integers(0, 256, size=(8, 3, 96, 96), dtype=numpy.uint8)
    pseudo_labels = numpy.array([0, 1, 2, 3, 4, 5, 6, -1], dtype=numpy.int64)
    confidence = numpy.linspace(0.1, 1.0, 8)

    normalization_mean = [0.45, 0.44, 0.40]
    normalization_standard_deviation = [0.26, 0.26, 0.27]

    model = classifier.PseudoLabelClassifier()
    history = classifier.train_classifier(
        model, images, pseudo_labels,
        normalization_mean, normalization_standard_deviation,
        device=DEVICE, sample_weights=confidence,
        epochs=1, batch_size=4, warmup_epochs=1, seed=0)

    assert len(history) == 1
    record = history[0]
    # 7 usable samples, batch 4 -> 1 full step; the ragged tail is dropped.
    assert record["samples"] == 4, record
    assert numpy.isfinite(record["loss"])
    assert 0.0 <= record["pseudo_label_accuracy"] <= 1.0
    assert record["first_learning_rate"] > 0.0

    # The -1 sentinel must never reach the loss.
    usable = classifier._usable_positions(pseudo_labels, 10, -1)
    assert numpy.array_equal(usable, numpy.arange(7))

    probabilities = classifier.predict_probabilities(
        model, images, normalization_mean, normalization_standard_deviation,
        device=DEVICE, batch_size=4)
    assert tuple(probabilities.shape) == (8, 10)
    assert probabilities.dtype == torch.float64
    assert torch.allclose(probabilities.sum(dim=1),
                          torch.ones(8, dtype=torch.float64), atol=1e-9)

    true_labels = numpy.arange(8) % 10
    confusion = classifier.evaluate(
        model, images, true_labels,
        normalization_mean, normalization_standard_deviation, device=DEVICE, batch_size=4)
    assert tuple(confusion.shape) == (10, 10)
    assert int(confusion.sum().item()) == 8
    assert 0.0 <= metrics.accuracy(confusion) <= 1.0

    # A run where label propagation reached nothing must fail loudly.
    try:
        classifier._usable_positions(numpy.full(4, -1), 10, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("an all-sentinel pseudo-label array was accepted")


@smoke_test
def test_classifier_loads_a_trunk_checkpoint_strictly():
    """`strict=True` loading is the cheap proof of trunk parity; it must work."""
    encoder = trunk.SEResNetEncoder()
    encoder_state = encoder.state_dict()

    from_bare = classifier.PseudoLabelClassifier.from_pretrained_encoder(encoder_state)
    for key, value in encoder_state.items():
        assert torch.equal(from_bare.encoder.state_dict()[key], value), key

    # A whole-VAE state dict carries an "encoder." prefix that must be stripped.
    prefixed = {"encoder." + key: value for key, value in encoder_state.items()}
    prefixed["decoder.latent_projection.weight"] = torch.zeros(1)
    from_prefixed = classifier.PseudoLabelClassifier.from_pretrained_encoder(prefixed)
    for key, value in encoder_state.items():
        assert torch.equal(from_prefixed.encoder.state_dict()[key], value), key

    # A frozen encoder must stay in eval mode even when the model is train()ed,
    # or its BatchNorm running statistics would keep drifting.
    frozen = classifier.PseudoLabelClassifier(freeze_encoder=True)
    frozen.train()
    assert not frozen.encoder.training
    assert all(not parameter.requires_grad for parameter in frozen.encoder.parameters())
    assert frozen.classification_head.weight.requires_grad


@smoke_test
def test_vae_encoder_accepts_the_trunk_state_dict():
    """The VAE's encoder must be the trunk, not a copy that drifted."""
    encoder_state = trunk.SEResNetEncoder().state_dict()
    model = vae.VariationalAutoencoder()
    model.encoder.load_state_dict(encoder_state, strict=True)
    assert model.to_mu.out_features == 512
    assert model.to_logvar.out_features == 512


@smoke_test
def test_global_pool_lowers_to_a_deterministic_mean():
    """AdaptiveAvgPool2d(1) must lower to `mean`, not to the pooling kernel.

    This one is about the SERVER, not the laptop. Every script calls
    `seeding.enable_deterministic()`, which makes PyTorch RAISE on any op with no
    deterministic CUDA kernel, and `AdaptiveAvgPool2d` backward is on that list.
    PyTorch special-cases an output size of 1x1 into `input.mean({-1, -2})`,
    which IS deterministic -- so the trunk's `global_pool` and the SE block's
    `pool` are safe. The lowering happens in the shared dispatch, so checking the
    grad_fn on CPU proves it for CUDA too.

    If a future edit ever changed the pooled output size away from 1, the whole
    100-epoch pretraining run would die on the server at the first backward pass
    and this test is the cheap early warning.
    """
    pooled = torch.nn.AdaptiveAvgPool2d(1)(torch.randn(2, 4, 6, 6, requires_grad=True))
    assert type(pooled.grad_fn).__name__ == "MeanBackward1", type(pooled.grad_fn).__name__

    encoder = trunk.SEResNetEncoder()
    embeddings = encoder(torch.randn(2, 3, 96, 96, requires_grad=True))
    # flatten(1) -> the pool. The pool must already be a mean by this point.
    assert type(embeddings.grad_fn.next_functions[0][0]).__name__ == "MeanBackward1"


@smoke_test
def test_pipeline_runs_under_enable_deterministic():
    """Run a slice of the pipeline with `use_deterministic_algorithms(True)` on.

    Every script in `scripts/` calls `seeding.enable_deterministic()` before it
    touches data, so this is the mode the real runs are in. Anything that raises
    here raises on the 5090 too -- and it would raise hours into a run rather
    than in the first second.

    Registered LAST on purpose: the flag is global process state and turning it
    on would otherwise leak into every test after it.
    """
    seeding.enable_deterministic()
    try:
        assert torch.are_deterministic_algorithms_enabled()

        seeding.reset_seed(config.SEED)

        # Encoder + VAE forward AND backward -- the pretraining hot path.
        model = vae.VariationalAutoencoder()
        model.train()
        batch = torch.randn(2, 3, 96, 96)
        reconstruction, mu, logvar = model(batch)
        total, _, _ = vae.vae_loss(reconstruction, batch, mu, logvar, beta=0.01)
        total.backward()
        assert torch.isfinite(total)
        assert any(parameter.grad is not None for parameter in model.parameters())

        # Augmentation (bilinear interpolate) on the normalized batch.
        images = numpy.random.default_rng(0).integers(
            0, 256, size=(4, 3, 96, 96), dtype=numpy.uint8)
        normalized = data.normalize_batch(images, [0.45] * 3, [0.26] * 3, DEVICE)
        augmented = augment.augment_batch(
            normalized, generator=seeding.seeded_generator(config.SEED))
        assert torch.isfinite(augmented).all()

        # Graph -> propagation -> pseudo-labels -> metrics, the sweep's inner loop.
        generator = torch.Generator().manual_seed(2)
        embeddings = torch.randn(60, 24, generator=generator)
        propagation, neighbor_indices, _, _ = knn.build_propagation_matrix(
            embeddings, k=5, chunk_rows=16)

        initial_F = labelprop.initial_label_matrix(
            60, 10, list(range(10)), list(range(10)), DEVICE)
        final_F, _, _ = labelprop.propagate(
            propagation, initial_F, list(range(10)), initial_F[:10],
            max_iterations=config.LP_MAX_ITERATIONS, tolerance=config.LP_TOLERANCE)
        pseudo_labels, _ = labelprop.pseudo_labels_from(final_F)
        assert torch.isfinite(final_F).all()

        summary = labelprop.propagation_summary(final_F, pseudo_labels)
        assert summary["num_nodes"] == 60

        component_ids, _ = unionfind.connected_components(neighbor_indices, 60)
        labeled_mask = numpy.zeros(60, dtype=bool)
        labeled_mask[:10] = True
        assert 0.0 <= metrics.component_label_coverage(component_ids, labeled_mask) <= 1.0

        reachable = pseudo_labels >= 0
        confusion = metrics.confusion_matrix(
            torch.zeros(int(reachable.sum().item()), dtype=torch.int64),
            pseudo_labels[reachable], 10)
        assert int(confusion.sum().item()) == int(reachable.sum().item())

        # The probe's L-BFGS fit.
        features = numpy.concatenate([
            numpy.random.default_rng(1).normal(-3.0, 0.5, size=(20, 3)),
            numpy.random.default_rng(2).normal(3.0, 0.5, size=(20, 3))], axis=0)
        labels = numpy.concatenate([numpy.zeros(20), numpy.ones(20)]).astype(numpy.int64)
        weights, bias = probe.fit_logistic_regression(
            features, labels, C=1.0, max_iterations=100, num_classes=2)
        assert torch.isfinite(weights).all()
    finally:
        # Leave the process as we found it, in case anything runs after us.
        torch.use_deterministic_algorithms(False)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(0)
    # Keep the CPU run single-threaded-ish so timings are stable on the laptop.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    passed = []
    failed = []

    for test_function in REGISTERED_TESTS:
        name = test_function.__name__
        try:
            test_function()
        except Exception:
            failed.append(name)
            print("FAIL  " + name)
            print(traceback.format_exc())
        else:
            passed.append(name)
            print("PASS  " + name)

    print("")
    print("=" * 72)
    print("%d passed, %d failed, %d total" % (len(passed), len(failed), len(REGISTERED_TESTS)))
    if failed:
        print("failed: " + ", ".join(failed))
    print("=" * 72)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
