"""STL-10 loading with no torchvision.

The lecturer's "build from 0" rule bans torchvision, so `torchvision.datasets.STL10`
is off the table and we parse the raw `stl10_binary/` files ourselves. That is also
the cheaper option on the lab server, where disk is scarce and pulling the whole
torchvision wheel just to read seven files is wasteful.

Everything here is deliberately kept bit-compatible with Lior's SimCLR notebook so
the two arms of the comparison share the same pixels, the same normalization
constants and the same folds. The one place we cannot be bit-compatible is the
80/20 development split, because it used sklearn -- see `stratified_split`.
"""

import os

import numpy
import torch


IMAGE_SIZE = 96
NUMBER_OF_CHANNELS = 3
NUMBER_OF_CLASSES = 10

# Sizes of the three official splits, used only as sanity checks after reading.
EXPECTED_SPLIT_SIZES = {"train": 5000, "test": 8000, "unlabeled": 100000}

SPLIT_FILE_NAMES = {
    "train": ("train_X.bin", "train_y.bin"),
    "test": ("test_X.bin", "test_y.bin"),
    "unlabeled": ("unlabeled_X.bin", None),
}


def resolve_binary_directory(data_root):
    """Return the directory that actually holds the .bin files.

    Callers are inconsistent about whether `data_root` means the parent that
    torchvision would have created (`.../data`) or the extracted archive itself
    (`.../data/stl10_binary`). Accepting both removes a whole class of silly
    path bugs when the same scripts run on the laptop and on the server.
    """
    nested = os.path.join(data_root, "stl10_binary")
    if os.path.isdir(nested):
        return nested
    return data_root


def load_images(path):
    """Read one STL-10 image binary into a uint8 array [N, 3, 96, 96].

    !!! THE TRANSPOSE BELOW IS NOT OPTIONAL !!!

    STL-10 stores each image COLUMN-MAJOR (Fortran order) inside a row-major
    byte stream. Reshaping to [N, 3, 96, 96] therefore lands the pixels in
    [N, C, W, H] order, not [N, C, H, W]. Swapping the last two axes undoes it.

    If you drop the transpose, every image is rotated/mirrored by a 90-degree
    transpose. Nothing raises, the VAE trains happily, the KNN graph builds,
    label propagation converges -- and every number in the report is quietly
    wrong. This is the single easiest way to ruin the whole project, hence the
    shouting.
    """
    raw = numpy.fromfile(path, dtype=numpy.uint8)
    images = raw.reshape(-1, NUMBER_OF_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    images = numpy.transpose(images, (0, 1, 3, 2))  # <- undoes column-major; REQUIRED
    return images


def load_labels(path):
    """Read one STL-10 label binary into an int64 array [N] with values 0..9.

    The files store labels 1-indexed as uint8 (1..10). Subtracting one before the
    cast keeps them in range and matches what torchvision hands Lior, so class id
    3 means the same animal in both arms.
    """
    raw = numpy.fromfile(path, dtype=numpy.uint8)
    return raw.astype(numpy.int64) - 1


def load_images_memory_mapped(path):
    """Memory-map an image binary instead of reading it into RAM.

    Used for the 100k unlabeled split, which is 100000 * 3 * 96 * 96 = 2.76 GB as
    uint8. Materialising that is fine on the 5090 server but painful on the
    laptop during tests, so we hand back a lazy view and let the caller pull it in
    chunk by chunk.

    The returned array is a *non-contiguous* view: the column-major fix is applied
    as a transpose, which memmap represents by swapping strides rather than moving
    bytes. Anything that needs a contiguous buffer (`torch.from_numpy`, a `.cuda()`
    upload) must call `numpy.ascontiguousarray` on the slice it wants first.
    """
    raw = numpy.memmap(path, dtype=numpy.uint8, mode="r")
    images = raw.reshape(-1, NUMBER_OF_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    return numpy.transpose(images, (0, 1, 3, 2))  # same column-major fix as load_images


def load_split(data_root, split, memory_map_unlabeled=True):
    """Load one STL-10 split.

    Returns (images, labels) where images is uint8 [N, 3, 96, 96] and labels is
    int64 [N] for "train"/"test" or None for "unlabeled".

    Memory cost, so nobody is surprised on a 16 GB laptop:
      - train     5000 images ->  138 MB uint8, always read fully into RAM
      - test      8000 images ->  221 MB uint8, always read fully into RAM
      - unlabeled 100000      -> 2.76 GB uint8

    The unlabeled split is memory-mapped by default. Pass
    `memory_map_unlabeled=False` on the server, where CLAUDE.md's plan is to hold
    the whole thing resident and augment on-GPU without a DataLoader.
    """
    if split not in SPLIT_FILE_NAMES:
        raise ValueError("split must be one of %s, got %r"
                         % (sorted(SPLIT_FILE_NAMES), split))

    binary_directory = resolve_binary_directory(data_root)
    image_file_name, label_file_name = SPLIT_FILE_NAMES[split]
    image_path = os.path.join(binary_directory, image_file_name)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            "Could not find %s. Expected the extracted stl10_binary/ archive under %r."
            % (image_path, data_root))

    if split == "unlabeled" and memory_map_unlabeled:
        images = load_images_memory_mapped(image_path)
    else:
        images = load_images(image_path)

    expected_size = EXPECTED_SPLIT_SIZES[split]
    if len(images) != expected_size:
        raise ValueError("Split %r should hold %d images but %s decoded to %d. "
                         "A truncated download is the usual cause."
                         % (split, expected_size, image_path, len(images)))

    if label_file_name is None:
        return images, None

    labels = load_labels(os.path.join(binary_directory, label_file_name))
    if len(labels) != len(images):
        raise ValueError("Image/label count mismatch for split %r: %d vs %d"
                         % (split, len(images), len(labels)))

    return images, labels


def load_class_names(data_root):
    """Read class_names.txt into a list of 10 strings, in label order 0..9."""
    path = os.path.join(resolve_binary_directory(data_root), "class_names.txt")
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def compute_channel_statistics(images, chunk_size=512):
    """Per-RGB-channel mean and standard deviation, in [0, 1] units.

    A line-for-line reproduction of cell 6 of Lior's notebook. Both arms must
    normalize with identical constants or the "same embedding space" claim in the
    fairness section falls apart, so this is copied rather than reinvented.

    Chunked and accumulated in float64 for two reasons: the sum of squares over
    100k * 96 * 96 pixels overflows float32 long before it finishes, and casting
    the whole 2.76 GB unlabeled array to float64 at once would need 22 GB.

    Per CLAUDE.md the statistics come from the UNLABELED split only.

    Returns (mean, standard_deviation) as plain lists of 3 floats.
    """
    channel_sum = numpy.zeros(NUMBER_OF_CHANNELS, dtype=numpy.float64)
    channel_square_sum = numpy.zeros(NUMBER_OF_CHANNELS, dtype=numpy.float64)
    pixel_count = 0

    for start in range(0, len(images), chunk_size):
        end = min(start + chunk_size, len(images))
        chunk = numpy.asarray(images[start:end]).astype(numpy.float64, copy=True)

        channel_sum += chunk.sum(axis=(0, 2, 3))
        channel_square_sum += numpy.einsum("nchw,nchw->c", chunk, chunk, optimize=True)
        pixel_count += chunk.shape[0] * chunk.shape[2] * chunk.shape[3]

    mean = channel_sum / pixel_count / 255.0
    second_moment = channel_square_sum / pixel_count / (255.0 ** 2)
    variance = second_moment - numpy.square(mean)
    # Clamped at zero because catastrophic cancellation in E[x^2] - E[x]^2 can
    # produce a tiny negative variance, and sqrt of that is NaN.
    standard_deviation = numpy.sqrt(numpy.maximum(variance, 0.0))

    return mean.tolist(), standard_deviation.tolist()


def load_official_folds(data_root):
    """Parse fold_indices.txt into an int64 array [10, 1000].

    STL-10 ships 10 official 1000-image folds, each a list of indices into the
    5000-image labeled train split. Using these rather than our own random
    subsets is what makes our numbers comparable to Lior's and to the published
    STL-10 literature.
    """
    path = os.path.join(resolve_binary_directory(data_root), "fold_indices.txt")
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    # Lior used numpy.fromstring(line, sep=" "), which is deprecated and behaves
    # differently across numpy 1.x/2.x. split() + array is the same parse with no
    # version risk -- the resulting index arrays are identical.
    folds = [numpy.array(line.split(), dtype=numpy.int64)
             for line in text.strip().splitlines() if line.strip()]
    folds = numpy.stack(folds)

    if folds.shape != (10, 1000):
        raise ValueError("fold_indices.txt should parse to [10, 1000], got %r"
                         % (folds.shape,))
    return folds


def stratified_split(labels, validation_fraction, seed):
    """Hand-rolled stratified train/validation split.

    Replaces `sklearn.model_selection.train_test_split(..., stratify=labels)`,
    which build-from-0 forbids. The algorithm is the obvious one: for each class,
    take the positions holding that class, shuffle them with a `numpy.random`
    Generator seeded from `seed`, and cut at `validation_fraction`. Because the
    cut happens inside each class, the class proportions of the input survive
    into both halves -- that is all "stratified" means here.

    DOCUMENTED ASYMMETRY, NOT A BUG:
    This cannot bit-match Lior's sklearn split. The *protocol* is identical --
    stratified 80/20, seeded with SEED + fold_index -- but sklearn shuffles with
    its own `check_random_state` / `MT19937` permutation scheme, so the exact
    partition of the 1000 fold members into 800/200 differs from ours. Both are
    valid draws from the same distribution of stratified splits; only the draw
    differs. Model selection is a mean over 10 folds, so the effect on any
    reported number is noise, and the alternative (importing sklearn) is banned.
    Flag it in the report rather than pretending the splits are identical.

    Returns (train_positions, validation_positions), both int64 and sorted
    ascending. The values are positions into `labels`, not global dataset indices
    -- the caller composes them with the fold's official indices.
    """
    labels = numpy.asarray(labels)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1, got %r"
                         % (validation_fraction,))

    generator = numpy.random.default_rng(seed)

    train_positions = []
    validation_positions = []

    # Iterating over sorted unique classes keeps the result independent of the
    # order the labels happen to appear in, which is what makes this deterministic.
    for class_id in numpy.unique(labels):
        class_positions = numpy.flatnonzero(labels == class_id)
        shuffled = generator.permutation(class_positions)

        validation_size = int(round(len(shuffled) * validation_fraction))
        # Never let a class vanish from either half; a class with no validation
        # members would silently distort per-class metrics downstream.
        validation_size = max(1, min(len(shuffled) - 1, validation_size))

        validation_positions.append(shuffled[:validation_size])
        train_positions.append(shuffled[validation_size:])

    train_positions = numpy.sort(numpy.concatenate(train_positions)).astype(numpy.int64)
    validation_positions = numpy.sort(
        numpy.concatenate(validation_positions)).astype(numpy.int64)

    return train_positions, validation_positions


def build_and_save_splits(data_root, output_path, seed=42, validation_fraction=0.20):
    """Compute the 10 official folds and their 800/200 dev splits once, to a .npz.

    Every script in both arms loads this file instead of re-deriving splits. That
    is the only way to be sure the VAE arm and the SimCLR arm are scored on the
    same held-out images: a split recomputed in three places is a split that will
    eventually disagree in one of them.

    Fold `i` uses seed `seed + i`, mirroring Lior's `random_state=SEED + fold_index`.

    The stored positions index into the 1000 members of a fold, not into the 5000
    labeled train images -- compose them as
    `official_folds[fold_index][train_positions]` to get global indices. Keeping
    them fold-relative is what lets the file stay meaningful if the labeled split
    is ever re-read from disk in a different order.

    Returns the same dictionary `load_splits` would give back.
    """
    official_folds = load_official_folds(data_root)
    _, train_labels = load_split(data_root, "train")

    arrays = {
        "official_folds": official_folds,
        "train_labels": train_labels,
        "seed": numpy.asarray(seed, dtype=numpy.int64),
        "validation_fraction": numpy.asarray(validation_fraction, dtype=numpy.float64),
    }

    development_splits = {}
    for fold_index in range(len(official_folds)):
        fold_labels = train_labels[official_folds[fold_index]]
        train_positions, validation_positions = stratified_split(
            fold_labels,
            validation_fraction=validation_fraction,
            seed=seed + fold_index,
        )
        development_splits[fold_index] = (train_positions, validation_positions)
        arrays["fold_%d_train_positions" % fold_index] = train_positions
        arrays["fold_%d_validation_positions" % fold_index] = validation_positions

    # numpy.savez silently appends ".npz" when the path lacks it, which would then
    # not be the path the caller hands to load_splits. Normalise up front.
    if not output_path.endswith(".npz"):
        output_path = output_path + ".npz"

    output_directory = os.path.dirname(os.path.abspath(output_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    numpy.savez(output_path, **arrays)

    return {
        "official_folds": official_folds,
        "train_labels": train_labels,
        "development_splits": development_splits,
        "seed": seed,
        "validation_fraction": validation_fraction,
    }


def load_splits(path):
    """Load the .npz written by `build_and_save_splits`.

    Returns a dictionary with:
      official_folds       int64 [10, 1000]  indices into the labeled train split
      train_labels         int64 [5000]      0..9 labels of the labeled train split
      development_splits   {fold_index: (train_positions, validation_positions)}
      seed                 int
      validation_fraction  float
    """
    with numpy.load(path) as stored:
        official_folds = stored["official_folds"]
        train_labels = stored["train_labels"]
        seed = int(stored["seed"])
        validation_fraction = float(stored["validation_fraction"])

        development_splits = {}
        for fold_index in range(len(official_folds)):
            development_splits[fold_index] = (
                stored["fold_%d_train_positions" % fold_index],
                stored["fold_%d_validation_positions" % fold_index],
            )

    return {
        "official_folds": official_folds,
        "train_labels": train_labels,
        "development_splits": development_splits,
        "seed": seed,
        "validation_fraction": validation_fraction,
    }


def normalize_batch(images_uint8, mean, standard_deviation, device):
    """uint8 [B, 3, 96, 96] -> normalized float32 tensor on `device`.

    Order of operations matters and is chosen to match torchvision's
    `ToTensor()` followed by `Normalize(mean, std)`, which is what Lior's
    pipeline applies: scale to [0, 1] by dividing by 255 FIRST, then subtract the
    mean and divide by the standard deviation. Normalizing before the /255 would
    give a completely different scale, since our constants are in [0, 1] units.

    Accepts a numpy array (including a non-contiguous memmap view of the
    unlabeled split) or an existing torch tensor. `device` is always taken from
    the caller -- nothing here hardcodes .cuda(), so the same code path serves the
    CPU tests and the 5090.
    """
    if isinstance(images_uint8, torch.Tensor):
        source = images_uint8.to(device)
    else:
        # ascontiguousarray is required because the column-major fix leaves memmap
        # views with swapped strides, which torch.from_numpy will not accept.
        source = torch.from_numpy(numpy.ascontiguousarray(images_uint8)).to(device)

    # `.to(float32)` returns the SAME tensor when the input is already float32, and
    # the in-place sub_/div_ below would then rewrite the caller's pixels. Every
    # current call site passes uint8, where the cast allocates a fresh tensor and
    # this branch never fires -- but the aliasing is worth closing explicitly,
    # because on the server the 100k unlabeled images are one resident tensor and
    # every batch is a view into it. An aliased in-place normalize would corrupt
    # the dataset a batch at a time, silently, with nothing raising.
    images = source.to(torch.float32)
    if images.data_ptr() == source.data_ptr():
        images = images.clone()

    images = images.div_(255.0)

    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    standard_deviation_tensor = torch.as_tensor(
        standard_deviation, dtype=torch.float32, device=device).view(1, -1, 1, 1)

    return images.sub_(mean_tensor).div_(standard_deviation_tensor)
