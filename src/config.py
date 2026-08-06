"""Single source of truth for every constant in Tomer's VAE + Label Propagation arm.

No logic lives here beyond path construction. The reason for centralising is fairness:
the comparison against Lior's SimCLR arm is only meaningful if the trunk, the budget and
the evaluation protocol are bit-for-bit the same, so those values are written down once
and imported everywhere rather than retyped per script.

Values marked "ground truth from Lior's notebook" were read out of
simclr_stl10_abilation_f.ipynb (cells 4, 12 and 18). Do not change one without changing
the other arm, or the comparison stops being a comparison.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Reproducibility and data geometry -- ground truth from Lior's notebook cell 4
# ---------------------------------------------------------------------------

SEED = 42

IMAGE_SIZE = 96
NUM_CLASSES = 10

# STL-10 split sizes. The unlabeled split is deliberately broader than the 10 labeled
# classes, which is why label propagation diffuses through out-of-distribution nodes --
# a finding to quantify, not a bug to fix.
NUM_UNLABELED_IMAGES = 100000
NUM_TRAIN_IMAGES = 5000
NUM_TEST_IMAGES = 8000

# The label-propagation graph is built over unlabeled + labeled train, so every node the
# labels can reach is in one array.
NUM_GRAPH_NODES = NUM_UNLABELED_IMAGES + NUM_TRAIN_IMAGES


# ---------------------------------------------------------------------------
# Pretraining budget -- ground truth from Lior's notebook cell 4
# Budget parity: same number of epochs over the same 100k images at the same batch size.
# ---------------------------------------------------------------------------

BATCH_SIZE = 256
FEATURE_BATCH_SIZE = 512

EPOCHS = 100
WARMUP_EPOCHS = 10

MAX_LEARNING_RATE = 5e-4
MIN_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4


# ---------------------------------------------------------------------------
# Encoder trunk -- ground truth from Lior's notebook cell 12
# Reimplemented from scratch (build-from-0 forbids torchvision.models), but the shape of
# the network must match his exactly so the pretraining objective stays the only variable.
# ---------------------------------------------------------------------------

STEM_WIDTH = 48
STAGE_DEPTHS = (2, 3, 4, 3)
STAGE_WIDTHS = (64, 128, 256, 384)
ENCODER_DIM = 512
SE_REDUCTION = 16


# ---------------------------------------------------------------------------
# VAE head
# ---------------------------------------------------------------------------

# LATENT_DIM MUST equal ENCODER_DIM. This is the single highest-risk fairness item in the
# whole project: Lior's embedding is the 512-d pooled trunk feature with the projection
# head discarded, so our mu -- the thing we build the KNN graph on -- has to be 512-d too.
# A smaller latent would hand the contrastive arm a dimensionality advantage that has
# nothing to do with generative vs contrastive.
LATENT_DIM = ENCODER_DIM

# Beta schedule. beta is warmed up linearly so the KL term does not dominate before the
# decoder can reconstruct anything, which is the usual cause of immediate posterior
# collapse.
BETA_MAX = 0.01
BETA_WARMUP_EPOCHS = 10

# The grid stops at 0.1 on purpose. The KL term pulls every posterior toward the same
# unit Gaussian; by beta = 1 it has flattened exactly the class structure that label
# propagation depends on, and past that the posterior collapses entirely -- mu becomes
# constant, every pairwise cosine similarity becomes ~1, and the KNN graph degenerates
# into noise. Values above 0.1 would only produce degenerate runs, so they buy no
# information about the research question.
# beta = 0.0 is kept as the plain-autoencoder control arm: it isolates how much of the
# result comes from the *variational* part rather than from reconstruction alone.
BETA_GRID = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1)


# ---------------------------------------------------------------------------
# KNN graph and label propagation
# ---------------------------------------------------------------------------

# K is a real hyperparameter of the method, not a detail: too small and the graph
# fragments into components labels cannot cross, too large and edges start bridging
# classes. Selected on the held-out labeled subset, never on test.
K_GRID = (5, 10, 15, 20, 30, 50)

# The cap and the tolerance have to be chosen together, or the convergence test is
# decorative. Measured on a 7200-node heat-kernel graph of this shape, max|F_next - F|
# needs 107-118 iterations to fall below 1e-6 (and the count barely moves with K:
# 118 at K=5, 110 at K=15, 107 at K=50). A cap of 50 therefore stopped EVERY fold of
# EVERY K short of the tolerance, at a delta around 5e-4 -- so `converged` was False
# on every row of the sweep, the "did not converge" warning fired unconditionally,
# and by the pipeline's own account every accuracy it reported was "a snapshot of a
# still-moving iteration". 300 leaves real headroom on the full 105k graph, whose
# diameter is larger; the cost is ~200 extra sparse matmuls on a [105000, 10] matrix
# per fold, which is seconds.
LP_MAX_ITERATIONS = 3000
LP_TOLERANCE = 1e-6

# Restart weight for Zhou label spreading, F <- alpha P F + (1 - alpha) Y, which is
# the default method. It exists because hard-clamped Zhu-Ghahramani propagation --
# what this pipeline originally ran -- has a nearly FLAT fixed point on this graph.
# Measured over the ten folds at beta=0.001: at convergence the mean winning-class
# share is 0.155 against a uniform floor of 0.10, and dev accuracy is 0.230 at K=5
# and 0.208 at K=15, below the 0.370 that a plain cosine 20-NN vote against the 800
# seeds gets with no graph at all. 800 seeds among 105000 nodes is simply too few
# for an unrestarted walk to remember where it started.
#
# alpha controls locality: the fixed point is the discounted random-walk-with-restart
# score, so alpha -> 1 recovers the flat solution and small alpha barely leaves the
# seeds. 0.90 is the middle of the grid the sweep searches; it is a SELECTED value,
# not a guess, and --alpha overrides it.
#
# LP_MAX_ITERATIONS went 300 -> 3000 with this change. Spreading converges
# geometrically at rate alpha, needing about log(tolerance)/log(alpha) iterations:
# ~130 at alpha=0.9 but ~1375 at alpha=0.99, and the sweep searches up to 0.99.
# Under the old 300 the high-alpha cells would have been cut off mid-iteration,
# which is the exact failure the previous run already made once.
LP_ALPHA = 0.90
LP_ALPHA_GRID = (0.50, 0.80, 0.90, 0.95, 0.99)

# Rows of the similarity matrix computed per chunk. The full matrix is out of the
# question: 105000 x 105000 fp32 = 105000 * 105000 * 4 bytes ~= 44 GB, which fits in
# neither VRAM nor RAM. One chunk is 1024 x 105000 fp32 = 1024 * 105000 * 4 bytes
# ~= 430 MB, which is comfortable, and we only ever keep the top-K indices and values
# per row -- [N, K] instead of [N, N].
SIMILARITY_CHUNK_ROWS = 1024


# ---------------------------------------------------------------------------
# Evaluation protocol -- mirrors Lior's exactly
# ---------------------------------------------------------------------------

# STL-10 ships 10 official folds in fold_indices.txt, each 1000 indices into the 5000
# labeled train images. One classifier is fit per fold and evaluated on ALL 8000 test
# images; the headline number is the mean over the 10 folds.
NUM_FOLDS = 10
FOLD_SIZE = 1000

# Model selection happens inside each fold on a stratified 800/200 split. Test is touched
# exactly once, at the very end, by both arms.
FOLD_TRAIN_SIZE = 800
FOLD_DEVELOPMENT_SIZE = 200


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Per-RGB-channel mean/std are computed from the UNLABELED split only -- using the labeled
# train or test images would leak evaluation data into preprocessing. Accumulated in
# float64 over chunks (100000 * 96 * 96 * 3 pixels overflows float32 accumulation) and
# divided by 255 to land in [0, 1].
NORMALIZATION_SOURCE_SPLIT = "unlabeled"
NORMALIZATION_CHUNK_IMAGES = 5000
PIXEL_SCALE = 255.0


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Resolved relative to this file rather than the process cwd, so a script works whether it
# is launched from the repo root, from scripts/, or from a notebook in notebooks/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
RUNS_ROOT = PROJECT_ROOT / "runs"
RESULTS_ROOT = PROJECT_ROOT / "results"

# numpy.fromfile parses stl10_binary directly; torchvision.datasets is both banned and
# unnecessary, and disk is scarce on the lab server.
STL10_BINARY_ROOT = DATA_ROOT / "stl10_binary"


def run_directory(run_name):
    """Directory holding one pretraining run's checkpoints and history."""
    return RUNS_ROOT / run_name


def checkpoint_path(run_name, filename):
    """Path to a single checkpoint file inside a run directory."""
    return run_directory(run_name) / "checkpoints" / filename


def embeddings_path(run_name, filename):
    """Path to a cached frozen-embedding array.

    Embeddings are cached once (105000 x 512 fp16 ~= 107 MB) so the K sweep and the beta
    ablation read an array off disk instead of re-encoding 105k images every time.
    """
    return run_directory(run_name) / "embeddings" / filename


def results_directory(run_name):
    """Directory holding one run's tables and figures."""
    return RESULTS_ROOT / run_name


def results_path(run_name, filename):
    """Path to a single result file (CSV, JSON or figure) inside a results directory."""
    return results_directory(run_name) / filename


def ensure_directory(path):
    """Create a directory (and parents) if it is missing, and return it.

    Every writer calls this rather than each script repeating the mkdir dance.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
