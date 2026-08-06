"""Label propagation over the KNN graph, hand-rolled.

Build-from-0: `scipy.sparse.csgraph` and sklearn's `LabelSpreading` are both
banned, so the iteration is written out here. That is no hardship -- label
propagation is one line of linear algebra repeated until it stops moving:

    F <- P F,   then reset the labeled rows to their known one-hot values

`P` is the row-stochastic transition matrix built by the graph module from the
symmetrized heat-kernel KNN weights: `P[i, j]` is the share of node i's total
edge weight that goes to neighbour j, so every row sums to 1 and row i's
non-zeros are exactly i's neighbours. Under that convention `(P F)[i]` is the
weighted average of i's neighbours' current label distributions, which is the
Zhu-Ghahramani-Lafferty formulation we cite in the report.

`P` arrives as a torch sparse COO tensor because the dense form is impossible:
105000 x 105000 fp32 is ~44 GB. Sparse it is N*K non-zeros -- at K=50 that is
5.2M values, ~60 MB -- and `torch.sparse.mm` does the whole propagation step in
one call. The dense side, F, is only [105000, 10].

Everything follows the device and dtype of the tensors it is handed, so the same
code runs on CPU for the tests and on the 5090 for the real sweep.
"""

import numpy
import torch


def _as_long_tensor(values, device=None):
    """Accept a torch tensor, numpy array or list and return int64 torch.

    The pipeline mixes both conventions: the graph and the embeddings are torch
    tensors, while the STL-10 fold indices come out of `data.load_splits` as
    numpy arrays. Funnelling every index argument through here means no caller
    has to convert first.
    """
    if isinstance(values, torch.Tensor):
        tensor = values.to(torch.int64)
    else:
        tensor = torch.as_tensor(numpy.asarray(values)).to(torch.int64)
    if device is not None:
        tensor = tensor.to(device)
    return tensor.reshape(-1)


def one_hot_matrix(labels, num_classes, device, dtype=torch.float32):
    """[M] integer labels -> [M, num_classes] one-hot rows.

    Written out rather than pulled from `torch.nn.functional.one_hot` only so the
    dtype is float from the start; F.one_hot returns int64 and would then need a
    cast anyway. Used both to seed F0 and to build the clamp target for
    `propagate`, so the two are guaranteed to agree by construction.
    """
    label_tensor = _as_long_tensor(labels, device=device)

    if label_tensor.numel() > 0:
        smallest = int(label_tensor.min().item())
        largest = int(label_tensor.max().item())
        if smallest < 0 or largest >= num_classes:
            raise ValueError(
                "labels must all lie in [0, " + str(num_classes) + "), got range ["
                + str(smallest) + ", " + str(largest) + "] -- the -1 unlabelled "
                "sentinel must not be seeded into the label matrix"
            )

    one_hot = torch.zeros(label_tensor.numel(), num_classes, dtype=dtype, device=device)
    if label_tensor.numel() > 0:
        one_hot.scatter_(1, label_tensor.reshape(-1, 1), 1.0)
    return one_hot


def initial_label_matrix(num_nodes, num_classes, labeled_indices, labeled_labels, device,
                         dtype=torch.float32):
    """Build F0, the [num_nodes, num_classes] starting state of the propagation.

    Labeled rows hold the one-hot vector of their known class; every other row is
    all zeros -- NOT uniform. Seeding the unlabeled rows with 1/C would inject a
    constant into the fixed point and flatten exactly the class contrast we are
    trying to measure, and it would also destroy our ability to detect nodes that
    never received any mass (see `pseudo_labels_from`): with a uniform start
    every row has mass forever and an unreachable node becomes invisible.

    `labeled_indices` are positions into the graph's node array (unlabeled split
    first, labeled train split after it -- see config.NUM_GRAPH_NODES), not
    positions into the 5000-image labeled split. The caller does that offset.
    """
    number_of_nodes = int(num_nodes)
    index_tensor = _as_long_tensor(labeled_indices, device=device)
    label_tensor = _as_long_tensor(labeled_labels, device=device)

    if index_tensor.numel() != label_tensor.numel():
        raise ValueError(
            "labeled_indices and labeled_labels must have the same length, got "
            + str(index_tensor.numel()) + " and " + str(label_tensor.numel())
        )
    if index_tensor.numel() > 0:
        smallest = int(index_tensor.min().item())
        largest = int(index_tensor.max().item())
        if smallest < 0 or largest >= number_of_nodes:
            raise ValueError(
                "labeled_indices must lie in [0, " + str(number_of_nodes) + "), got range ["
                + str(smallest) + ", " + str(largest) + "]"
            )

    initial_F = torch.zeros(number_of_nodes, num_classes, dtype=dtype, device=device)
    if index_tensor.numel() > 0:
        initial_F[index_tensor] = one_hot_matrix(label_tensor, num_classes, device, dtype=dtype)
    return initial_F


def propagate(propagation_matrix, initial_F, labeled_indices, labeled_onehot,
              max_iterations, tolerance):
    """Iterate F <- P F with the labeled rows re-clamped every step.

    THE RE-CLAMP IS THE ALGORITHM, not a tidying step. `P` is row-stochastic, so
    iterating `F <- P F` on its own is a random walk: the labeled rows are
    themselves averages of their neighbours, so the seed labels immediately start
    diffusing OUT of the nodes we actually know, and the iteration converges to
    the walk's stationary distribution -- which depends only on the graph's degree
    structure and not at all on the labels. Every row would end up at (very
    nearly) the same vector and the argmax would be meaningless. Resetting the
    labeled rows to their one-hot values after every single multiply turns those
    nodes into permanent sources instead of participants, and it is that boundary
    condition that makes the fixed point the harmonic function we want.

    Returns (F, iterations_run, final_delta). The last two are not decoration:
    a run that exhausts `max_iterations` without `final_delta` dropping below
    `tolerance` has NOT converged, and the pseudo-labels it produced are a
    snapshot of a still-moving iteration. That is a finding to report, not
    something to silently accept, so the caller gets the numbers to log.

    `labeled_onehot` is the clamp target, [len(labeled_indices), num_classes]. The
    natural way to obtain it is `initial_F[labeled_indices]`, which is what
    `initial_label_matrix` already wrote there.

    `propagation_matrix` is expected to be a torch sparse COO tensor; a dense
    matrix is accepted too so the unit tests can use a hand-written 5-node graph
    without constructing sparse tensors.
    """
    if propagation_matrix.dim() != 2:
        raise ValueError(
            "propagation_matrix must be [N, N], got shape " + str(tuple(propagation_matrix.shape))
        )
    if propagation_matrix.shape[0] != propagation_matrix.shape[1]:
        raise ValueError(
            "propagation_matrix must be square, got shape " + str(tuple(propagation_matrix.shape))
        )
    if initial_F.dim() != 2:
        raise ValueError("initial_F must be [N, C], got shape " + str(tuple(initial_F.shape)))
    if propagation_matrix.shape[0] != initial_F.shape[0]:
        raise ValueError(
            "propagation_matrix has " + str(propagation_matrix.shape[0]) + " nodes but "
            "initial_F has " + str(initial_F.shape[0]) + " rows"
        )

    device = propagation_matrix.device
    dtype = propagation_matrix.dtype

    # F must live where P lives and share its dtype -- torch.sparse.mm will not
    # promote for us, and a silent CPU/CUDA mismatch here is a very confusing error.
    #
    # The clone matters. `.to()` returns the SAME tensor when the device and dtype
    # already match, which is the normal case here, so without it `current_F` IS
    # the caller's `initial_F` and the clamp below writes straight into the
    # caller's argument. Two consequences, one of them fatal:
    #   * the caller's F0 is silently mutated, so it cannot be reused or compared
    #     against the result;
    #   * if `labeled_onehot` is a VIEW of `initial_F` -- and this docstring
    #     invites exactly that, "the natural way is initial_F[labeled_indices]",
    #     which is a view whenever the indices are slice-shaped -- then the clamp
    #     is an index_put_ whose source and destination overlap. Under
    #     seeding.enable_deterministic(), which every script in scripts/ turns on,
    #     torch refuses that outright with "some elements of the input tensor and
    #     the written-to tensor refer to a single memory location".
    # Cloning breaks the aliasing chain for both, and costs one [N, C] buffer --
    # 105000 x 10 fp32 is 4 MB, nothing next to the graph.
    current_F = initial_F.to(device=device, dtype=dtype)
    if current_F is initial_F:
        current_F = current_F.clone()

    index_tensor = _as_long_tensor(labeled_indices, device=device)
    clamp_target = labeled_onehot.to(device=device, dtype=dtype)
    if clamp_target.shape[0] != index_tensor.numel():
        raise ValueError(
            "labeled_onehot must have one row per labeled index, got "
            + str(clamp_target.shape[0]) + " rows for " + str(index_tensor.numel()) + " indices"
        )

    # Clamp before the first multiply as well, so a caller who passed an F0 that
    # does not already carry the seeds still gets a correct first step.
    if index_tensor.numel() > 0:
        current_F[index_tensor] = clamp_target

    iterations_run = 0
    # inf, not 0: with max_iterations = 0 nothing was measured, and reporting a
    # delta of 0 would read as "converged immediately".
    final_delta = float("inf")

    for _ in range(int(max_iterations)):
        if propagation_matrix.is_sparse:
            next_F = torch.sparse.mm(propagation_matrix, current_F)
        else:
            next_F = torch.matmul(propagation_matrix, current_F)

        # Re-clamp, then measure. Measuring before the clamp would report the
        # (large) movement of the seed rows that we are about to undo, and the
        # convergence test would never fire.
        if index_tensor.numel() > 0:
            next_F[index_tensor] = clamp_target

        final_delta = float((next_F - current_F).abs().max().item())
        current_F = next_F
        iterations_run += 1

        if final_delta < tolerance:
            break

    return current_F, iterations_run, final_delta


def spread(propagation_matrix, initial_F, labeled_indices, labeled_onehot,
           alpha, max_iterations, tolerance):
    """Iterate F <- alpha P F + (1 - alpha) Y. Zhou label spreading, soft clamp.

    WHY THIS EXISTS. `propagate` above implements Zhu-Ghahramani hard-clamped
    propagation, and on this graph its fixed point is very nearly useless. With
    800 seeds among 105000 nodes the walk mixes long before it is absorbed, so
    the absorption probabilities stop depending on where the walk started: at
    convergence the mean winning-class share is 0.155 against a uniform floor of
    0.10, and mean dev accuracy over the ten folds falls to 0.23 at K=15. It is
    not a bug -- the iteration reaches its fixed point correctly, to a max delta
    of 6e-8 -- the fixed point is simply flat. Measured on the way there, dev
    accuracy peaks at 0.401 around iteration 8 and then decays monotonically into
    that flat solution.

    The `(1 - alpha) Y` restart term is what fixes it. Y is the seed matrix --
    one-hot on labeled rows, zero elsewhere, i.e. exactly the `initial_F` that
    `initial_label_matrix` builds -- and re-injecting it at every step means a
    walk that wanders too far keeps being pulled back to where it started. The
    fixed point is

        F* = (1 - alpha) (I - alpha P)^-1 Y

    the discounted random-walk-with-restart score, which stays local and stays
    informative. Setting alpha = 1 recovers the degenerate flat solution, so
    alpha is the knob that trades locality against reach.

    NO HARD CLAMP HERE, deliberately. Zhou's formulation soft-clamps: the seed
    rows are updated like every other row and held near their one-hot value by
    the restart term rather than pinned to it. `labeled_onehot` is therefore used
    to build Y, not to overwrite rows mid-iteration. That also means a seed row
    can end up slightly contested by its neighbours, which is the intended
    behaviour -- it lets the method tolerate a mislabeled seed instead of
    propagating it with full confidence forever.

    CONVERGENCE. `alpha P` has spectral radius alpha < 1, so the iteration is a
    contraction and converges geometrically at rate alpha -- unconditionally, no
    reliance on the graph's structure. The cost is that the iteration count
    needed scales as log(tolerance) / log(alpha): about 130 steps at alpha=0.9
    but about 1375 at alpha=0.99. Pass `max_iterations` accordingly; the caller
    gets `iterations_run` and `final_delta` back and should check them, exactly
    as with `propagate`.

    A NOTE ON SCALE. Row masses here are much smaller than 1 -- a row's total is
    the expected discounted number of visits to seed nodes, which for a distant
    node is a small number. That is harmless: `pseudo_labels_from` takes an
    argmax and normalizes confidence by the row sum, and both are invariant to
    the overall scale. It does mean an absolute `tolerance` is a tighter test
    here than it looks.

    Returns (F, iterations_run, final_delta), same contract as `propagate`.
    """
    if propagation_matrix.dim() != 2:
        raise ValueError(
            "propagation_matrix must be [N, N], got shape " + str(tuple(propagation_matrix.shape))
        )
    if propagation_matrix.shape[0] != propagation_matrix.shape[1]:
        raise ValueError(
            "propagation_matrix must be square, got shape " + str(tuple(propagation_matrix.shape))
        )
    if initial_F.dim() != 2:
        raise ValueError("initial_F must be [N, C], got shape " + str(tuple(initial_F.shape)))
    if propagation_matrix.shape[0] != initial_F.shape[0]:
        raise ValueError(
            "propagation_matrix has " + str(propagation_matrix.shape[0]) + " nodes but "
            "initial_F has " + str(initial_F.shape[0]) + " rows"
        )

    restart_weight = float(alpha)
    if not 0.0 < restart_weight < 1.0:
        raise ValueError(
            "alpha must lie strictly in (0, 1) -- alpha=1 is the hard-clamped "
            "iteration `propagate` already implements and its fixed point is flat, "
            "alpha=0 never propagates at all. Got " + str(restart_weight)
        )

    device = propagation_matrix.device
    dtype = propagation_matrix.dtype

    index_tensor = _as_long_tensor(labeled_indices, device=device)
    clamp_target = labeled_onehot.to(device=device, dtype=dtype)
    if clamp_target.shape[0] != index_tensor.numel():
        raise ValueError(
            "labeled_onehot must have one row per labeled index, got "
            + str(clamp_target.shape[0]) + " rows for " + str(index_tensor.numel()) + " indices"
        )

    # Y, the restart distribution. Built from the seeds rather than trusting the
    # caller's initial_F to be exactly the seed matrix, so a caller who warm-starts
    # the iteration from some other F0 still restarts towards the right thing.
    restart_F = torch.zeros_like(initial_F, device=device, dtype=dtype)
    if index_tensor.numel() > 0:
        restart_F[index_tensor] = clamp_target
    restart_term = (1.0 - restart_weight) * restart_F

    # Same aliasing hazard as in `propagate`: .to() is a no-op when device and
    # dtype already match, and we must not write into the caller's array.
    current_F = initial_F.to(device=device, dtype=dtype)
    if current_F is initial_F:
        current_F = current_F.clone()

    iterations_run = 0
    final_delta = float("inf")

    for _ in range(int(max_iterations)):
        if propagation_matrix.is_sparse:
            next_F = torch.sparse.mm(propagation_matrix, current_F)
        else:
            next_F = torch.matmul(propagation_matrix, current_F)

        next_F = restart_weight * next_F + restart_term

        final_delta = float((next_F - current_F).abs().max().item())
        current_F = next_F
        iterations_run += 1

        if final_delta < tolerance:
            break

    return current_F, iterations_run, final_delta


def pseudo_labels_from(F, minimum_mass=0.0, epsilon=1e-12):
    """Turn the converged F into (pseudo_labels, confidence).

    pseudo_labels is [N] int64, confidence is [N] float. Confidence is the
    winning class's share of the row, max(row) / (sum(row) + eps): a row where one
    class holds almost all the mass scores ~1, a row split evenly between two
    classes scores ~0.5. It is not a calibrated probability and we do not claim it
    is -- it is a ranking signal for "how contested was this node", which is what
    the downstream CNN needs to decide which pseudo-labels to trust.

    Nodes that received NO mass at all get label -1 instead of a class. This
    matters: F0 seeds only the labeled rows, so a node sitting in a connected
    component with no labeled seed -- an isolated node, or one of the small
    islands a low K shatters off the graph -- keeps an exactly-zero row forever.
    argmax on an all-zero row returns index 0 with no complaint, so without this
    check every unreachable node would silently be pseudo-labeled "airplane",
    contaminating the training set for the downstream classifier with a bias that
    is completely invisible in the accuracy number. The -1 sentinel forces the
    caller to decide what to do with them, and `metrics.confusion_matrix` refuses
    to score -1 rather than folding it into class 0.

    `minimum_mass` defaults to 0.0, i.e. only exactly-zero rows are flagged. Raise
    it to also discard rows whose mass is numerically negligible (long chains of
    heat-kernel weights can underflow towards zero in fp32).
    """
    if F.dim() != 2:
        raise ValueError("F must be [N, C], got shape " + str(tuple(F.shape)))

    row_mass = F.sum(dim=1)
    winning_mass, winning_class = F.max(dim=1)

    confidence = winning_mass / (row_mass + epsilon)

    unreachable = row_mass <= minimum_mass
    pseudo_labels = torch.where(
        unreachable,
        torch.full_like(winning_class, -1),
        winning_class
    )
    confidence = torch.where(unreachable, torch.zeros_like(confidence), confidence)

    return pseudo_labels, confidence


def _quantile(sorted_values, fraction):
    """Linear-interpolation quantile of an ALREADY SORTED 1-D float tensor.

    Hand-rolled for two reasons: numpy.quantile would mean moving a 105k CUDA
    tensor back to host on every summary call, and `torch.quantile` carries an
    input-size limit that we would rather not have to think about when N grows.
    The interpolation rule is numpy's default ("linear"), so the numbers in the
    report match what a reader gets if they check us with numpy.
    """
    count = sorted_values.numel()
    if count == 0:
        return float("nan")
    if count == 1:
        return float(sorted_values[0].item())

    position = fraction * (count - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, count - 1)
    weight = position - lower_index

    lower_value = float(sorted_values[lower_index].item())
    upper_value = float(sorted_values[upper_index].item())
    return lower_value + weight * (upper_value - lower_value)


def propagation_summary(F, pseudo_labels):
    """Diagnostics for one propagation run, as plain Python types for logging.

    Reports three things, each of which answers a question the K sweep has to
    answer with a number rather than a hunch:

      1. how many nodes were unreachable (label -1), i.e. the hard ceiling the
         graph's component structure puts on label propagation at this K;
      2. the distribution of the confidence scores, since a run can reach every
         node and still be worthless if every row ends up evenly contested;
      3. the per-class pseudo-label distribution.

    On (3): STL-10's unlabeled split is DELIBERATELY broader than the 10 labeled
    classes -- it contains animals and vehicles from classes that are not in the
    label set at all. Those out-of-distribution nodes are in the graph, they have
    neighbours, and label mass diffuses through them and gets assigned to whichever
    of the 10 classes happens to be nearest in the latent space. So a skewed
    per-class distribution here is EXPECTED, not a bug in the propagation. Per
    CLAUDE.md we quantify it and report it as a finding rather than trying to
    filter it out: any OOD rejection heuristic we invented would be a second
    uncontrolled variable sitting between the two arms, and it would break the
    apples-to-apples comparison with Lior's SimCLR arm, which does no such
    filtering either.

    Returns a dictionary; every value is a Python int/float/list so the whole
    thing can be dropped straight into a JSON results file.
    """
    if F.dim() != 2:
        raise ValueError("F must be [N, C], got shape " + str(tuple(F.shape)))

    label_tensor = pseudo_labels.reshape(-1)
    number_of_nodes = F.shape[0]
    number_of_classes = F.shape[1]

    if label_tensor.numel() != number_of_nodes:
        raise ValueError(
            "pseudo_labels must have one entry per node: got " + str(label_tensor.numel())
            + " for " + str(number_of_nodes) + " nodes"
        )

    row_mass = F.sum(dim=1)
    unreachable = label_tensor < 0
    unreachable_count = int(unreachable.sum().item())
    reachable_count = number_of_nodes - unreachable_count

    # Confidence is recomputed here rather than taken as an argument so the
    # summary can be produced from (F, pseudo_labels) alone -- the two things a
    # results file actually stores.
    winning_mass, _ = F.max(dim=1)
    confidence = winning_mass / (row_mass + 1e-12)
    reachable_confidence = confidence[~unreachable].to(torch.float64)

    if reachable_confidence.numel() > 0:
        sorted_confidence, _ = torch.sort(reachable_confidence)
        confidence_quantiles = {
            "min": _quantile(sorted_confidence, 0.0),
            "p10": _quantile(sorted_confidence, 0.10),
            "p25": _quantile(sorted_confidence, 0.25),
            "median": _quantile(sorted_confidence, 0.50),
            "p75": _quantile(sorted_confidence, 0.75),
            "p90": _quantile(sorted_confidence, 0.90),
            "max": _quantile(sorted_confidence, 1.0),
        }
        confidence_mean = float(reachable_confidence.mean().item())
    else:
        confidence_quantiles = {
            "min": float("nan"), "p10": float("nan"), "p25": float("nan"),
            "median": float("nan"), "p75": float("nan"), "p90": float("nan"),
            "max": float("nan"),
        }
        confidence_mean = float("nan")

    # Same bincount trick the metrics module uses. minlength keeps the list length
    # at C even when a class wins nothing, so results files stay comparable across
    # K values and across the two arms.
    if reachable_count > 0:
        class_counts = torch.bincount(
            label_tensor[~unreachable].to(torch.int64),
            minlength=number_of_classes
        ).to(torch.int64)
    else:
        class_counts = torch.zeros(number_of_classes, dtype=torch.int64, device=F.device)

    counts_list = [int(value) for value in class_counts.tolist()]
    if reachable_count > 0:
        fractions_list = [value / reachable_count for value in counts_list]
    else:
        fractions_list = [0.0 for _ in counts_list]

    return {
        "num_nodes": int(number_of_nodes),
        "num_classes": int(number_of_classes),
        "unreachable_count": unreachable_count,
        "unreachable_fraction": unreachable_count / number_of_nodes if number_of_nodes > 0 else 0.0,
        "labeled_count": reachable_count,
        "confidence_mean": confidence_mean,
        "confidence_quantiles": confidence_quantiles,
        "class_counts": counts_list,
        "class_fractions": fractions_list,
        # The gap between the most and least popular class. On STL-10 this is
        # driven partly by the out-of-distribution unlabeled images described
        # above, so it is reported rather than corrected.
        "largest_class_fraction": max(fractions_list) if fractions_list else 0.0,
        "smallest_class_fraction": min(fractions_list) if fractions_list else 0.0,
    }
