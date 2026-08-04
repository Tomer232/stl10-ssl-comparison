"""Hand-rolled evaluation metrics for the VAE + label-propagation arm.

The lecturer's "build from 0" constraint rules out sklearn.metrics, so every
classification number we report is derived here from a single object: the
CONFUSION MATRIX, built with one `torch.bincount` call. That is the whole point
of this module -- accuracy, precision, recall and F1 are not four independent
algorithms, they are four different ways of reading the same [C, C] table of
counts. Once the table exists there is nothing left to import.

Where sklearn has a documented convention that changes the number (its
`zero_division=0` rule, its clipping inside `log_loss`), we deliberately copy
that convention so our numbers stay directly comparable to Lior's SimCLR arm,
which is allowed to keep sklearn.

Everything stays on whatever device the inputs live on -- CPU for the tests,
CUDA for the real runs -- so nothing here hardcodes `.cuda()`.
"""

import numpy as np
import torch


def _as_long_tensor(values, device=None):
    """Accept a torch tensor or a numpy array / list and return int64 torch.

    The pipeline mixes both: embeddings and the KNN graph come back as torch
    tensors, while the STL-10 fold indices are parsed with numpy.fromfile. Every
    public function funnels through here so callers never have to care.
    """
    if isinstance(values, torch.Tensor):
        tensor = values.to(torch.int64)
    else:
        tensor = torch.as_tensor(np.asarray(values)).to(torch.int64)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _as_bool_tensor(values, device=None):
    """Same idea as _as_long_tensor, for masks."""
    if isinstance(values, torch.Tensor):
        tensor = values.to(torch.bool)
    else:
        tensor = torch.as_tensor(np.asarray(values)).to(torch.bool)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _as_double_tensor(values, device=None):
    """Float64 for the metric arithmetic.

    C is 10, so the cost of double precision is nil, and it removes any doubt
    about float32 rounding when we compare our accuracy to Lior's to 4 decimals.
    """
    if isinstance(values, torch.Tensor):
        tensor = values.to(torch.float64)
    else:
        tensor = torch.as_tensor(np.asarray(values)).to(torch.float64)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def confusion_matrix(true_labels, predicted_labels, num_classes):
    """Build the [num_classes, num_classes] confusion matrix. Rows = true, columns = predicted.

    THIS IS THE ONE PRIMITIVE. Every other classification metric in this module
    reads off the matrix returned here; none of them touches the raw label
    vectors again.

    The trick that makes it a one-liner without sklearn: a pair (true, predicted)
    is uniquely encoded by the single integer `true * num_classes + predicted`,
    which lands in [0, num_classes**2). Counting how often each of those integers
    occurs is exactly what `torch.bincount` does, and reshaping that flat count
    vector to [C, C] recovers the matrix. `minlength` is required so that classes
    which never appear still get their (all-zero) row and column -- otherwise the
    matrix would silently change shape between folds.
    """
    true_tensor = _as_long_tensor(true_labels)
    predicted_tensor = _as_long_tensor(predicted_labels, device=true_tensor.device)

    true_tensor = true_tensor.reshape(-1)
    predicted_tensor = predicted_tensor.reshape(-1)

    if true_tensor.numel() != predicted_tensor.numel():
        raise ValueError(
            "true_labels and predicted_labels must have the same length, got "
            + str(true_tensor.numel()) + " and " + str(predicted_tensor.numel())
        )

    # Catches the classic bug of feeding in the -1 sentinel we use for "still
    # unlabelled" nodes: bincount would either wrap or throw something cryptic.
    if true_tensor.numel() > 0:
        smallest = int(min(true_tensor.min().item(), predicted_tensor.min().item()))
        largest = int(max(true_tensor.max().item(), predicted_tensor.max().item()))
        if smallest < 0 or largest >= num_classes:
            raise ValueError(
                "labels must all lie in [0, " + str(num_classes) + "), got range ["
                + str(smallest) + ", " + str(largest) + "] -- unlabelled sentinels "
                "(-1) must be filtered out before scoring"
            )

    flat_counts = torch.bincount(
        true_tensor * num_classes + predicted_tensor,
        minlength=num_classes * num_classes
    )
    return flat_counts.reshape(num_classes, num_classes)


def accuracy(confusion):
    """Overall accuracy = correct predictions / all predictions.

    On the confusion matrix the correct predictions are exactly the diagonal
    (true class == predicted class), so this is trace / total.
    """
    counts = _as_double_tensor(confusion)
    total = counts.sum()
    if total.item() == 0.0:
        return 0.0
    return float((torch.diagonal(counts).sum() / total).item())


def per_class_precision_recall_f1(confusion):
    """Return (precision, recall, f1), each a float64 tensor of length C.

    Read straight off the matrix:
      true positives for class c  = confusion[c, c]
      predicted as c              = column sum  -> precision denominator
      actually c                  = row sum     -> recall denominator

    Zero denominators are reported as 0.0 rather than NaN. That mirrors
    sklearn's `zero_division=0`, which is what Lior passes, so a class that never
    gets predicted contributes 0 to our macro average exactly as it does to his.
    Without matching this convention the two arms' macro-F1 would not be
    comparable at all.
    """
    counts = _as_double_tensor(confusion)

    true_positives = torch.diagonal(counts)
    predicted_totals = counts.sum(dim=0)
    actual_totals = counts.sum(dim=1)

    precision = torch.where(
        predicted_totals > 0,
        true_positives / torch.clamp(predicted_totals, min=1.0),
        torch.zeros_like(true_positives)
    )
    recall = torch.where(
        actual_totals > 0,
        true_positives / torch.clamp(actual_totals, min=1.0),
        torch.zeros_like(true_positives)
    )

    f1_denominator = precision + recall
    f1 = torch.where(
        f1_denominator > 0,
        2.0 * precision * recall / torch.clamp(f1_denominator, min=1e-300),
        torch.zeros_like(precision)
    )

    return precision, recall, f1


def macro_f1(confusion):
    """Unweighted mean of the per-class F1 scores (sklearn's average="macro")."""
    _, _, f1 = per_class_precision_recall_f1(confusion)
    return float(f1.mean().item())


def macro_precision(confusion):
    """Unweighted mean of the per-class precisions."""
    precision, _, _ = per_class_precision_recall_f1(confusion)
    return float(precision.mean().item())


def cross_entropy_loss(probabilities, labels):
    """Mean negative log-likelihood of the true class -- our sklearn `log_loss`.

    Lior reports log_loss for model selection, so we need the same quantity to
    put the two arms in one table. sklearn does two things beyond the textbook
    formula and we copy both, otherwise the numbers drift:
      1. it clips probabilities away from 0 and 1, because log(0) is -inf and a
         single over-confident wrong prediction would otherwise make the whole
         mean infinite;
      2. it renormalizes each row after clipping so the rows still sum to 1.

    `probabilities` is [N, C] (softmax output of the probe / classifier),
    `labels` is [N] of integer class ids.
    """
    probability_tensor = _as_double_tensor(probabilities)
    label_tensor = _as_long_tensor(labels, device=probability_tensor.device).reshape(-1)

    if probability_tensor.dim() != 2:
        raise ValueError("probabilities must be [N, C], got shape " + str(tuple(probability_tensor.shape)))
    if probability_tensor.shape[0] != label_tensor.numel():
        raise ValueError(
            "probabilities and labels disagree on N: "
            + str(probability_tensor.shape[0]) + " vs " + str(label_tensor.numel())
        )
    if label_tensor.numel() == 0:
        return 0.0

    epsilon = float(np.finfo(np.float64).eps)
    clipped = torch.clamp(probability_tensor, min=epsilon, max=1.0 - epsilon)
    clipped = clipped / clipped.sum(dim=1, keepdim=True)

    true_class_probability = clipped.gather(1, label_tensor.reshape(-1, 1)).reshape(-1)
    return float((-torch.log(true_class_probability)).mean().item())


def edge_purity(neighbor_indices, labels, valid_mask=None):
    """Fraction of KNN edges that join two nodes of the SAME class.

    This is a graph-quality diagnostic that is completely independent of label
    propagation: it asks "how often does the embedding put a node next to
    something of its own class", which is the property LP actually consumes. A
    K sweep judged on LP accuracy alone cannot separate a bad graph from a bad
    propagation schedule, so CLAUDE.md requires this number next to accuracy for
    every K.

    `neighbor_indices` is the [N, K] index array produced by the chunked topk.
    Only edges where BOTH endpoints have a known label are counted -- on STL-10
    the vast majority of the 105k nodes are unlabelled, and there is no honest
    way to score an edge whose endpoint class we do not know.

    We count the K directed arcs each node emits rather than the edges of the
    symmetrized graph. A mutual pair therefore contributes twice, which is fine
    for a ratio and keeps the number comparable across K without depending on
    how the symmetrization was done.
    """
    neighbors = _as_long_tensor(neighbor_indices)
    if neighbors.dim() != 2:
        raise ValueError("neighbor_indices must be [N, K], got shape " + str(tuple(neighbors.shape)))

    device = neighbors.device
    label_tensor = _as_long_tensor(labels, device=device).reshape(-1)

    number_of_nodes = neighbors.shape[0]
    if label_tensor.numel() != number_of_nodes:
        raise ValueError(
            "labels must have one entry per node: got " + str(label_tensor.numel())
            + " labels for " + str(number_of_nodes) + " nodes"
        )

    if valid_mask is None:
        known = torch.ones(number_of_nodes, dtype=torch.bool, device=device)
    else:
        known = _as_bool_tensor(valid_mask, device=device).reshape(-1)
        if known.numel() != number_of_nodes:
            raise ValueError(
                "valid_mask must have one entry per node: got " + str(known.numel())
                + " for " + str(number_of_nodes) + " nodes"
            )

    source_indices = torch.arange(number_of_nodes, device=device).reshape(-1, 1)
    source_indices = source_indices.expand_as(neighbors)

    # Self-loops would be trivially pure and inflate the score. The graph builder
    # masks the diagonal before topk, so this is belt-and-braces.
    countable = known[source_indices] & known[neighbors] & (source_indices != neighbors)

    total_edges = int(countable.sum().item())
    if total_edges == 0:
        return 0.0

    same_label = (label_tensor[source_indices] == label_tensor[neighbors]) & countable
    return float(int(same_label.sum().item()) / total_edges)


def component_label_coverage(component_ids, labeled_mask):
    """Fraction of nodes sitting in a connected component that holds >= 1 labeled node.

    Label propagation can only move mass along edges. A node in a component with
    no labeled seed is unreachable: no number of iterations will ever give it a
    label, and it will end up with the uniform/zero row that our argmax then
    resolves arbitrarily. So this is a hard ceiling on what LP can do for a given
    K, and CLAUDE.md asks for it in the K sweep -- a small K may look fine on
    edge purity while shattering the graph into islands.

    `component_ids` is the [N] array from unionfind.connected_components (already
    relabelled to a contiguous 0..M-1 range, which is what makes the bincount
    below valid). `labeled_mask` is [N] and True where a seed label exists.
    """
    components = _as_long_tensor(component_ids).reshape(-1)
    device = components.device
    labeled = _as_bool_tensor(labeled_mask, device=device).reshape(-1)

    number_of_nodes = components.numel()
    if labeled.numel() != number_of_nodes:
        raise ValueError(
            "component_ids and labeled_mask must agree on N: "
            + str(number_of_nodes) + " vs " + str(labeled.numel())
        )
    if number_of_nodes == 0:
        return 0.0

    number_of_components = int(components.max().item()) + 1

    # Same bincount idea as the confusion matrix: weight each node by whether it
    # is labeled, bucket by component id, and a component is "covered" iff its
    # bucket is non-zero.
    labeled_per_component = torch.bincount(
        components,
        weights=labeled.to(torch.float64),
        minlength=number_of_components
    )
    component_is_covered = labeled_per_component > 0

    return float(int(component_is_covered[components].sum().item()) / number_of_nodes)
