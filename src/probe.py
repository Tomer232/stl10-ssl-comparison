"""Hand-rolled multinomial logistic-regression probe -- the apples-to-apples row.

Lior's downstream evaluation is exactly

    Pipeline([("scaler", StandardScaler()),
              ("classifier", LogisticRegression(C=..., solver="lbfgs", max_iter=5000))])

fitted per fold, with an inner 5-split StratifiedKFold search over C in
(0.1, 1.0, 10.0, 100.0). Build-from-0 forbids us that pipeline, so this module
rebuilds it out of torch: a standardizer, a regularized softmax regression fitted
with L-BFGS, and a stratified k-fold splitter.

The point is NOT to have "a linear classifier". The point is to optimize the
*same objective with the same regularization convention*, because the headline
claim of the project is a comparison between two embedding spaces. If our probe
regularized differently, a gap between the arms could just be a gap between two
different estimators, and the research question would be unanswerable. So every
place below where a convention changes a number -- the ddof=0 variance, the
zero-variance guard, the 0.5*w'w penalty scaled against a C-weighted data term,
the unpenalized intercept, the L-BFGS gradient tolerance -- copies the reference
implementation deliberately, and says so.

Everything runs on whatever device the input features live on, in float64. N here
is at most 1000 rows of 512 features, so double precision costs nothing and takes
any doubt about float32 rounding out of a comparison we quote to four decimals.
"""

import numpy as np
import torch
import torch.nn.functional as functional

try:
    # How the notebooks and scripts import the project: src/ itself on sys.path.
    import metrics
except ImportError:
    # How a test runner at the repo root imports it. Both spellings have to work
    # or the module is only usable from one of the two places we run it from.
    from src import metrics


# The probe's own hyperparameters, copied from cell 4 of Lior's notebook so the
# inner model selection is the same search over the same grid in both arms.
LINEAR_C_GRID = (0.1, 1.0, 10.0, 100.0)
CROSS_VALIDATION_SPLITS = 5
LINEAR_MAX_ITERATIONS = 5000

# LogisticRegression's default tol. It is handed to the lbfgs solver as `gtol`,
# i.e. the stopping threshold on the largest absolute gradient component.
LINEAR_TOLERANCE = 1e-4

NUMBER_OF_CLASSES = 10


def _as_feature_matrix(values, device=None):
    """Accept numpy or torch, return a float64 [N, D] tensor.

    Features arrive as numpy from the cached embedding array and as torch from a
    live encoder, so every entry point funnels through here instead of each
    function re-deriving the conversion.
    """
    if isinstance(values, torch.Tensor):
        tensor = values.to(torch.float64)
    else:
        tensor = torch.as_tensor(np.asarray(values)).to(torch.float64)
    if device is not None:
        tensor = tensor.to(device)
    if tensor.dim() != 2:
        raise ValueError("features must be [N, D], got shape " + str(tuple(tensor.shape)))
    return tensor


def _as_label_vector(values, device=None):
    """Accept numpy or torch, return an int64 [N] tensor."""
    if isinstance(values, torch.Tensor):
        tensor = values.to(torch.int64)
    else:
        tensor = torch.as_tensor(np.asarray(values)).to(torch.int64)
    if device is not None:
        tensor = tensor.to(device)
    return tensor.reshape(-1)


class Standardizer:
    """Per-feature zero-mean unit-variance rescaling -- our StandardScaler.

    FIT ON TRAIN FEATURES ONLY. The mean and standard deviation are statistics of
    the data, so fitting them on the validation or test features leaks information
    about the evaluation set into the model: the probe would see features centred
    using knowledge it is not allowed to have, and the reported accuracy would be
    optimistically biased. Every call site here fits on the training half of a
    split and then *transforms* the held-out half with those stored constants.
    That is also precisely what sklearn's Pipeline enforces mechanically, and
    since we do not have a Pipeline, the discipline has to be manual.

    Two conventions are copied rather than chosen:

    * the variance is the population variance (ddof = 0, i.e. divide by N, not
      N - 1), which is what StandardScaler uses;
    * a feature whose standard deviation is ~0 gets a scale of 1.0 instead of
      being divided by ~0. A constant feature carries no information, so leaving
      it at zero after centering is the right answer; dividing would produce
      inf/NaN and poison the whole fit. sklearn's `_handle_zeros_in_scale` does
      exactly this substitution.
    """

    def __init__(self, zero_variance_threshold=1e-12):
        self.mean = None
        self.standard_deviation = None
        self.zero_variance_threshold = zero_variance_threshold

    def fit(self, features):
        """Store the per-feature mean and standard deviation of `features`."""
        feature_matrix = _as_feature_matrix(features)

        self.mean = feature_matrix.mean(dim=0)
        # unbiased=False is ddof=0, matching StandardScaler.
        self.standard_deviation = feature_matrix.std(dim=0, unbiased=False)
        self.standard_deviation = torch.where(
            self.standard_deviation > self.zero_variance_threshold,
            self.standard_deviation,
            torch.ones_like(self.standard_deviation)
        )
        return self

    def transform(self, features):
        """Apply the stored constants. Never recomputes them -- see the class docstring."""
        if self.mean is None:
            raise RuntimeError("Standardizer.transform called before fit")

        feature_matrix = _as_feature_matrix(features)
        mean = self.mean.to(feature_matrix.device)
        standard_deviation = self.standard_deviation.to(feature_matrix.device)

        if feature_matrix.shape[1] != mean.numel():
            raise ValueError(
                "Standardizer was fitted on " + str(mean.numel()) + " features but got "
                + str(feature_matrix.shape[1])
            )

        return (feature_matrix - mean) / standard_deviation

    def fit_transform(self, features):
        """Convenience for the training half only. Do not call this on held-out data."""
        return self.fit(features).transform(features)


def fit_logistic_regression(features, labels, C, max_iterations=LINEAR_MAX_ITERATIONS,
                            num_classes=NUMBER_OF_CLASSES, tolerance=LINEAR_TOLERANCE):
    """Multinomial logistic regression fitted with L-BFGS. Returns (W, b).

    Minimizes

        0.5 * ||W||_F^2  +  C * sum_i cross_entropy(x_i W + b, y_i)

    which is sklearn's convention EXACTLY, and getting this right is the whole
    reason the module exists. sklearn does not put a lambda in front of the
    penalty; it fixes the penalty at 0.5 * w'w and instead scales the *data-fit*
    term by C, so C is an inverse regularization strength. A textbook
    implementation that writes `mean_cross_entropy + lambda * ||W||^2` would give
    a completely different model for the same numeric "C", and the value of C
    selected in Lior's arm would mean nothing in ours. With the form above, a
    given C is the same amount of regularization in both arms and the two rows of
    the results table are directly comparable.

    (sklearn's source actually minimizes the 1/C-scaled twin,
    `sum_i cross_entropy_i + (1 / (2C)) * ||W||^2`. Multiplying an objective by
    the positive constant C does not move its argmin, so the fitted model is
    identical; only the gradient magnitude differs, which is why the stopping
    tolerance is rescaled below.)

    Note the SUM, not the mean, over samples: with a mean the effective
    regularization would silently depend on how many samples the fold happens to
    hold, and our 800-sample development fits would be regularized 800x more
    strongly than intended.

    The intercept `b` is deliberately absent from the penalty. Penalizing it
    would tie the decision threshold to the (arbitrary) location of the origin in
    feature space and break the model's invariance to a constant shift of the
    features -- sklearn excludes it for the same reason.

    Both the softmax parameterization and the solver mirror the reference: W is
    the full [D, num_classes] matrix (the over-parameterized multinomial form
    sklearn uses for multi_class="multinomial"), the parameters start at zero
    (sklearn's w0 = zeros, which also makes this fit deterministic), and the
    optimizer is L-BFGS with a strong-Wolfe line search, torch's closest
    equivalent to solver="lbfgs".
    """
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, device=feature_matrix.device)

    number_of_samples, number_of_features = feature_matrix.shape
    if label_vector.numel() != number_of_samples:
        raise ValueError(
            "features and labels disagree on N: " + str(number_of_samples)
            + " vs " + str(label_vector.numel())
        )
    if number_of_samples == 0:
        raise ValueError("cannot fit a logistic regression on zero samples")
    if int(label_vector.min().item()) < 0 or int(label_vector.max().item()) >= num_classes:
        raise ValueError(
            "labels must all lie in [0, " + str(num_classes) + ") -- unlabelled "
            "sentinels (-1) must be filtered out before fitting the probe"
        )

    weights = torch.zeros(
        number_of_features, num_classes,
        dtype=torch.float64, device=feature_matrix.device, requires_grad=True
    )
    bias = torch.zeros(
        num_classes,
        dtype=torch.float64, device=feature_matrix.device, requires_grad=True
    )

    regularization_strength = float(C)

    optimizer = torch.optim.LBFGS(
        [weights, bias],
        lr=1.0,
        max_iter=max_iterations,
        # Our objective is C times sklearn's, so its gradient is C times larger.
        # Scaling the threshold by C is what makes "tol = 1e-4" stop the solver at
        # the same point it would stop in Lior's arm. torch's test is
        # max(|grad|) <= tolerance_grad, the same criterion lbfgs uses for gtol.
        tolerance_grad=tolerance * regularization_strength,
        # torch's second stopping test, on the loss/step change. Kept at its
        # default rather than disabled: lbfgs has an equivalent ftol test, and
        # without it a stalled line search would burn all 5000 iterations.
        tolerance_change=1e-9,
        history_size=10,  # lbfgs's m = 10, its default and sklearn's
        line_search_fn="strong_wolfe",
    )

    def objective():
        optimizer.zero_grad(set_to_none=True)
        logits = feature_matrix @ weights + bias
        # reduction="sum" is required by the objective above; see the docstring.
        data_fit = functional.cross_entropy(logits, label_vector, reduction="sum")
        penalty = 0.5 * (weights * weights).sum()
        loss = penalty + regularization_strength * data_fit
        loss.backward()
        return loss

    # One step() call runs the entire L-BFGS loop, because torch's max_iter is the
    # number of iterations *within* a step. This is the direct analogue of
    # sklearn's max_iter, not of an outer training loop.
    optimizer.step(objective)

    return weights.detach(), bias.detach()


def predict_proba(features, W, b):
    """Class probabilities, [N, num_classes] float64. Softmax of the linear scores."""
    feature_matrix = _as_feature_matrix(features, device=W.device)
    logits = feature_matrix @ W + b
    return torch.softmax(logits, dim=1)


def predict(features, W, b):
    """Hard class predictions, [N] int64.

    argmax of the logits rather than of the probabilities: softmax is monotone, so
    the two agree, and skipping it avoids a pointless exp/normalize pass.
    """
    feature_matrix = _as_feature_matrix(features, device=W.device)
    logits = feature_matrix @ W + b
    return logits.argmax(dim=1)


def stratified_k_fold(labels, num_splits, seed):
    """Hand-rolled stratified k-fold. Returns a list of (train_positions, validation_positions).

    Replaces StratifiedKFold(n_splits=..., shuffle=True, random_state=...), which
    build-from-0 forbids. The algorithm is the obvious one: take the positions
    holding each class, shuffle them with a seeded numpy Generator, and deal them
    round-robin into the k folds. Dealing per class is what makes it stratified --
    every fold ends up with (within one) the same number of members of each class
    as every other fold, so a rare class cannot be missing from a validation fold
    and silently distort macro-F1.

    Stratification matters here for a concrete reason: the inner search runs on
    800 images across 10 classes, so a fold is ~160 images and an unstratified
    draw could easily hand one fold 8 members of a class and another 24. The C
    selected would then partly reflect the luck of the draw.

    DOCUMENTED ASYMMETRY, NOT A BUG: this cannot bit-match sklearn's partition.
    The protocol is identical -- k stratified shuffled splits from one seed -- but
    sklearn permutes with its own MT19937 stream, so which image lands in which
    split differs. Both are valid draws from the same distribution of stratified
    partitions; only the draw differs, and the reported number is a mean over 5
    splits and 10 folds, so the effect is noise. `data.stratified_split` carries
    the same caveat for the 800/200 development split.

    Positions are into `labels`, not global dataset indices -- the caller composes
    them with the fold's official indices.
    """
    label_array = np.asarray(labels).reshape(-1)
    number_of_samples = label_array.shape[0]

    if num_splits < 2:
        raise ValueError("num_splits must be at least 2, got " + str(num_splits))
    if number_of_samples < num_splits:
        raise ValueError(
            "cannot make " + str(num_splits) + " splits from " + str(number_of_samples) + " samples"
        )

    generator = np.random.default_rng(seed)

    validation_positions_per_split = [[] for _ in range(num_splits)]

    # Where the next class starts dealing. Without this rotation every class whose
    # count is not divisible by num_splits would dump its remainder on the first
    # folds, and the bias would accumulate over all 10 classes into visibly uneven
    # fold sizes. Carrying the offset spreads the remainders instead, which is the
    # effect StratifiedKFold's global allocation table has.
    rotation = 0

    # Sorted unique classes so the result does not depend on the order the labels
    # happen to appear in -- that is what makes this deterministic given the seed.
    for class_id in np.unique(label_array):
        class_positions = np.flatnonzero(label_array == class_id)
        shuffled = generator.permutation(class_positions)

        split_assignment = (np.arange(len(shuffled)) + rotation) % num_splits
        for split_index in range(num_splits):
            validation_positions_per_split[split_index].append(
                shuffled[split_assignment == split_index])

        rotation = (rotation + len(shuffled)) % num_splits

    splits = []
    all_positions = np.arange(number_of_samples, dtype=np.int64)

    for split_index in range(num_splits):
        validation_positions = np.sort(
            np.concatenate(validation_positions_per_split[split_index])).astype(np.int64)

        held_out = np.zeros(number_of_samples, dtype=bool)
        held_out[validation_positions] = True
        train_positions = all_positions[~held_out]

        if len(validation_positions) == 0 or len(train_positions) == 0:
            raise ValueError(
                "split " + str(split_index) + " came out empty -- too few samples per class "
                "for " + str(num_splits) + " stratified splits"
            )

        splits.append((train_positions, validation_positions))

    return splits


def evaluate_probe(features, labels, W, b, num_classes=NUMBER_OF_CLASSES):
    """Score a fitted probe. Returns {"accuracy", "f1_macro", "log_loss"}.

    Key names and metric conventions match Lior's `classification_metrics` so the
    two arms' result tables can be concatenated without a rename step. All three
    numbers come out of our own hand-written metrics module.
    """
    probabilities = predict_proba(features, W, b)
    predictions = probabilities.argmax(dim=1)
    label_vector = _as_label_vector(labels, device=probabilities.device)

    confusion = metrics.confusion_matrix(label_vector, predictions, num_classes)

    return {
        "accuracy": metrics.accuracy(confusion),
        "f1_macro": metrics.macro_f1(confusion),
        "log_loss": metrics.cross_entropy_loss(probabilities, label_vector),
    }


def cross_validate_c(features, labels, c_grid=LINEAR_C_GRID, num_splits=CROSS_VALIDATION_SPLITS,
                     seed=0, max_iterations=LINEAR_MAX_ITERATIONS,
                     num_classes=NUMBER_OF_CLASSES):
    """Inner C search. Returns one dict per C, in the order of `c_grid`.

    This is the direct translation of Lior's `cross_validate_c_values`: for each C
    in the grid, run `num_splits` stratified splits over the *development-train*
    features only, refit from scratch inside every split, and average accuracy,
    macro-F1 and log loss across the splits. Test is not involved at any point,
    and neither are the 200 development-validation images -- those are scored once
    afterwards with the single selected C.

    THE STANDARDIZER IS REFITTED INSIDE EVERY SPLIT, on that split's training rows
    only. This is the part a Pipeline does for free and a hand-rolled version gets
    wrong: fitting one standardizer on all 800 rows before splitting would let
    each validation fold's mean and variance influence the model that scores it,
    which inflates the CV estimate and biases the choice of C.

    Returned keys ("c", "log_loss_mean", "accuracy_mean", "f1_macro_mean") are his.
    """
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, device=feature_matrix.device)

    # The splitter works on numpy because it is pure index arithmetic; keeping it
    # off the GPU also means the partition is identical whether the features are
    # on CPU or CUDA.
    splits = stratified_k_fold(label_vector.cpu().numpy(), num_splits, seed)

    rows = []

    for c_value in c_grid:
        split_accuracies = []
        split_f1_scores = []
        split_log_losses = []

        for train_positions, validation_positions in splits:
            train_index = torch.as_tensor(train_positions, device=feature_matrix.device)
            validation_index = torch.as_tensor(validation_positions, device=feature_matrix.device)

            standardizer = Standardizer().fit(feature_matrix[train_index])
            train_features = standardizer.transform(feature_matrix[train_index])
            validation_features = standardizer.transform(feature_matrix[validation_index])

            weights, bias = fit_logistic_regression(
                train_features,
                label_vector[train_index],
                C=c_value,
                max_iterations=max_iterations,
                num_classes=num_classes,
            )

            scores = evaluate_probe(
                validation_features, label_vector[validation_index], weights, bias, num_classes)

            split_accuracies.append(scores["accuracy"])
            split_f1_scores.append(scores["f1_macro"])
            split_log_losses.append(scores["log_loss"])

        rows.append({
            "c": float(c_value),
            "log_loss_mean": float(np.mean(split_log_losses)),
            "accuracy_mean": float(np.mean(split_accuracies)),
            "f1_macro_mean": float(np.mean(split_f1_scores)),
        })

    return rows


def select_best_c(cross_validation_rows):
    """Pick one C from the rows returned by `cross_validate_c`.

    Lior's rule, reproduced exactly. Cell 16 of his notebook sorts the C grid by

        log_loss   ascending   <- PRIMARY criterion
        accuracy   descending
        macro-F1   descending
        C          ascending

    and takes the first row. LOG LOSS IS THE PRIMARY CRITERION, not accuracy.
    That distinction is not cosmetic: a large C is barely regularized, and such a
    model routinely wins on accuracy while being badly over-confident and losing
    on log loss. Ranking on accuracy first therefore selects a systematically
    larger C than Lior's arm would, and a gap between the two arms would then be
    partly a gap between two differently-regularized probes rather than between
    two latent spaces -- which is the one thing this module exists to prevent.

    The final ascending-C tie-break keeps the more regularized of two models the
    development data cannot tell apart.
    """
    if not cross_validation_rows:
        raise ValueError("cannot select a C from an empty result list")

    def ranking_key(row):
        # Ascending sort, so accuracy and F1 are negated to make "larger is better"
        # come out first, and log loss and C stay as-is.
        return (row["log_loss_mean"], -row["accuracy_mean"], -row["f1_macro_mean"], row["c"])

    return float(min(cross_validation_rows, key=ranking_key)["c"])


def select_best_c_across_folds(cross_validation_rows_per_fold):
    """Lior's ACTUAL selection: one global C shared by all 10 folds.

    His inner C search runs inside every fold, but the choice is made once: cell
    16 concatenates every fold's search, groups by C, averages log loss / accuracy
    / macro-F1 ACROSS THE FOLDS, and applies `select_best_c`'s ordering to that
    aggregate. Every fold is then refitted with that single C.

    Selecting a C per fold instead is a different -- and slightly more permissive
    -- protocol: it gives our arm ten chances to pick a good regularization
    strength where his arm gets one. Use this function for any number that is
    going to be compared against his.

    `cross_validation_rows_per_fold` is an iterable of the row lists
    `cross_validate_c` returns, one per fold. Returns (selected_c, aggregate_rows)
    where aggregate_rows has the same shape as one `cross_validate_c` result, so
    it can be written to CSV next to the per-fold search.
    """
    totals = {}
    order = []
    for rows in cross_validation_rows_per_fold:
        for row in rows:
            c_value = float(row["c"])
            if c_value not in totals:
                totals[c_value] = {"log_loss_mean": [], "accuracy_mean": [], "f1_macro_mean": []}
                order.append(c_value)
            for metric_name in ("log_loss_mean", "accuracy_mean", "f1_macro_mean"):
                totals[c_value][metric_name].append(float(row[metric_name]))

    if not order:
        raise ValueError("cannot select a C from an empty result list")

    aggregate_rows = [
        {"c": c_value,
         "log_loss_mean": float(np.mean(totals[c_value]["log_loss_mean"])),
         "accuracy_mean": float(np.mean(totals[c_value]["accuracy_mean"])),
         "f1_macro_mean": float(np.mean(totals[c_value]["f1_macro_mean"]))}
        for c_value in order
    ]

    return select_best_c(aggregate_rows), aggregate_rows
