"""Chunked cosine KNN graph and the row-normalized propagation matrix P.

This is the memory-critical module of Tomer's arm, and the one place where the
"build from 0" constraint costs us the most: faiss, sklearn's NearestNeighbors
and every approximate-NN library are banned, so the neighbour search is an exact
brute-force cosine search written out here in torch.

WHY CHUNKING IS NOT OPTIONAL
----------------------------
The graph is built over all N = 100000 unlabeled + 5000 labeled train images, so
N = 105000. The full similarity matrix is

    105000 * 105000 * 4 bytes (fp32) = 44,100,000,000 bytes ~= 44.1 GB

which fits in neither the RTX 5090's 32 GB of VRAM nor the server's RAM. But we
never actually need the matrix -- we need the top-K of each row. So we compute it
one horizontal strip at a time and throw the strip away as soon as its top-K has
been read off:

    one chunk   1024 * 105000 * 4 bytes  ~= 430 MB     (transient)
    embeddings  105000 *   512 * 4 bytes ~= 215 MB     (resident)
    output idx  105000 *    50 * 8 bytes ~=  42 MB     (resident, K = 50)
    output sim  105000 *    50 * 4 bytes ~=  21 MB     (resident, K = 50)
                                            --------
    peak                                    ~< 1 GB

A 44 GB problem becomes a sub-1 GB one, at the cost of 103 sequential matmuls
instead of a single big one -- the same total FLOPs either way. That trade is the
paragraph this module contributes to the report.

CORRECTNESS OF THE CHUNKING
---------------------------
Chunk c computes `embeddings[start:end] @ embeddings.T`, which is exactly rows
start..end of the full product `embeddings @ embeddings.T`. Nothing about a row's
top-K depends on the other rows, so the chunked result is not an approximation --
it is the same algorithm as the textbook one, not a cheaper stand-in, and a unit
test on a small N can prove it against a direct full-matrix `topk`.

The one honest caveat, measured rather than assumed: the returned INDICES are
bit-identical to the full-matrix reference for every chunk size, but the returned
similarity VALUES can differ by one fp32 ULP (~1.2e-7). BLAS picks a different
blocking strategy for a [1, D] x [D, N] product than for a [37, D] x [D, N] one,
and summing 512 products in a different order rounds differently in the last bit.
So a test should assert `torch.equal` on the indices and `torch.allclose` on the
values. See the note in `build_knn_graph` for when that could in principle also
flip an index.

Everything takes a `device` argument or infers it from its inputs; nothing here
hardcodes `.cuda()`, so the whole module runs under CPU unit tests and streams on
CUDA for the real N.
"""

import torch


# Default rows of the similarity matrix per chunk. Mirrors
# config.SIMILARITY_CHUNK_ROWS; kept as a literal default here so this module has
# no import-time dependency on the config module (the rest of src/ is standalone
# the same way).
DEFAULT_CHUNK_ROWS = 1024

# Floor for any quantity we are about to divide by. Small enough to be invisible
# next to a real norm or bandwidth, large enough that fp32 division stays finite.
DIVISION_EPSILON = 1e-12


def _resolve_device(embeddings, device):
    """Pick the compute device: the caller's choice, else wherever the data already is."""
    if device is not None:
        return torch.device(device)
    if isinstance(embeddings, torch.Tensor):
        return embeddings.device
    return torch.device("cpu")


def _as_float32_tensor(values, device):
    """Accept torch or numpy, return a contiguous float32 tensor on `device`.

    Cached embeddings are stored as fp16 on disk (105000 x 512 fp16 ~= 107 MB per
    CLAUDE.md), so an up-cast to fp32 here is the normal path, not an edge case.
    The similarity search runs in fp32: fp16 accumulation over a 512-long dot
    product loses enough precision to reorder near-ties in the top-K, which would
    silently change the graph between runs.
    """
    if isinstance(values, torch.Tensor):
        tensor = values.to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    return tensor.contiguous()


def l2_normalize(embeddings, device=None):
    """Scale every row to unit L2 norm, so a dot product IS the cosine similarity.

    This is the whole reason the search below can be a single matmul: after
    normalization, `a . b == cos(a, b)`, and the most-similar neighbours are just
    the largest entries of `E @ E.T`. Without it we would need per-row norm
    divisions inside every chunk.

    Zero-norm rows are the failure mode to guard: a collapsed posterior (the
    beta = 1 degenerate case CLAUDE.md warns about) can produce a mu of all
    zeros, and dividing by its norm would fill the embedding matrix with NaN,
    which then poisons `topk` for every other row too. Clamping the norm at
    DIVISION_EPSILON leaves such a row as an exact zero vector instead: it gets
    similarity 0 to everything, which is a meaningless but finite and debuggable
    result.

    Idempotent -- normalizing an already-normalized matrix is a no-op -- which is
    why `build_knn_graph` can call it unconditionally without the caller having
    to track whether it was already done.
    """
    compute_device = _resolve_device(embeddings, device)
    tensor = _as_float32_tensor(embeddings, compute_device)

    if tensor.dim() != 2:
        raise ValueError("embeddings must be [N, D], got shape " + str(tuple(tensor.shape)))

    norms = torch.linalg.vector_norm(tensor, ord=2, dim=1, keepdim=True)
    return tensor / torch.clamp(norms, min=DIVISION_EPSILON)


def build_knn_graph(embeddings, k, chunk_rows=DEFAULT_CHUNK_ROWS, device=None):
    """Exact cosine K-nearest-neighbour search, streamed one strip of rows at a time.

    `embeddings` is [N, D] (torch or numpy, fp16 or fp32). It is L2-normalized
    internally, so the caller cannot forget to do it; the operation is idempotent
    so passing already-normalized input costs one wasted pass and nothing else.

    Returns (neighbor_indices, neighbor_similarities):
        neighbor_indices        int64   [N, K]  column j = the j-th closest node
        neighbor_similarities   float32 [N, K]  its cosine similarity, in [-1, 1]
    both sorted by DECREASING similarity (column 0 is the closest neighbour and
    column K-1 the furthest), which `heat_kernel_weights` and the report's
    neighbour-distance statistics rely on. Both come back on the device the
    embeddings arrived on -- a numpy input yields CPU tensors -- so a CUDA build
    does not silently strand 63 MB of graph on the GPU for the pure-Python
    union-find to fetch back one element at a time.

    The loop body is exactly the four steps the memory argument in the module
    docstring describes:
      1. `chunk @ embeddings.T` -> [chunk_rows, N] fp32, ~430 MB at the defaults;
      2. write -inf onto the diagonal entries, so a node is never returned as its
         own nearest neighbour (it would otherwise always win, with similarity
         exactly 1.0, and we would effectively get K-1 real neighbours);
      3. `topk(k)` -> [chunk_rows, K];
      4. drop the strip before allocating the next one.

    On step 4: `del` is enough. CUDA's caching allocator hands the freed 430 MB
    block straight back to the identically-shaped next chunk, so the loop reaches
    a steady state after the first iteration and never grows. Calling
    `torch.cuda.empty_cache()` per chunk would defeat exactly that reuse.

    EXACTNESS. Because each chunk is a genuine slice of `E @ E.T`, this returns
    what a direct full-matrix `topk` returns, and a unit test on a small N should
    check that for several `chunk_rows` values including ones that do not divide
    N: `torch.equal` on the indices, `torch.allclose` on the similarities. The
    values need the tolerance because BLAS blocks a 1-row product differently
    from a 37-row one and the 512-term dot product rounds differently in the last
    bit -- measured at 1 ULP, ~1.2e-7. That ULP can only change an INDEX if two
    neighbours are within 1 ULP of each other, i.e. an all-but-exact tie, which
    `topk` would then break by index anyway. On CUDA keep TF32 matmuls off: they
    truncate the mantissa to ~10 bits, which manufactures exactly those ties and
    would make the graph differ between machines.
    """
    compute_device = _resolve_device(embeddings, device)
    normalized = l2_normalize(embeddings, device=compute_device)

    number_of_nodes = normalized.shape[0]
    neighbors_per_node = int(k)

    if neighbors_per_node < 1:
        raise ValueError("k must be >= 1, got " + str(neighbors_per_node))
    # The diagonal is masked out, so a node has only N-1 candidate neighbours.
    if neighbors_per_node > number_of_nodes - 1:
        raise ValueError(
            "k must be <= N - 1 (a node is not its own neighbour): k=" + str(neighbors_per_node)
            + " with N=" + str(number_of_nodes)
        )

    rows_per_chunk = int(chunk_rows)
    if rows_per_chunk < 1:
        raise ValueError("chunk_rows must be >= 1, got " + str(rows_per_chunk))

    neighbor_indices = torch.empty(
        (number_of_nodes, neighbors_per_node), dtype=torch.int64, device=compute_device)
    neighbor_similarities = torch.empty(
        (number_of_nodes, neighbors_per_node), dtype=torch.float32, device=compute_device)

    # Transposed once outside the loop: this is a view, not a copy, but hoisting
    # it makes it obvious that the 215 MB embedding matrix is the only resident
    # allocation the loop touches.
    transposed = normalized.t()

    for chunk_start in range(0, number_of_nodes, rows_per_chunk):
        chunk_end = min(chunk_start + rows_per_chunk, number_of_nodes)

        similarity_chunk = normalized[chunk_start:chunk_end] @ transposed

        # Mask the diagonal. Row r of the chunk is global node chunk_start + r,
        # so its self-similarity sits at column chunk_start + r. -inf rather than
        # -1 because -1 is a legal cosine similarity and could still be selected
        # in a pathological graph.
        positions_in_chunk = torch.arange(chunk_end - chunk_start, device=compute_device)
        similarity_chunk[positions_in_chunk, positions_in_chunk + chunk_start] = float("-inf")

        chunk_similarities, chunk_indices = torch.topk(
            similarity_chunk, neighbors_per_node, dim=1, largest=True, sorted=True)

        neighbor_indices[chunk_start:chunk_end] = chunk_indices
        neighbor_similarities[chunk_start:chunk_end] = chunk_similarities

        # Free the strip before the next one is allocated -- the point of the
        # whole exercise. Without this the two 430 MB blocks would be alive
        # simultaneously at the top of the next iteration.
        del similarity_chunk, chunk_similarities, chunk_indices

    output_device = _resolve_device(embeddings, None)
    return neighbor_indices.to(output_device), neighbor_similarities.to(output_device)


def heat_kernel_weights(neighbor_similarities, sigma=None):
    """Turn cosine similarities into edge weights with a heat (Gaussian) kernel.

        weight = exp((similarity - 1) / sigma)

    Since the rows are L2-normalized, `1 - similarity` is the cosine distance, so
    this is the familiar `exp(-distance / sigma)`: two identical vectors get
    weight exactly 1.0 and the weight decays smoothly to 0 as the neighbour gets
    further away. Writing it as `similarity - 1` rather than `-(1 - similarity)`
    keeps the exponent's argument <= 0 by construction, so the exponential can
    only underflow towards 0, never overflow.

    Why a kernel at all instead of binary 0/1 edges: label propagation averages
    over a node's neighbourhood, and with binary weights the K-th neighbour --
    which for K = 50 on STL-10 can be a completely unrelated image -- pulls
    exactly as hard as the nearest one. The kernel makes the graph's confidence
    in an edge part of the propagation.

    CHOOSING SIGMA. `sigma=None` selects a self-tuning bandwidth: the mean over
    all nodes of the cosine distance to that node's furthest kept neighbour (the
    K-th one). The rationale is that sigma has to be on the scale of the
    distances actually present in the graph, and that scale is not knowable in
    advance -- it changes with beta (a stronger KL term shrinks the latent and
    compresses every distance), with the encoder checkpoint, and with K itself.
    A fixed sigma would therefore mean something different in every cell of the
    beta x K sweep: too large and every weight collapses to ~1, which is the
    binary graph we just argued against; too small and only the single nearest
    neighbour carries any mass, which fragments propagation. Tying sigma to the
    K-th-neighbour distance makes "the furthest neighbour we kept" land at
    roughly exp(-1) of the nearest one's weight in every configuration, so the
    sweep compares graphs of comparable sharpness.

    We use one global scalar rather than the per-node local scaling of
    Zelnik-Manor & Perona (sigma_i * sigma_j). Local scaling would give every
    node its own bandwidth and produce an asymmetric weight matrix whose
    asymmetry then interacts with the `max(W, W^T)` union step in a way that is
    hard to reason about -- and a single scalar is a number we can report and
    compare across runs, which a per-node vector is not.

    `neighbor_similarities` is the [N, K] value array from `build_knn_graph`. The
    bandwidth uses `min` over each row rather than the last column, so it is
    correct whether or not the rows happen to be sorted.

    Returns (weights float32 [N, K], sigma float) -- sigma is returned because it
    is a derived hyperparameter of the graph and belongs in the run's metadata.
    """
    similarities = neighbor_similarities
    if not isinstance(similarities, torch.Tensor):
        similarities = torch.as_tensor(similarities, dtype=torch.float32)
    similarities = similarities.to(torch.float32)

    if similarities.dim() != 2:
        raise ValueError(
            "neighbor_similarities must be [N, K], got shape " + str(tuple(similarities.shape)))

    if sigma is None:
        distance_to_furthest_neighbor = 1.0 - similarities.min(dim=1).values
        bandwidth = float(distance_to_furthest_neighbor.mean().item())
        # Degenerate graph guard: if every kept neighbour is at distance 0 (a
        # fully collapsed latent, where all mu are the same vector) the mean is
        # 0 and the division below would be 0/0 -> NaN. The floor keeps the run
        # alive so the collapse shows up as a useless-but-finite accuracy rather
        # than a crash halfway through a sweep.
        bandwidth = max(bandwidth, DIVISION_EPSILON)
    else:
        bandwidth = float(sigma)
        if bandwidth <= 0.0:
            raise ValueError("sigma must be > 0, got " + str(bandwidth))

    weights = torch.exp((similarities - 1.0) / bandwidth)
    return weights, bandwidth


def symmetrize_union(neighbor_indices, weights, num_nodes):
    """Build the symmetric weight matrix W = max(W, W^T) as a sparse COO tensor.

    The raw KNN relation is directed: "j is one of i's K nearest" does not imply
    the reverse. Label propagation needs an undirected graph, and there are two
    standard ways to get one:

        intersection (mutual KNN)  keep the edge only if both lists contain the other
        union        (this one)    keep the edge if EITHER list contains the other

    We take the union. A hub node -- a dense, prototypical image that sits in the
    middle of its class cluster -- appears in hundreds of other nodes' neighbour
    lists while its own K slots are filled by its immediate neighbours only.
    Under intersection almost all of those incoming edges are deleted and the hub
    is nearly disconnected, which is exactly backwards: the hub is the node label
    mass should be flowing THROUGH. The union keeps them. The price is that the
    union graph is denser and can bridge two classes through one bad edge, which
    is precisely what `metrics.edge_purity` measures in the K sweep.

    `max` rather than `(W + W^T)/2` or `W + W^T`: the weight is a confidence in
    the edge, and if either endpoint considers the other a close neighbour then
    the edge is that close, regardless of the other's crowded neighbourhood.
    Averaging would halve the weight of every one-directional edge purely because
    of the other endpoint's local density.

    IMPLEMENTATION. `torch.sparse_coo_tensor(...).coalesce()` reduces duplicate
    (row, col) entries by SUM, which is not what we want, and there is no
    `coalesce(reduce="amax")`. So the max-reduce is done by hand:

      1. flatten the directed edges to (row, col, weight) triples, then append
         the reversed triples (col, row, weight) -- the union of both directions;
      2. encode each (row, col) pair as the single integer `row * N + col`
         (unique because col < N; at N = 105000 the largest key is ~1.1e10, far
         inside int64);
      3. sort by weight ascending, then STABLE-sort by key, so within each key's
         run the weights are ascending and the LAST entry of the run is the max;
      4. keep the last entry of each run.

    Sorting is used rather than `scatter_reduce(reduce="amax")` because sort is
    deterministic on CUDA, and `seeding.enable_deterministic()` makes PyTorch
    raise on ops that have no deterministic kernel. The graph must be
    reproducible: it is the object the entire comparison rests on.

    Note that each node's own K entries are already distinct, so a key can repeat
    at most twice (once per direction) and step 3's max is exactly max(W, W^T).

    On memory, since that is this module's theme: the edge list is 2 * N * K
    entries, which at N = 105000 and K = 50 is 10.5M edges, and the sort keeps
    roughly a dozen buffers of that length alive (~800 MB peak). That is the
    second-largest allocation in the pipeline after the similarity strip, and it
    is why the intermediates are freed explicitly below.

    Returns a coalesced sparse COO float32 tensor of shape [num_nodes, num_nodes]
    on the same device as `weights`.
    """
    indices = neighbor_indices
    if not isinstance(indices, torch.Tensor):
        indices = torch.as_tensor(indices)
    indices = indices.to(torch.int64)

    edge_weights = weights
    if not isinstance(edge_weights, torch.Tensor):
        edge_weights = torch.as_tensor(edge_weights, dtype=torch.float32, device=indices.device)
    edge_weights = edge_weights.to(device=indices.device, dtype=torch.float32)

    if indices.dim() != 2:
        raise ValueError("neighbor_indices must be [N, K], got shape " + str(tuple(indices.shape)))
    if edge_weights.shape != indices.shape:
        raise ValueError(
            "weights must have the same shape as neighbor_indices, got "
            + str(tuple(edge_weights.shape)) + " and " + str(tuple(indices.shape))
        )

    number_of_nodes = int(num_nodes)
    if indices.shape[0] != number_of_nodes:
        raise ValueError(
            "neighbor_indices has " + str(indices.shape[0]) + " rows but num_nodes is "
            + str(number_of_nodes)
        )
    if indices.numel() > 0:
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= number_of_nodes:
            raise ValueError("neighbor_indices contains an index outside [0, num_nodes)")

    device = indices.device
    neighbors_per_node = indices.shape[1]

    source_nodes = torch.arange(number_of_nodes, device=device).repeat_interleave(neighbors_per_node)
    target_nodes = indices.reshape(-1)
    flat_weights = edge_weights.reshape(-1)

    # Self-loops carry no information for propagation (a node would just recycle
    # its own mass) and would distort the row normalization. build_knn_graph
    # masks the diagonal so there should be none; this is belt-and-braces for
    # hand-constructed test graphs, and mirrors the same guard in
    # metrics.edge_purity and unionfind.connected_components.
    keeps_edge = source_nodes != target_nodes
    source_nodes = source_nodes[keeps_edge]
    target_nodes = target_nodes[keeps_edge]
    flat_weights = flat_weights[keeps_edge]

    # Step 1: the union of both directions.
    all_rows = torch.cat([source_nodes, target_nodes])
    all_columns = torch.cat([target_nodes, source_nodes])
    all_weights = torch.cat([flat_weights, flat_weights])

    if all_rows.numel() == 0:
        empty_indices = torch.empty((2, 0), dtype=torch.int64, device=device)
        empty_values = torch.empty((0,), dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(
            empty_indices, empty_values, (number_of_nodes, number_of_nodes)).coalesce()

    # Step 2: one integer per (row, column) pair.
    edge_keys = all_rows * number_of_nodes + all_columns

    # Step 3: weight-ascending, then stable by key.
    by_weight = torch.argsort(all_weights)
    keys_by_weight = edge_keys[by_weight]
    weights_by_weight = all_weights[by_weight]

    by_key = torch.argsort(keys_by_weight, stable=True)
    sorted_keys = keys_by_weight[by_key]
    sorted_weights = weights_by_weight[by_key]

    del all_rows, all_columns, all_weights, edge_keys
    del by_weight, keys_by_weight, weights_by_weight, by_key

    # Step 4: the last entry of each equal-key run holds that pair's max weight.
    is_last_of_run = torch.ones_like(sorted_keys, dtype=torch.bool)
    is_last_of_run[:-1] = sorted_keys[1:] != sorted_keys[:-1]

    unique_keys = sorted_keys[is_last_of_run]
    unique_weights = sorted_weights[is_last_of_run]

    unique_rows = unique_keys // number_of_nodes
    unique_columns = unique_keys - unique_rows * number_of_nodes

    symmetric_indices = torch.stack([unique_rows, unique_columns])
    return torch.sparse_coo_tensor(
        symmetric_indices, unique_weights, (number_of_nodes, number_of_nodes)).coalesce()


def _row_sums(row_indices, values, num_nodes):
    """Sum the sparse values per row.

    `torch.bincount` with `weights` is the same one-call primitive metrics.py
    uses for the confusion matrix, and it is the natural fit here. The catch is
    that bincount on CUDA is implemented with atomics, and
    `seeding.enable_deterministic()` makes PyTorch raise rather than run it. The
    row sums are a few million float adds, so when that happens we simply do the
    reduction on the CPU and move the [N] result back -- once, not per chunk.
    """
    try:
        return torch.bincount(row_indices, weights=values.to(torch.float64), minlength=num_nodes)
    except RuntimeError:
        cpu_sums = torch.bincount(
            row_indices.cpu(), weights=values.detach().cpu().to(torch.float64), minlength=num_nodes)
        return cpu_sums.to(values.device)


def row_normalize(sparse_weight_matrix):
    """Row-normalize W into the propagation matrix P, so every non-empty row sums to 1.

    P is the operator label propagation iterates: `F <- P F` replaces each node's
    label distribution with the weighted average of its neighbours'. Rows must
    sum to 1 for that to be an average rather than an arbitrary rescaling -- with
    unnormalized rows, high-degree nodes would amplify their mass at every
    iteration and the whole thing would diverge.

    Isolated nodes are the case to get right. A node with no edges simply has no
    entries in the sparse matrix, so its row of P is all zeros and it keeps
    whatever it started with under `F <- P F` (zeros for an unlabeled node, and
    labeled rows are re-clamped every step anyway). A node whose weights all
    underflowed to exactly 0.0 -- possible when sigma is tiny and every neighbour
    is far -- would divide 0 by 0, so the reciprocal is taken only where the row
    sum is positive and set to 0 elsewhere. Both cases end up as an all-zero row
    rather than NaN, and NaN would propagate to every node in the component
    within a couple of iterations.

    The sums are accumulated in float64 before the reciprocal: a hub in the union
    graph can have thousands of incident edges, and we want the "rows sum to 1"
    invariant to hold to more digits than a float32 accumulation of thousands of
    terms would give.

    Returns a coalesced sparse COO float32 tensor of the same shape.
    """
    if not sparse_weight_matrix.is_sparse:
        raise ValueError("row_normalize expects a sparse COO tensor")

    coalesced = sparse_weight_matrix.coalesce()
    indices = coalesced.indices()
    values = coalesced.values()
    number_of_nodes = coalesced.shape[0]

    if values.numel() == 0:
        return coalesced

    row_indices = indices[0]
    row_totals = _row_sums(row_indices, values, number_of_nodes)

    inverse_totals = torch.where(
        row_totals > 0,
        1.0 / torch.clamp(row_totals, min=DIVISION_EPSILON),
        torch.zeros_like(row_totals)
    )

    normalized_values = (values.to(torch.float64) * inverse_totals[row_indices]).to(torch.float32)

    return torch.sparse_coo_tensor(
        indices, normalized_values, coalesced.shape).coalesce()


def build_propagation_matrix(embeddings, k, sigma=None,
                             chunk_rows=DEFAULT_CHUNK_ROWS, device=None):
    """Embeddings -> propagation matrix P, running the whole graph pipeline in order.

    This is the single entry point the K sweep calls: L2-normalize, chunked
    cosine KNN, heat-kernel weights, union symmetrization, row normalization.

    Returns (propagation_matrix, neighbor_indices, neighbor_similarities, sigma):
        propagation_matrix      sparse COO float32 [N, N], rows summing to 1
        neighbor_indices        int64   [N, K]
        neighbor_similarities   float32 [N, K]
        sigma                   float, the bandwidth actually used

    The raw neighbour arrays are handed back rather than discarded because two
    required diagnostics consume them directly and neither can recover them from
    P: `metrics.edge_purity` needs the directed [N, K] lists (P's symmetrized,
    reweighted entries would give a different, less interpretable number), and
    `unionfind.connected_components` needs them to report the component coverage
    that bounds what LP can achieve at this K. Recomputing the graph for each
    would mean a second 105k-node search per sweep cell.
    """
    neighbor_indices, neighbor_similarities = build_knn_graph(
        embeddings, k, chunk_rows=chunk_rows, device=device)

    weights, bandwidth = heat_kernel_weights(neighbor_similarities, sigma=sigma)

    symmetric_weights = symmetrize_union(
        neighbor_indices, weights, neighbor_indices.shape[0])

    propagation_matrix = row_normalize(symmetric_weights)

    return propagation_matrix, neighbor_indices, neighbor_similarities, bandwidth
