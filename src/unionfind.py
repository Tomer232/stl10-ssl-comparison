"""Union-find (disjoint set) and connected components of the KNN graph.

Build-from-0: `scipy.sparse.csgraph.connected_components` is the obvious library
call and it is banned, so the disjoint-set forest is written out here. It is a
short algorithm and the two standard optimisations -- path compression in `find`
and union by rank in `union` -- are what keep it effectively linear, so both are
implemented rather than hand-waved.

Why we need components at all: label propagation cannot cross between components.
Any node in a component with no labeled seed is permanently unreachable, so the
component structure is a hard ceiling on LP and gets reported next to accuracy in
the K sweep (see metrics.component_label_coverage).
"""

import numpy as np
import torch


class UnionFind:
    """Disjoint-set forest over nodes 0..size-1, with path compression and union by rank."""

    def __init__(self, size):
        self.size = int(size)
        # parent[x] == x means x is the root of its component.
        self.parent = list(range(self.size))
        # Rank is an upper bound on tree height, used only to decide which root
        # gets attached to which. It is NOT the component size -- component sizes
        # are tracked separately because path compression invalidates rank as a
        # size proxy.
        self.rank = [0] * self.size
        self.size_of_component = [1] * self.size
        self.number_of_components = self.size

    def find(self, x):
        """Return the root of x's component, compressing the path on the way out.

        Written iteratively on purpose. The recursive one-liner is prettier but
        N is 105k here and a long chain would blow Python's recursion limit
        before compression ever had a chance to flatten it.
        """
        root = x
        while self.parent[root] != root:
            root = self.parent[root]

        # Second pass: point every node we walked through straight at the root,
        # so the next find on any of them is O(1).
        while self.parent[x] != root:
            next_node = self.parent[x]
            self.parent[x] = root
            x = next_node

        return root

    def union(self, x, y):
        """Merge the components of x and y. Returns True if they were distinct."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by rank: hang the shallower tree under the deeper one so the
        # height only grows when both sides are equally deep.
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x
        self.size_of_component[root_x] += self.size_of_component[root_y]
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

        self.number_of_components -= 1
        return True

    def component_id(self, x):
        """Representative id of x's component.

        Identical to `find`; the separate name is for call sites that are asking
        "which component is this node in" rather than performing a merge, which
        reads better at the point of use.
        """
        return self.find(x)

    def component_sizes(self):
        """Return {root id: number of nodes in that component}.

        Useful for the report: on STL-10 we expect one giant component plus a
        tail of small islands, and the size of that tail is the story.
        """
        sizes = {}
        for node in range(self.size):
            root = self.find(node)
            sizes[root] = self.size_of_component[root]
        return sizes


def connected_components(neighbor_indices, num_nodes):
    """Components of the SYMMETRIZED KNN graph.

    `neighbor_indices` is the [N, K] neighbour index array from the chunked topk.
    Returns (component_ids, number_of_components) where component_ids is an int64
    numpy array of length num_nodes, relabelled to a contiguous 0..M-1 range --
    metrics.component_label_coverage bincounts over it, which requires contiguous
    ids.

    Symmetrization is implicit: union(i, j) makes no distinction between "j is in
    i's top-K" and "i is in j's top-K", which is exactly the `W = max(W, W^T)`
    union rule the pipeline uses for the weights.

    On cost: this is one union per directed edge, so N*K operations, each
    effectively O(1) after path compression and union by rank. At N=105k and
    K=50 that is ~5.2M unions -- a handful of seconds in pure Python. There is no
    performance argument for scipy.sparse.csgraph here, which is convenient given
    that build-from-0 forbids it anyway.
    """
    if isinstance(neighbor_indices, torch.Tensor):
        neighbors = neighbor_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        neighbors = np.asarray(neighbor_indices).astype(np.int64, copy=False)

    if neighbors.ndim != 2:
        raise ValueError("neighbor_indices must be [N, K], got shape " + str(neighbors.shape))

    number_of_nodes = int(num_nodes)
    if neighbors.shape[0] != number_of_nodes:
        raise ValueError(
            "neighbor_indices has " + str(neighbors.shape[0]) + " rows but num_nodes is "
            + str(number_of_nodes)
        )
    if number_of_nodes > 0 and neighbors.size > 0:
        if int(neighbors.min()) < 0 or int(neighbors.max()) >= number_of_nodes:
            raise ValueError("neighbor_indices contains an index outside [0, num_nodes)")

    disjoint_set = UnionFind(number_of_nodes)

    # .tolist() once up front: indexing a numpy array element by element inside a
    # Python loop pays a scalar-boxing cost on every access, and at 5M edges that
    # dominates the runtime. A plain list of lists is several times faster here.
    neighbor_rows = neighbors.tolist()
    for node in range(number_of_nodes):
        for neighbor in neighbor_rows[node]:
            if neighbor != node:  # self-loops carry no connectivity information
                disjoint_set.union(node, neighbor)

    roots = np.empty(number_of_nodes, dtype=np.int64)
    for node in range(number_of_nodes):
        roots[node] = disjoint_set.find(node)

    # Root ids are arbitrary node indices; remap them to 0..M-1 so downstream
    # bincounts do not allocate an N-length histogram full of zeros.
    _, component_ids = np.unique(roots, return_inverse=True)
    component_ids = component_ids.astype(np.int64, copy=False)

    return component_ids, disjoint_set.number_of_components
