# Barnes-Hut Acceleration for GLOBE

This document describes the Barnes-Hut tree acceleration applied to GLOBE's
field kernel evaluation, reducing the O(N^2) all-to-all interaction cost to
O(N log N).  It assumes familiarity with the base GLOBE architecture (the
whitepaper's Sections 3-4) and focuses on the changes introduced by this
refactor.

---

## 1. Motivation

GLOBE's field kernel computes, for each target point, the influence of *every*
source face on the boundary mesh.  This produces an `(N_tgt, N_src, D)`
displacement tensor, followed by per-pair feature engineering, neural network
evaluation, and an aggregation sum over sources.  The cost is
O(N_tgt * N_src) - quadratic in the mesh size.

This quadratic cost appears in two places:

- **Communication hyperlayers** (boundary-to-boundary): N_src = N_tgt = N_faces.
  With N_faces = 20k, this is 400M interactions per layer.
- **Final prediction** (boundary-to-volume): N_src = N_faces, N_tgt = N_prediction.
  At DrivAerML scale (100k+ faces, 180k prediction points), this is 18 billion
  interactions.

The key observation enabling acceleration is GLOBE's explicit far-field decay
envelope.  The kernel output is multiplied by a Lamb-Oseen-like factor
`(1 - exp(-|r|^2)) / (|r|^2 + 1)^p` that forces contributions to decay as
`1/r^(d-1)` at large distances.  This means distant sources contribute weakly,
and grouping them into clusters introduces only small approximation error.

---

## 2. The Monopole Approximation

For a target point far from a cluster C of source faces, the exact sum

```
exact = sum_{s in C}  strength_s * K(target, source_s, data_s)
```

is approximated by

```
approx = total_strength_C * K(target, centroid_C, avg_data_C)
```

where:

- `centroid_C` is the area-weighted centroid of sources in C
- `avg_data_C` is the area-weighted average of source features (normals,
  latent scalars/vectors)
- `total_strength_C = sum_{s in C} strength_s` is the sum of learned
  per-source strengths

The same neural network evaluates both exact and approximate interactions -
cluster centroids are treated as "virtual sources" with averaged features.
This is a zeroth-order (monopole) Taylor expansion of the kernel about the
cluster centroid.

### Why area-weighting, not strength-weighting?

A subtle but important design choice: the spatial averages (centroid, feature
means) use *area*-weighting, while the multiplicative strength factor is summed
separately.  Areas are fixed geometric properties of the mesh (always positive,
always stable), making the aggregates reusable across kernel branches (the
`MultiscaleKernel` has multiple branches sharing the same source geometry).
Strengths, by contrast, are learned per-source and per-branch values that
change between communication layers.  Separating these concerns means:

1. Aggregates are computed once per forward pass and shared across branches.
2. Only strength summation is per-branch (cheap O(N) work).
3. The aggregation is numerically stable (no division by near-zero when
   learned strengths cancel within a cluster).

---

## 3. Spatial Data Structure: ClusterTree

### 3.1 Construction via LBVH

The tree is built using a Linear Bounding Volume Hierarchy (LBVH) algorithm
(Karras 2012), the same approach used in PhysicsNeMo Mesh's existing `BVH` class
for mesh spatial decomposition. The algorithm:

1. **Morton codes**: Each source point is assigned a 63-bit morton code that
   interleaves the quantized coordinates of the point.  Morton codes produce a
   space-filling Z-curve ordering that preserves spatial locality - nearby
   points in space tend to have nearby morton codes.

2. **Sort**: Sources are sorted by morton code.  After sorting, spatially
   nearby sources are contiguous in the array.

3. **Top-down recursive splitting**: Starting from the full sorted range as
   the root, each segment with more than `leaf_size` sources is split at its
   midpoint.  Because the morton-sorted order preserves spatial locality,
   midpoint splitting approximates a spatial median split, producing a balanced
   binary tree.  Each iteration processes all segments at the current depth in
   parallel, yielding O(log N) Python-level iterations.

4. **Bottom-up axis-aligned bounding box (AABB) propagation**: Leaf AABBs are
   computed from the actual source points they contain.  Internal node AABBs are
   the union of their children's AABBs.  Total areas are similarly propagated
   (sum, not average).

The tree is stored as flat tensor arrays (node_aabb_min, node_aabb_max,
node_left_child, etc.) indexed by node ID, making it fully GPU-compatible.

### 3.2 Node Pre-allocation Bounds

Before construction, we pre-allocate arrays at the worst-case node count.
The midpoint split guarantees each child gets at least `floor(parent_size/2)`
sources, so the minimum leaf occupancy is `ceil(leaf_size/2)`.  The maximum
number of leaves is then `ceil(N / min_per_leaf)`, and by the full-binary-tree
identity (`n_internal = n_leaves - 1`), the maximum total node count is
`2 * max_leaves - 1`.  After construction, the arrays are trimmed to the
actual node count.

### 3.3 Source Aggregates

Per-node aggregate data is computed bottom-up for far-field evaluation:

- **Centroid**: area-weighted mean of source positions
- **Source features** (normals, latent scalars/vectors): area-weighted mean
  via `TensorDict.apply()` with segmented scatter operations
- **Total area**: sum (not average) of children's areas

Internal node aggregates are computed from their children's aggregates using
area-weighted averaging.  This is done via a BFS level-ordering: we discover
which internal nodes are at each depth via a breadth-first traversal from the
root, then process `reversed(depth_levels)` so children are correct before
their parents read from them.

Aggregates depend on the source data (which changes between communication
layers as latent features are updated) but NOT on the tree structure (which
depends only on geometry).  The tree is built once per forward pass; aggregates
are recomputed each time the source data changes.

---

## 4. Interaction Pair Finding

The tree traversal classifies every (target, source) interaction as either
near-field (exact) or far-field (approximate).  The output is an
`InteractionPlan` containing four index arrays:

- `(near_target_ids, near_source_ids)`: pairs requiring exact evaluation
- `(far_target_ids, far_node_ids)`: pairs using the monopole approximation

### 4.1 Opening-Angle Criterion

The standard Barnes-Hut opening criterion compares the ratio of a cluster's
spatial extent to the distance from the target.  Our implementation uses
AABB-distance rather than centroid-distance:

```
is_far = D / r < theta
```

equivalently, `D^2 < theta^2 * r^2`, where D is the AABB diagonal and r is
the distance from the target to the nearest point on the AABB.  In code:

```python
is_far = dist_sq * theta_sq > diam_sq
```

where `dist(target, AABB)` is the distance from the target to the nearest
point on the axis-aligned bounding box, computed as:

```python
clamped = torch.clamp(target, min=aabb_min, max=aabb_max)
dist_sq = (target - clamped).pow(2).sum(dim=-1)
```

This is more robust than centroid-distance because:

- Targets inside a node's AABB always have `dist_sq = 0`, forcing exact
  evaluation.  This eliminates edge cases where the centroid might be near
  the cluster boundary.
- The AABB diameter is a tighter bound on the cluster's spatial extent than
  the centroid-to-farthest-point distance.

### 4.2 Theta Parameter Semantics

The `theta` parameter follows the standard Barnes-Hut convention (Barnes &
Hut 1986, Demmel lecture notes, Heer interactive article).  A cluster is
approximated when `D/r < theta`, where D is the AABB diagonal and r is the
distance from the target to the nearest point on the AABB.

- **Larger theta** = more aggressive (more approximations, faster).
- **Smaller theta** = more conservative (more exact interactions, higher
  accuracy, slower).
- **theta = 0** = all interactions are exact (no approximation).

Typical values for GLOBE: `theta = 0.5` (conservative) to `theta = 1.5`
(aggressive).  The default is `theta = 1.0`, which is the standard moderate
setting recommended in the literature.

### 4.3 Breadth-First Traversal

The traversal processes all active (target, node) pairs at each tree level
simultaneously:

1. Initialize: every target starts paired with the root node.
2. For each level, classify active pairs into three categories:
   - **Far-field**: passes the opening-angle test.  Accumulate the (target,
     node) pair.
   - **Near-field leaf**: fails the test, and the node is a leaf.  Expand
     into per-source (target, source) pairs.
   - **Near-field internal**: fails the test, and the node has children.
     Replace with (target, left_child) and (target, right_child) for the
     next iteration.
3. After all levels, concatenate the accumulated pairs and sort them (near
   pairs by source index, far pairs by node index) for cache-friendly memory
   access during kernel evaluation.

This produces O(N log N) total interaction pairs: each target interacts with
O(log N) tree nodes/sources on average.

### 4.4 Caching Interaction Plans

The interaction plan depends only on the geometric positions of sources and
targets, not on the source data or strengths.  For communication hyperlayers
(where targets = sources), the same plan is reused across all layers.  For the
final prediction evaluation, a separate plan is computed (different target
points).  This eliminates redundant traversals.

---

## 5. Kernel Evaluation Strategy

### 5.1 "Accumulate Pairs, Evaluate Once"

The most important performance principle: the tree traversal produces *only*
integer index arrays.  The expensive neural network evaluation happens exactly
once on the combined batch of all near-field and far-field pairs.  This
minimizes GPU kernel launches and maximizes GPU utilization by running the
network on one large, dense batch.

### 5.2 The `_evaluate_interactions()` Factoring

The core feature engineering pipeline (vector magnitudes, spherical harmonics,
network evaluation, far-field decay, vector reprojection) was extracted into
`Kernel._evaluate_interactions()`.  This method operates on generic
`(*interaction_dims, ...)` tensors - it doesn't know or care whether the
interactions are dense `(N_tgt, N_src)` or sparse `(N_pairs,)`.

- `Kernel.forward()` calls it with `interaction_dims = (N_tgt, N_src)` (dense,
  exact evaluation for all pairs)
- `BarnesHutKernel.forward()` calls it with `interaction_dims = (N_chunk,)`
  (sparse, a chunk of near+far pairs)

This avoids duplicating the ~250-line feature engineering pipeline.

### 5.3 Deferred Per-Chunk Gathering

For large problems, the total number of interaction pairs (near + far) can be
in the millions.  Pre-gathering all float data (displacement vectors, source
features, strengths) for all pairs simultaneously would consume O(N_total *
features * 4 bytes) of GPU memory, potentially exceeding capacity.

The solution: only the compact int64 index arrays are concatenated upfront.
Inside the chunk loop, each chunk gathers its own float data from the raw
per-source and per-node arrays.  This keeps peak float memory at O(chunk_size *
features) regardless of total pair count.

The concatenated index array `all_source_ids` has dual semantics: its first
`n_near` entries are source-point indices (indexing into `source_points`),
while its last `n_far` entries are tree-node indices (indexing into
`aggregates.node_centroid`).  The chunk loop determines how many of each
chunk's pairs are near vs. far and gathers from the appropriate source.

### 5.4 Auto-Chunk Sizing

The `_auto_chunk_size()` method estimates peak memory per interaction pair
from the kernel's feature engineering pipeline and sizes chunks to fit within
50% of *free* GPU memory.  During training, an additional 5x multiplier
accounts for autograd tensor retention.  During inference (no grad), this
multiplier is dropped, allowing larger chunks.

### 5.5 Gradient Checkpointing

When `use_gradient_checkpointing=True` (the default), each chunk's
`_evaluate_interactions()` call is wrapped in `torch.utils.checkpoint.checkpoint`.
This trades compute for memory by recomputing activations during backward
instead of storing them.  Combined with per-chunk gathering, this keeps peak
memory bounded even for very large interaction counts.

---

## 6. Integration with GLOBE

### 6.1 Tree and Plan Lifecycle

Within a single `GLOBE.forward()` call:

1. **Phase 1 (init)**: Build one `ClusterTree` per boundary condition type
   from the cell centroids.  Compute interaction plans for communication
   (targets = sources).  Both are cached for the duration of the forward pass.

2. **Phase 2 (communication)**: For each communication hyperlayer, reuse the
   cached trees and plans.  Only source aggregates are recomputed (the latent
   features change between layers).

3. **Phase 3 (prediction)**: Compute new interaction plans (targets =
   prediction points, different from boundary centroids).  Reuse the same
   trees.

Tree construction and plan finding are decorated with
`@torch.compiler.disable` because they involve irregular control flow (morton
code bit operations, data-dependent loop termination) that `torch.compile`
cannot trace.  The expensive kernel evaluation inside `_evaluate_interactions`
compiles normally.

### 6.2 Shared Aggregates Across Branches

`MultiscaleKernel` computes source aggregates once and passes them to all
`BarnesHutKernel` branches via the `source_aggregates` parameter.  Since
aggregates depend only on geometry and source data (both shared across
branches), this eliminates redundant computation.  Only per-node strength
summation (which depends on per-branch strengths) is computed per-branch.

### 6.3 Dynamic Shapes

The Barnes-Hut approach naturally requires dynamic tensor shapes (each mesh
produces a different tree, different interaction plan, different pair counts).
The training scripts use `torch.compile(dynamic=True)` and
`compile_mode="max-autotune-no-cudagraphs"` to accommodate this.  Mesh padding
(previously used for static-shape CUDA graph compatibility) has been removed.

---

## 7. Parameter Tuning

### 7.1 Theta (Barnes-Hut opening angle)

The `theta` parameter controls accuracy vs. speed, following the standard
Barnes-Hut convention where larger values are more aggressive:

| theta | Character         | Typical use case                       |
|-------|-------------------|----------------------------------------|
| 0     | Exact             | No approximation (equivalent to dense) |
| 0.5   | Conservative      | High accuracy, for validation          |
| 1.0   | Moderate          | Good default for production training   |
| 1.5   | Aggressive        | Fast approximate evaluation            |
| 100+  | Extremely aggressive | Testing only                        |

The approximation error scales as O(theta) per interaction, but the total
error is bounded by the kernel's far-field decay.  Distant clusters contribute
little regardless of approximation quality, providing a natural error ceiling.

### 7.2 Leaf Size

The `leaf_size` parameter (default 32) controls the tree granularity:

- **Smaller leaf_size** (e.g., 4-8): deeper trees, finer-grained near/far
  classification, more far-field approximations, but more traversal overhead
  and more near-field pairs per target.
- **Larger leaf_size** (e.g., 32-64): shallower trees, coarser
  classification, fewer traversal iterations, but each near-field leaf hit
  expands into more individual source interactions.

The optimal leaf size balances:

- **Traversal cost**: O(log(N/leaf_size)) iterations, each processing all
  active pairs.
- **Near-field granularity**: Each near-field leaf contributes up to
  `leaf_size` exact interactions per target.
- **Far-field quality**: Larger leaves have larger AABBs, making the
  opening-angle test harder to pass (fewer far-field approximations for the
  same theta).

For GLOBE with typical boundary mesh sizes (1k-100k faces), `leaf_size=32`
provides a good balance.

---

## 8. Complexity Analysis

| Component          | Time complexity     | Memory complexity   |
|--------------------|---------------------|---------------------|
| Tree construction  | O(N log N)          | O(N)                |
| Aggregate computation | O(N)             | O(N)                |
| Interaction finding | O(N log N)         | O(N log N)          |
| Kernel evaluation  | O(N log N)          | O(chunk_size)       |
| Scatter-add        | O(N log N)          | O(N_targets)        |
| **Total**          | **O(N log N)**      | **O(N log N)**      |

Compare with the all-to-all baseline:

| Component          | Time complexity     | Memory complexity   |
|--------------------|---------------------|---------------------|
| Dense displacement | O(N^2)              | O(N^2)              |
| Feature engineering| O(N^2)              | O(N^2)              |
| Network evaluation | O(N^2)              | O(N^2)              |
| Aggregation        | O(N^2)              | O(N)                |
| **Total**          | **O(N^2)**          | **O(N^2)**          |

For N = 100k sources and targets, this represents a ~5000x reduction in
interaction count (from 10 billion to ~2 million at theta=1.0).

---

## 9. Architecture Summary

```
GLOBE.forward()
  |
  +-- _build_trees_and_plans()        [outside torch.compile]
  |     Build ClusterTree per BC type
  |     Find InteractionPlan (comm: targets=sources)
  |
  +-- Phase 2: Communication hyperlayers (repeat n_comm times)
  |     |
  |     +-- _evaluate_hyperlayer()
  |           |
  |           +-- MultiscaleKernel.forward()
  |                 |
  |                 +-- compute_source_aggregates()  [once, shared]
  |                 |
  |                 +-- BarnesHutKernel.forward()    [per branch]
  |                       |
  |                       +-- _compute_node_strengths()
  |                       +-- for each chunk:
  |                       |     gather near/far data
  |                       |     _evaluate_interactions()  [the hot path]
  |                       |     scatter_add to output
  |                       +-- return per-target result
  |
  +-- _build_prediction_plans()       [outside torch.compile]
  |     Find InteractionPlan (pred: targets=prediction_points)
  |
  +-- Phase 3: Final evaluation
        (same structure as communication, different target points)
```

---

## 10. Testing Strategy

The implementation is validated through several complementary test categories:

- **Convergence to exact**: At large theta (100+), `BarnesHutKernel` output
  converges to the exact `Kernel` output within floating-point tolerance.
  Tested across all combinations of 2D/3D, scalar/vector outputs, and
  scalar/vector source features.

- **Source coverage invariant**: For every target, the union of near-field
  sources and far-field node subtrees equals the complete source set
  `{0, ..., N-1}` with no duplicates and no omissions.  This is the
  fundamental correctness property of the tree traversal.

- **Gradient correctness**: Gradients through `BarnesHutKernel` match exact
  `Kernel` gradients at high theta, verifying that the non-differentiable
  traversal decisions don't corrupt gradient flow through the differentiable
  kernel evaluation.

- **Equivariance preservation**: Translation, rotation, and source-permutation
  equivariance are preserved by the Barnes-Hut approximation, verified at
  both moderate and high theta.

- **Nested key structure**: Tests with deeply nested TensorDict keys matching
  GLOBE's actual production data format (physical/latent/strength namespaces).

---

## 11. References

- Barnes & Hut (1986). "A hierarchical O(N log N) force-calculation algorithm."
  *Nature* 324, 446-449.
- Karras (2012). "Maximizing Parallelism in the Construction of BVHs, Octrees,
  and k-d Trees." *HPG 2012*.  The LBVH construction algorithm used here.
- Burtscher & Pingali (2011). "An Efficient CUDA Implementation of the
  Tree-Based Barnes Hut n-Body Algorithm." *GPU Computing Gems Emerald Edition*.
- Lukat & Banerjee (2015). "A GPU accelerated Barnes-Hut tree code for FLASH4."
  Describes AABB-distance opening criterion.
- Madan et al. (2025). "Stochastic Barnes-Hut Approximation of Kernel Matrices."
  *SIGGRAPH 2025*.  Uses the `beta = 1/theta_classical` convention adopted here.
