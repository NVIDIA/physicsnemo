# Hierarchical Acceleration for GLOBE

This document describes the dual-tree hierarchical acceleration applied to
GLOBE's field kernel evaluation, reducing the O(N^2) all-to-all interaction
cost to O(N log N).  It assumes familiarity with the base GLOBE architecture
(the whitepaper's Sections 3-4) and focuses on the acceleration strategy.

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

For a target point far from a cluster C_S of source faces, the exact sum

```text
exact = sum_{s in C_S}  strength_s * K(target, source_s, data_s)
```

is approximated by

```text
approx = total_strength_{C_S} * K(target, centroid_{C_S}, avg_data_{C_S})
```

where:

- `centroid_{C_S}` is the area-weighted centroid of sources in C_S
- `avg_data_{C_S}` is the area-weighted average of source features (normals,
  latent scalars/vectors)
- `total_strength_{C_S} = sum_{s in C_S} strength_s` is the sum of learned
  per-source strengths

The same neural network evaluates both exact and approximate interactions -
cluster centroids are treated as "virtual sources" with averaged features.
This is a zeroth-order (monopole) Taylor expansion of the kernel about the
cluster centroid.

### Dual-tree extension: node-to-node evaluation

The dual-tree variant goes further.  When a cluster C_T of *targets* is
well-separated from a cluster C_S of *sources*, the kernel varies slowly
across all targets in C_T (the target cluster is small relative to the
inter-cluster distance).  The kernel is evaluated **once** at the pair of
centroids `(centroid_{C_T}, centroid_{C_S})` and the result is broadcast to
all individual targets in C_T via scatter-add.

This reduces far-field evaluations from O(N_target * #source_nodes) to
O(#node_pairs), which is typically O(N) for well-separated geometries.

### Why area-weighting, not strength-weighting?

The spatial averages (centroid, feature means) use *area*-weighting, while the
multiplicative strength factor is summed separately.  Areas are fixed
geometric properties of the mesh (always positive, always stable), making the
aggregates reusable across kernel branches (the `MultiscaleKernel` has
multiple branches sharing the same source geometry).  Strengths, by contrast,
are learned per-source and per-branch values that change between communication
layers.  Separating these concerns means:

1. Aggregates are computed once per forward pass and shared across branches.
2. Only strength summation is per-branch (cheap O(N) work).
3. The aggregation is numerically stable (no division by near-zero when
   learned strengths cancel within a cluster).

---

## 3. Spatial Data Structure: ClusterTree

### 3.1 Construction via LBVH

The tree is built using a Linear Bounding Volume Hierarchy (LBVH) algorithm
(Karras 2012), the same approach used in PhysicsNeMo Mesh's existing `BVH`
class for mesh spatial decomposition:

1. **Morton codes**: Each point is assigned a 63-bit Morton code that
   interleaves the quantized coordinates.  Morton codes produce a
   space-filling Z-curve ordering that preserves spatial locality - nearby
   points in space tend to have nearby codes.

2. **Sort**: Points are sorted by Morton code.  After sorting, spatially
   nearby points are contiguous in the array.

3. **Top-down recursive splitting**: Starting from the full sorted range as
   the root, each segment with more than `leaf_size` points is split at its
   midpoint.  Because Morton-sorted order preserves spatial locality, midpoint
   splitting approximates a spatial median split, producing a balanced binary
   tree.  Each iteration processes all segments at the current depth in
   parallel, yielding O(log N) Python-level iterations.

4. **Bottom-up axis-aligned bounding box (AABB) propagation**: Leaf AABBs are
   computed from the actual points they contain.  Internal node AABBs are the
   union of their children's AABBs.  Total areas are similarly propagated
   (sum, not average).

The tree is stored as flat tensor arrays (`node_aabb_min`, `node_aabb_max`,
`node_left_child`, etc.) indexed by node ID, making it fully GPU-compatible.

### 3.2 Node Pre-allocation Bounds

Before construction, arrays are pre-allocated at the worst-case node count.
The midpoint split guarantees each child gets at least `floor(parent_size/2)`
sources, so the minimum leaf occupancy is `ceil(leaf_size/2)`.  The maximum
number of leaves is `ceil(N / min_per_leaf)`, and by the full-binary-tree
identity (`n_internal = n_leaves - 1`), the maximum total node count is
`2 * max_leaves - 1`.  After construction, the arrays are trimmed to the
actual count.

### 3.3 Source Aggregates

Per-node aggregate data is computed bottom-up for far-field evaluation:

- **Centroid**: area-weighted mean of source positions
- **Source features** (normals, latent scalars/vectors): area-weighted mean
  via `TensorDict.apply()` with segmented scatter operations
- **Total area**: sum (not average) of children's areas

Internal node aggregates are computed from their children's aggregates using
area-weighted averaging via a BFS level-ordering: internal nodes are
discovered by depth, then processed deepest-first so children are correct
before their parents read from them.

Aggregates depend on the source data (which changes between communication
layers as latent features are updated) but NOT on the tree structure (which
depends only on geometry).  The tree is built once per forward pass; aggregates
are recomputed each time the source data changes.

---

## 4. Dual-Tree Traversal

The classical Barnes-Hut algorithm pairs each *individual* target point with
tree nodes, yielding O(N_tgt * log N_src) far-field evaluations. The
dual-tree variant builds trees for **both** sources and targets, then
traverses pairs of nodes from the two trees simultaneously. This produces
far-field node-to-node pairs whose count can be as low as O(N).

### 4.1 Acceptance Criterion

The dual-tree acceptance criterion generalizes the single-tree Barnes-Hut
opening test by accounting for the spatial extent of *both* nodes:

```text
(D_T + D_S) / r  <  theta
```

where D_T and D_S are the AABB diagonals of the target and source nodes, and
r is the minimum distance between the two AABBs (gap distance).  In code:

```python
# Per-dimension gap between AABBs (0 where they overlap)
gap = torch.clamp(
    torch.maximum(aabb_min_T - aabb_max_S, aabb_min_S - aabb_max_T),
    min=0,
)
min_dist_sq = gap.pow(2).sum(dim=-1)

combined_diam_sq = (diam_T + diam_S).pow(2)

is_far = min_dist_sq * theta_sq > combined_diam_sq
```

The combined-diameter criterion is more conservative than the single-tree test
(which effectively sets D_T = 0).  This is appropriate because the far-field
broadcast approximation assumes the kernel is roughly constant across the
*target* node as well - an assumption that degrades when the target node is
large relative to the inter-node distance.

When both AABBs overlap (`gap = 0` in some dimension, `min_dist_sq = 0`),
the criterion always fails, forcing refinement.  This eliminates edge cases
where a node's centroid might be close to the cluster boundary.

### 4.2 Theta Parameter Semantics

The `theta` parameter follows the standard Barnes-Hut convention (Barnes &
Hut 1986):

- **Larger theta** = more aggressive (more approximations, faster).
- **Smaller theta** = more conservative (more exact interactions, slower).
- **theta = 0** = all interactions are exact (no approximation).

Typical values for GLOBE: `theta = 0.5` (conservative) to `theta = 1.5`
(aggressive).  The default is `theta = 1.0`.

### 4.3 Breadth-First Traversal

The traversal processes all active (target_node, source_node) pairs at each
level simultaneously:

1. **Initialize**: the single pair `(root_T, root_S)`.
2. **For each iteration** (bounded by `depth_T + depth_S + 1`), classify
   active pairs into three categories:

   - **Far-field**: passes the acceptance criterion.  Record the
     `(target_node, source_node)` pair.
   - **Near-field leaves**: fails the criterion, and BOTH nodes are leaves.
     Expand into the Cartesian product of individual targets and sources
     within those leaves.
   - **Needs refinement**: fails the criterion, and at least one node is
     internal.  Split into child pairs for the next iteration.

3. **Splitting rule**: Split the node with the larger AABB diameter (by
   squared diagonal).  If both nodes have equal diameter, split both.  If one
   side is a leaf, only the other can be split.

   - Split target only: 2 child pairs `(left_T, S)` and `(right_T, S)`.
   - Split source only: 2 child pairs `(T, left_S)` and `(T, right_S)`.
   - Split both: 4 child pairs `(left_T, left_S)`, `(left_T, right_S)`,
     `(right_T, left_S)`, `(right_T, right_S)`.

4. **Post-processing**: Near pairs are sorted by source index, far pairs by
   source node, for cache-friendly memory access during kernel evaluation.

The output is a `DualInteractionPlan` containing four index arrays:

- `(near_target_ids, near_source_ids)`: individual target-source pairs
  requiring exact evaluation.
- `(far_target_node_ids, far_source_node_ids)`: node-to-node pairs using the
  monopole approximation with target-side broadcast.

### 4.4 Self-Interaction and Cross-BC Interaction

For communication layers with a single BC type (or the self-interaction
portion of a multi-BC model), the same `ClusterTree` is used for both the
source and target sides.  The traversal starts with `(root, root)` and
proceeds normally.  The splitting rule defaults to splitting both nodes when
diameters are equal, which is always the case for self-interaction (both sides
reference the same tree).

When multiple BC types are present, communication layers also evaluate
cross-BC interactions: source BC "A" contributes to destination BC "B" and
vice versa.  For cross-BC pairs, the source tree and target tree are different
objects (built from different point sets), and a separate
`DualInteractionPlan` is computed for each (source BC, destination BC) pair.
This produces B^2 plans for B BC types.  Since B is small in practice (1-4),
the additional traversal cost is negligible.

### 4.5 Caching Interaction Plans

The interaction plan depends only on the geometric positions of sources and
targets, not on the source data or strengths.  For communication hyperlayers,
all B^2 plans (covering both self-interaction and cross-BC pairs) are computed
once and reused across all layers.  For the final prediction evaluation,
separate plans are computed from each source BC tree to the prediction-point
target tree.  This eliminates redundant traversals.

---

## 5. Two-Phase Kernel Evaluation

`BarnesHutKernel.forward()` evaluates near-field and far-field interactions in
two distinct phases, each with its own gather-evaluate-scatter pipeline.  The
same `_evaluate_interactions()` method handles both - it operates on generic
`(N_pairs, ...)` tensors and is agnostic to whether the pairs are individual
points or node centroids.

### 5.1 Phase A: Near-Field (Individual Pairs)

Near-field pairs are individual (target, source) interactions requiring exact
kernel evaluation.  They are processed in chunks:

1. **Chunk the pair arrays**: Slice `near_target_ids[start:end]` and
   `near_source_ids[start:end]`.
2. **Gather**: Index into the shared source/target point arrays and feature
   data to build per-chunk float tensors.
3. **Evaluate**: Run `_evaluate_interactions()` (feature engineering + MLP +
   post-processing).
4. **Weight and scatter**: Multiply by per-source strengths, then
   `scatter_add` into the output buffer at the target indices.

### 5.2 Phase B: Far-Field (Node Pairs with Broadcast)

Far-field pairs are node-to-node interactions that exploit the monopole
approximation with target-side broadcast:

1. **Gather**: Index into node centroids and aggregate features for both
   the target nodes (`far_target_node_ids`) and source nodes
   (`far_source_node_ids`).
2. **Evaluate**: Run `_evaluate_interactions()` at the centroid pair, yielding
   one result per node pair.
3. **Weight**: Multiply by total source-node strength.
4. **Broadcast to targets**: For each target node, use `_ragged_arange` to
   expand the node-level result to all individual targets within that node,
   then `scatter_add` to the output buffer.

The broadcast step uses the target tree's `node_range_start` and
`node_range_count` arrays to find which individual targets belong to each
target node, and `sorted_source_order` to map back to original target indices.

### 5.3 The _evaluate_interactions() Factoring

The core feature engineering pipeline (vector magnitudes, spherical harmonics,
network evaluation, far-field decay, vector reprojection) lives in a shared
`_evaluate_interactions()` method.  This method operates on generic
`(*interaction_dims, ...)` tensors - it does not know or care whether the
interactions are dense `(N_tgt, N_src)` or sparse `(N_chunk,)`.

- `Kernel.forward()` calls it with `interaction_dims = (N_tgt, N_src)` (dense,
  brute-force evaluation)
- `BarnesHutKernel.forward()` calls it with `interaction_dims = (N_chunk,)` in
  both Phase A and Phase B

This avoids duplicating the ~250-line feature engineering pipeline.

---

## 6. Memory Management

### 6.1 Gather-Inside-Checkpoint Pattern

The key memory optimization: each chunk's gather and evaluate steps are
wrapped together in a single `torch.utils.checkpoint.checkpoint` call.  The
checkpoint boundary is drawn so that autograd saves only the compact int64
index arrays (~8 bytes/pair) and references to the shared source data (O(1)),
rather than the gathered float data (~300 bytes/pair).  This is a ~37x
reduction in checkpoint-saved memory per chunk.

### 6.2 Auto-Chunk Sizing

`_auto_chunk_size()` estimates peak memory per interaction pair from the
kernel's feature engineering pipeline (counting intermediate floats for
spatial vectors, scalar features, MLP layers, and post-processing) and sizes
chunks to fit within ~50% of free GPU memory.  During training, a 5x
multiplier accounts for autograd tensor retention; during inference, this
multiplier is dropped, allowing larger chunks.

### 6.3 Branch-Level Checkpointing

`MultiscaleKernel` wraps each `BarnesHutKernel` branch call in
`checkpoint(use_reentrant=False)`.  This ensures only ONE branch's autograd
graph exists at a time during backward, preventing autograd memory from
accumulating across all branches.  Combined with the gather-inside-checkpoint
pattern, peak autograd memory scales as O(chunk_size \* indices_only) rather
than O(n_branches \* n_pairs \* features).

The branch-level and chunk-level checkpoints nest correctly:
`use_reentrant=False` composes via `saved_tensors_hooks`.

### 6.4 Chunk-Size Determinism

`_auto_chunk_size()` queries free GPU memory (`torch.cuda.mem_get_info`),
which changes between the forward pass and a checkpoint replay during
backward.  If the chunk size changes, intermediate tensors have different
shapes, and the outer (branch-level) checkpoint raises `CheckpointError`.

To prevent this, `MultiscaleKernel.forward()` precomputes each branch's chunk
size **outside** the checkpoint boundary and passes it as a fixed input via
the `near_chunk_size` kwarg.  The checkpoint saves this value as an input and
replays with the identical value, regardless of current GPU memory state.

---

## 7. Integration with GLOBE

### 7.1 Tree and Plan Lifecycle

Within a single `GLOBE.forward()` call:

1. **Phase 1 (init)**: Build one `ClusterTree` per boundary condition type
   from the cell centroids.  Compute `DualInteractionPlan`s for communication
   covering all (source BC, destination BC) pairs - B^2 plans for B BC types.
   For self-interaction pairs (source == destination), the target tree is the
   same object as the source tree.  All trees and plans are cached for the
   duration of the forward pass.

2. **Phase 2 (communication)**: For each communication hyperlayer, reuse the
   cached trees and plans.  Only source aggregates are recomputed (the latent
   features change between layers).

3. **Phase 3 (prediction)**: Build a single target tree for prediction points
   and compute one interaction plan per source BC type (B plans total).
   Source trees are reused from Phase 1.

Tree construction and plan finding are decorated with
`@torch.compiler.disable` because they involve irregular control flow (Morton
code bit operations, data-dependent loop termination) that `torch.compile`
cannot trace.  The kernel evaluation inside `_evaluate_interactions` compiles
normally.

### 7.2 Shared Aggregates Across Branches

`MultiscaleKernel` computes source aggregates once and passes them to all
`BarnesHutKernel` branches via the `source_aggregates` parameter.  Since
aggregates depend only on geometry and source data (both shared across
branches), this eliminates redundant computation.  Only per-node strength
summation (which depends on per-branch strengths) is computed per-branch.

### 7.3 Dynamic Shapes

The hierarchical approach naturally requires dynamic tensor shapes (each mesh
produces a different tree, different interaction plan, different pair counts).
Training scripts use `torch.compile(dynamic=True)` and
`compile_mode="max-autotune-no-cudagraphs"` to accommodate this.  Mesh padding
(previously used for static-shape CUDA graph compatibility) has been removed.

---

## 8. Parameter Tuning

### 8.1 Theta (opening angle)

The `theta` parameter controls accuracy vs. speed:

| theta | Character            | Typical use case                       |
|-------|----------------------|----------------------------------------|
| 0     | Exact                | No approximation (equivalent to dense) |
| 0.5   | Conservative         | High accuracy, for validation          |
| 1.0   | Moderate             | Good default for production training   |
| 1.5   | Aggressive           | Fast approximate evaluation            |
| 100+  | Extremely aggressive | Testing only                           |

The approximation error per interaction scales with theta, but the total
error is bounded by the kernel's far-field decay.  Distant clusters contribute
little regardless of approximation quality, providing a natural error ceiling.

### 8.2 Leaf Size

The `leaf_size` parameter (default 1) controls tree granularity:

- **Smaller leaf_size** (e.g., 1-4): deeper trees, finer-grained near/far
  classification, more far-field approximations at higher precision (each
  node represents a smaller spatial region, so centroids are more accurate).
  Near-field count drops dramatically since the opening criterion passes more
  easily for small-diameter nodes.
- **Larger leaf_size** (e.g., 32-64): shallower trees, coarser
  classification, fewer traversal iterations, but each near-field leaf-pair
  hit expands into up to `leaf_size^2` individual interactions, and far-field
  node centroids are coarser averages over larger spatial regions.

Crucially, **smaller leaf_size does not reduce accuracy** for a fixed theta.
The far-field approximation for a single-point leaf (leaf_size=1) is exact in
the source coordinate (the "centroid" is the point itself), so all
approximation error comes from the target side, which is controlled by theta.
Smaller leaves produce strictly finer-resolution far-field evaluations.

Benchmarks on DrivAerML (20k boundary faces, H100) show `leaf_size=1` is
3.8x faster than `leaf_size=32` with no accuracy penalty.  The default is
`leaf_size=1`.

---

## 9. Complexity Analysis

| Component          | Time complexity     | Memory complexity   |
|--------------------|---------------------|---------------------|
| Tree construction  | O(N log N)          | O(N)                |
| Aggregate computation | O(N)             | O(N)                |
| Dual-tree traversal | O(N log N)         | O(N log N)          |
| Near-field evaluation | O(N log N)       | O(chunk_size)       |
| Far-field evaluation | O(N)              | O(N_far_pairs)      |
| Far-field broadcast | O(N log N)         | O(N_targets)        |
| **Total**          | **O(N log N)**      | **O(N log N)**      |

The far-field evaluation step is O(N) rather than O(N log N) because the
number of well-separated node pairs grows linearly for typical point
distributions.  This is a concrete improvement over single-tree Barnes-Hut,
where each target individually evaluates against O(log N) source nodes.

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

## 10. Architecture Summary

```text
GLOBE.forward()
  |
  +-- _build_trees_and_plans()              [outside torch.compile]
  |     Build ClusterTree per BC type (B trees)
  |     Find DualInteractionPlan for all (src, dst) BC pairs (B^2 plans)
  |
  +-- Phase 2: Communication hyperlayers (repeat n_comm times)
  |     |
  |     +-- _evaluate_hyperlayer()
  |           |
  |           +-- MultiscaleKernel.forward()
  |                 |
  |                 +-- compute_source_aggregates()  [once, shared across branches]
  |                 +-- precompute near_chunk_sizes  [outside checkpoint boundary]
  |                 |
  |                 +-- for each branch:
  |                       checkpoint(BarnesHutKernel.forward(), ...)
  |                       |
  |                       +-- _compute_node_strengths()
  |                       |
  |                       +-- Phase A: Near-field
  |                       |     for each chunk:
  |                       |       checkpoint(_gather_and_evaluate())
  |                       |       weight by strength, scatter_add to output
  |                       |
  |                       +-- Phase B: Far-field
  |                             checkpoint(_gather_and_evaluate())  [node centroids]
  |                             weight by node strength
  |                             broadcast to individual targets via scatter_add
  |
  +-- _build_prediction_plans()             [outside torch.compile]
  |     Build target tree for prediction points
  |     Find DualInteractionPlan (pred: different target points)
  |
  +-- Phase 3: Final evaluation
        (same structure as communication, different target points)
```

---

## 11. Testing Strategy

The implementation is validated through several complementary test categories:

- **Convergence to exact**: As theta decreases toward 0, `BarnesHutKernel`
  output converges monotonically to the exact `Kernel` output.  At
  theta = 0.01, the two agree within floating-point tolerance.  Tested
  across all combinations of 2D/3D, scalar/vector outputs, and
  scalar/vector source features.

- **Source coverage invariant**: For every target, the union of near-field
  sources and far-field node subtrees equals the complete source set
  `{0, ..., N-1}` with no duplicates and no omissions.  This is the
  fundamental correctness property of the dual-tree traversal.

- **Gradient correctness**: Gradients through `BarnesHutKernel` match exact
  `Kernel` gradients at high theta, verifying that the non-differentiable
  traversal decisions do not corrupt gradient flow through the differentiable
  kernel evaluation.

- **Equivariance preservation**: Translation, rotation, and source-permutation
  equivariance are preserved by the hierarchical approximation, verified at
  both moderate and high theta.

- **Nested key structure**: Tests with deeply nested TensorDict keys matching
  GLOBE's actual production data format (physical/latent/strength namespaces).

---

## 12. References

- Barnes & Hut (1986). "A hierarchical O(N log N) force-calculation algorithm."
  *Nature* 324, 446-449.
- Appel (1985). "An Efficient Program for Many-Body Simulation." *SIAM J. Sci.
  Stat. Comput.* 6(1), 85-103.  Early dual-tree variant of the Barnes-Hut idea.
- Gray & Moore (2001). "'N-Body' Problems in Statistical Learning."
  *NIPS 2001*.  Formalized dual-tree algorithms with generalized acceptance
  criteria.
- Karras (2012). "Maximizing Parallelism in the Construction of BVHs, Octrees,
  and k-d Trees." *HPG 2012*.  The LBVH construction algorithm used here.
- Burtscher & Pingali (2011). "An Efficient CUDA Implementation of the
  Tree-Based Barnes Hut n-Body Algorithm." *GPU Computing Gems Emerald Edition*.
- Lukat & Banerjee (2015). "A GPU accelerated Barnes-Hut tree code for FLASH4."
  Describes AABB-distance opening criterion.
- Madan et al. (2025). "Stochastic Barnes-Hut Approximation of Kernel Matrices."
  *SIGGRAPH 2025*.  Uses a `beta = 1/theta` convention (inverted relative to
  the original Barnes & Hut convention used in this codebase).
