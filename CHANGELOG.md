<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2026-XX-YY

### Added

- Adds `zenith_azimuth_angles` and `zenith_azimuth_angles_from_timestamp` to
  `physicsnemo.utils.zenith_angle`, returning
  `(sin_zenith, cos_zenith, sin_azimuth, cos_azimuth)` alongside the existing
  `cos_zenith_angle` / `cos_zenith_angle_from_timestamp` helpers. The azimuth
  follows the north-clockwise convention.
- Adds `physicsnemo.nn.shrink_and_perturb_`, an in-place shrink-and-perturb
  weight re-initialization for warm-starting from pretrained weights.
- Adds dimension-generic volume mesh generation for implicit domains to
  `physicsnemo.mesh.generate`. `mesh_implicit_domain` meshes
  `{x : phi(x) < 0}`, clipped to the bounding box (box faces are honored
  as a boundary, so that external-flow "box minus obstacle" domains work
  directly), for any implicit function (signed-distance functions, level
  sets, or neural fields).
- Adds `integrate_moment` and `Mesh.integrate_moment` for measure-weighted
  outer-product moments. Mesh integration APIs now accept `nan_policy`.
- Adds per-cell measure weights that are preserved through cell subsampling
  and consumed by mesh integration routines and GLOBE.
- Adds a `global_shape` argument to `ShardTensor.from_local`, enabling the
  no-communication `sharding_shapes="chunk"` path.
- Adds exact-boundary quality mesh generation to
  `physicsnemo.mesh.tessellation`: `fill_interior` takes a closed
  codimension-one boundary `Mesh` (2D edge loops today; loops in any order
  and orientation, with holes, multiple components, and nested islands
  resolved automatically) and fills the interior with quality simplices via
  constrained Delaunay triangulation with Ruppert refinement — every input
  vertex is preserved bit-identically, every output triangle meets the
  guaranteed minimum-angle bound, output is a `Mesh` with provenance
  `point_data`, deterministic, with optional bound-preserving ODT smoothing.
  The contract is dimension-generic; `n = 3` raises `NotImplementedError`
  pending exact boundary recovery. Also adds `polygon_interior_point`,
  which returns a point strictly inside a simple polygon.
- Adds `rectilinear_grid_divergence`, `rectilinear_grid_curl`, and
  `rectilinear_grid_laplacian` to `physicsnemo.nn.functional`, with Torch and
  fused Warp implementations for periodic, nonuniform rectilinear grids.
- Adds uniform triangle-surface remeshing with NVIDIA Warp on CPU and CUDA,
  including `remesh`, `Mesh.remesh`, topology cleanup, and advanced
  tensor-level controls for runtime tuning.
- Adds coverage reporting on PRs — an informational `Coverage %` check plus a
  ready-to-enable Codecov integration.
- Adds differentiable mesh morphing: Torch-backed dense ``displace_points`` /
  ``Mesh.displace`` and Torch/NVIDIA Warp compact sparse-control
  ``morph_points`` / ``Mesh.morph`` / ``DomainMesh.morph``.
- Adds thin-plate-spline radial-basis deformation through
  `radial_basis_function_deform_points`,
  `Mesh.radial_basis_function_deform`, and
  `DomainMesh.radial_basis_function_deform`. PyTorch performs the differentiable
  coefficient solve. Torch and fused NVIDIA Warp backends evaluate the field.
- Adds differentiable lattice free-form deformation with Torch and NVIDIA Warp
  backends: dimension-generic ``free_form_deform_points`` /
  ``Mesh.free_form_deform`` / ``DomainMesh.free_form_deform`` with Bernstein
  (classic FFD) and locally supported uniform cubic B-spline bases, plus
  node-interpolating `linear`, `cubic_hermite`, and `quintic_hermite` modes.
- Adds `uniform_grid_divergence`, `uniform_grid_curl`, and
  `uniform_grid_laplacian` to `physicsnemo.nn.functional`, with Torch and fused
  Warp implementations for periodic Cartesian grids.
- Adds the experimental Strata weather-emulation models —
  `physicsnemo.experimental.models.strata.Strata` and `StrataTransformer3D` — plus
  the continuous / stereographic RoPE helpers `build_rope_cos_sin_1d_continuous`,
  `build_axial_rope_cos_sin_2d_continuous`, `stereographic_projection`, and
  `spherical_centroid` in `physicsnemo.experimental.nn`.
- Adds Point-Transformer local vector-attention blocks to `physicsnemo.nn`.
- Adds an `is_causal` option to `TimmSelfAttention` in `physicsnemo.nn` for
  causal self-attention.
- FSDP2 checkpoint support: full save/load round-trip for
  ``torch.distributed.fsdp`` v2 models, including DTensor edge cases,
  cross-mesh reloads, and optimizer state loading.
- Migrated the StormCast example from DDP + Domain Parallel  to FSDP2 +
  Domain Parallel. StormCast previously used ``FullyShardedDataParallel``
  with ``ShardingStrategy.NO_SHARD`` (equivalent to DDP) alongside domain
  parallelism; it now uses the FSDP2 ``fully_shard`` API, producing 2D-mesh
  DTensor parameters when ``use_shard_tensor`` is enabled.
- Adds tensor-returning `Mesh.gradient`, `Mesh.divergence`, `Mesh.curl`, and
  `Mesh.laplacian` convenience methods to `physicsnemo.mesh`, mirroring
  `Mesh.integrate` (each returns a tensor and accepts a data key or a raw
  tensor, with a `data_source="points"|"cells"` kwarg selecting vertex or
  cell-centered fields). This gives the discrete differential operators a
  consistent, discoverable surface on `Mesh`; previously divergence/curl/
  laplacian were reachable only as free functions in `physicsnemo.mesh.calculus`.
  Adds `compute_divergence_cells_lsq` and `compute_curl_cells_lsq` free
  functions (cell-centered LSQ analogues); DEC operators and the cotangent
  Laplacian remain vertex-only and raise `NotImplementedError` for cell data.
- Adds `farthest_point_sampling` to `physicsnemo.nn.functional`, a greedy
  farthest-point sampling (FPS) functional for point clouds.
- Adds `FourierPositionalEmbedding` to `physicsnemo.nn`, a deterministic
  axis-wise (NeRF-style) Fourier positional embedding for continuous
  coordinates with no learnable parameters.
- Adds radiation transport example (`examples/nuclear_engineering/radiation_transport`)
- Adds agent skills structure, and initial skill for 'discoverability'.
- Adds the experimental AeroJEPA model
  (`physicsnemo.experimental.models.aerojepa.AeroJEPA`), a joint-embedding
  predictive architecture for 3D aerodynamic fields composing context and
  target encoders, a query-token field decoder, and a JEPA predictor head,
  together with its SIGReg, token-latent, and reconstruction loss family.
  The generic point-cloud tokenizer and the batch/mask/k-NN helpers are added
  under `physicsnemo.experimental.nn`; the local point-transformer attention
  blocks it composes come from `physicsnemo.nn`.
- Adds the AeroJEPA SuperWing tutorial recipe
  (`examples/cfd/external_aerodynamics/aerojepa`), an end-to-end Hydra-driven
  workflow covering dataset download, normalization, JEPA training, chunked
  inference, field-error plots, and CL/CD post-processing.
- Adds xDeepONet to experimental models
  (`physicsnemo.experimental.models.xdeeponet.DeepONet`).  A single
  dimension-generic (2D/3D) DeepONet that accepts a spatial or MLP branch,
  an optional trunk, and an optional second branch as `nn.Module` inputs
  (dependency injection).  Six forward-call conventions cover trunked,
  trunkless, packed/auto-padded, and xFNO-style time-axis-extend modes.
  Supports multi-channel output, multiple decoder types (MLP, Conv,
  temporal projection), composable Fourier / UNet / Conv spatial branches
  (`SpatialBranch`), and coordinate features.
- Adds `FNO4DWrapper` to the xdeeponet package: a thin wrapper around the
  library `physicsnemo.models.fno.FNO` (`dimension=4`) that adds
  autoregressive time-axis extension over `(B, X, Y, Z, T, C)` inputs (predict
  a `K`-step forecast horizon via `target_times`).  Use
  `physicsnemo.models.fno.FNO(dimension=4)` directly when the time-axis
  extension is not needed.  3D FNO / Conv-FNO / U-FNO operators are expressed
  as `DeepONet(trunk=None, dimension=3)` with a Fourier/UNet/Conv
  `SpatialBranch`.
- Adds `Sin` elementwise sine activation to `physicsnemo.nn`, registered
  in `ACT2FN` so it can be looked up by name (`get_activation("sin")`).
- Adds active-learning recipe for external-aerodynamics surrogates
  (`examples/cfd/external_aerodynamics/active_learning_aero/`). Iteratively
  fine-tunes a GP-augmented GeoTransolver onto an out-of-distribution
  target class by scoring unlabeled candidates with a joint UQ signal
  (GP-vs-integrated-drag disagreement + GP posterior std) and selecting
  the top-`k` per round. Built on the `physicsnemo.active_learning`
  protocols and `physicsnemo.experimental.uq.VariationalGPHead`, with a
  layered structure (generic AL driver / GP-UQ recipe / aero adapter)
  designed for reuse on other UQ-based regression problems.
- Adds `LatentNoveltyQueryStrategy` to the active-learning aero recipe,
  a third acquisition strategy that ranks unlabeled samples by their
  average kNN cosine distance in the encoder's learned geometry latent
  — reusing the same `OODGuard`
  (`physicsnemo.experimental.guardrails.embedded`) that flags
  out-of-distribution inputs at inference time. The guard is calibrated
  on the currently labeled set each round; round 1 falls back to
  class-balanced random because the calibration buffer is empty. New
  public `OODGuard.score_geometry()` method exposes the raw per-sample
  geometry-latent kNN distance as a continuous score for downstream
  consumers (e.g. AL acquisition) without the boolean thresholding /
  warning emission of `OODGuard.check()`.
- Adds rotary position embedding (RoPE) modules to `phyiscsnemo.nn` and
  integrates support for 2D RoPE in the neighborhood attention backend
  of `DiT` layers.
- Adds support for RoPE, dynamic invalid-region masking, and a new
  `ConvDetokenizer` in `phyiscsnemo.models.DiT`. Invalid regions are supplied
  per forward call via the `invalid_mask` argument of `DiT.forward` (a
  per-sample, batch-variable pixel mask, domain-parallel safe), replacing
  flagged tokens with a learned mask token.
- Adds an inference script (`src/infer.py` + `conf/infer.yaml`) to the
  Unified External Aero Recipe
  (`examples/cfd/external_aerodynamics/unified_external_aero_recipe`),
  with integrated aerodynamic force/moment coefficients (`src/forces.py`:
  CD/CL/CS/CMR/CMP/CMY). The script is model/dataset-agnostic, writes one
  native `.pdmsh` `DomainMesh` per sample (carrying physical-unit
  `pred_<field>` / `true_<field>`), reports training-space metrics
  (matching the training/validation loop), and reuses the trainer's
  dataloader / collate / metric tooling (refactored into `datasets.py`
  and `utils.py`).
- Adds a mesh-native signed distance field to `physicsnemo.mesh.spatial`
  (`physicsnemo.mesh.spatial.signed_distance_field`), built on the `BVH`
  and `ClusterTree` spatial structures it lives alongside. Returns a
  `SignedDistanceFieldResult` named tuple: the signed distance, the closest
  surface point, and the nearest-face index per query.
  The nearest-triangle query runs as a single-kernel per-thread BVH traversal
  (Triton on CUDA, a bounded-stack PyTorch DFS as the CPU reference; per-query
  indices are int64 so query counts past tens of millions do not overflow). The
  sign is computed either from the angle-weighted pseudo-normal of the closest
  mesh feature — face, edge, or vertex, which stays correct at sharp/non-convex
  edges where a single face normal flips the sign — or, with
  `use_sign_winding_number=True`, from
  a `ClusterTree` dual-tree Barnes-Hut generalized-winding-number summation that
  runs identically on CPU and GPU (robust on non-watertight meshes). The private
  datapipes implementation (`physicsnemo.datapipes.transforms._sdf_torch` /
  `_sdf_triton`, including its bespoke Triton winding kernel) is superseded and
  removed; the public datapipes SDF transform delegates here.
- Added an iterable style dataset to physicsnemo datapipes, for on-the-fly gpu simulations.
- DPS guidance now supports **non-uniform guidance strength**: the `std_y` and
  `gamma` arguments of `physicsnemo.diffusion.guidance.ModelConsistencyDPSGuidance`
  / `DataConsistencyDPSGuidance` and their
  `physicsnemo.diffusion.multi_diffusion` counterparts accept tensors as well as
  floats. A tensor assigns a different measurement-noise level / SDA scaling to
  each observation component, e.g. per-channel (`(1, C, 1, 1)`) or pointwise
  (full observation shape). Passing floats keeps the previous uniform
  behavior unchanged.
- Adds `relative_mse` and `relative_l2` (target-normalized regression errors,
  `relative_l2 = sqrt(relative_mse)`) to `physicsnemo.metrics.general`
  (`relative_error.py`), with optional element `weights` and `dim`-based
  reduction matching `general.mse`.
- `physicsnemo.metrics.general.mse` `mse`/`rmse` gain an optional `weights`
  argument for a masked/weighted mean (backward-compatible; `weights=None`
  reproduces the prior unweighted result).
- Adds a kinetic Monte Carlo (KMC) surrogate example
  (`examples/kinetic_monte_carlo`): a probabilistic autoregressive surrogate
  (`ParticleGeoTransolver`) that emulates a KMC event stream, predicting the
  next event (the new particle's features and inter-event delay) from the
  current particle population, an optional background mesh, and the simulation
  time. Independent rollouts form an ensemble for uncertainty quantification.

### Changed

- Optimizes the production container build by consolidating related filesystem
  operations, using BuildKit bind and cache mounts, and separating custom,
  declared, and project dependency installation. Reduces total physicsnemo layers
  by around 78%.
- `ShardTensor.redistribute` now computes receive shapes analytically when
  sharding shapes are known, skipping the shape-negotiation `all_to_all`
  collective (falls back to the collective only when shapes are unavailable).
- PhysicsNeMo-Mesh tensor-valued gradients now consistently use the documented
  derivative-first layout `(entity, spatial_dimension, *value_shape)` across
  LSQ, intrinsic LSQ, and DEC. Earlier LSQ releases returned
  `(entity, *value_shape, spatial_dimension)` instead; migrate a stored legacy
  gradient with `legacy_gradient.movedim(-1, 1)`. Divergence and curl values
  are unchanged.
- xDeepONet `SpatialBranch`
  (`physicsnemo.experimental.models.xdeeponet.SpatialBranch`) now supports
  mixed-precision (AMP/autocast) training: FFT-based spectral convolutions are
  evaluated in float32 internally (cuFFT lacks complex-half support) while the
  rest of the branch uses autocast. This is a no-op under full precision, so
  fp32 outputs are unchanged. Also fixes a stale module docstring that
  referenced removed trunk/MLP-branch builder helpers.
- `physicsnemo.mesh.remeshing.remesh` now raises `NotImplementedError` for
  non-2D-in-3D inputs (the remeshing implementation is surface-only) instead
  of failing confusingly downstream, and its docstring reflects that
  restriction.
- `physicsnemo.mesh.spatial`: `BVH.from_mesh` and `ClusterTree.from_points` now
  share a single morton-LBVH node-topology builder (`spatial/_lbvh.py`),
  removing ~80 lines of duplicated build logic; construction output is
  byte-identical. `BVH.from_mesh` now defaults to `leaf_size=1` (was 8),
  matching `ClusterTree.from_points` and measured to be more performant across
  platforms (smaller leaves yield fewer candidate cells per query). Containment /
  nearest-cell query results are unchanged. Adds the first direct unit tests for
  `ClusterTree` (construction invariants, aggregates, dual-tree cover).
- `physicsnemo.mesh` performance: eliminated host-device syncs on hot paths.
  Cached topological adjacencies now store the `Adjacency` object directly instead
  of reconstructing it (which re-ran its syncing `__post_init__` validation) on every
  lookup — making cached adjacency lookups ~120x faster on GPU (~335us → ~3us for a
  10k-point sphere); the BVH leaf-hit expansion drops two per-traversal-level syncs;
  and the Laplacian smoother reuses its per-iteration buffers in place instead of
  reallocating them.
- `physicsnemo.mesh.Mesh.slice_cells` now accepts `None`/`Ellipsis` (keep all
  cells, return self), matching its type hint and `slice_points`;
  `gaussian_curvature_cells` reuses the cached `gaussian_curvature_vertices`
  property instead of recomputing it.
- `physicsnemo.mesh`: `validate_mesh(check_self_intersection=True)` now raises
  `NotImplementedError` (the check is unimplemented) instead of silently returning a
  `None` sentinel that masquerades as "no self-intersections found".
- Performance improvements in the diffusion module: reduced peak memory of
  DPS-guided diffusion sampling most notably for multi-diffusion at large
  domains. A guided `sample()` loop run under `torch.no_grad()` now detaches the
  state between solver steps, so the guidance autograd graph is no longer
  accumulated across the sampling trajectory (sampled outputs are unchanged;
  use `torch.no_grad()`, not `torch.inference_mode()`). Also expands CI test
  coverage and adds an API documentation page for
  `physicsnemo.diffusion.multi_diffusion`.
- Performance improvements in IO prefetching and GPU preprocessing in physicsnemo datapipes.
- &#9888;&#65039; **BC-impact (DPS guidance):** a custom `norm` callback passed to
  `physicsnemo.diffusion.guidance.ModelConsistencyDPSGuidance` /
  `DataConsistencyDPSGuidance` (and their `physicsnemo.diffusion.multi_diffusion`
  counterparts) must now return an **elementwise** loss (same shape as its
  inputs) instead of a per-batch-element reduced scalar of shape `(B,)`.
  Migration: drop the reduction from your `norm`, e.g. return
  `(y_pred - y_true).abs().pow(2)` rather than
  `(y_pred - y_true).pow(2).reshape(B, -1).sum(dim=1)`. For
  `DataConsistencyDPSGuidance` (and its `multi_diffusion` counterpart) the
  `norm` callback now also receives the **unmasked** `(x_0, y)` and the mask is
  applied to its output (`mask * norm(x_0, y)`), where it previously received
  the pre-masked `(mask * x_0, mask * y)`; the two agree for the built-in `Lp`
  norms, but a custom `norm` that relies on unobserved entries being zeroed
  before the call may differ. The integer `norm` selector (e.g. `norm=2`) is
  unaffected.

### Deprecated

- `physicsnemo.mesh.calculus.integrate_cell_data` and `integrate_point_data`
  are deprecated in favor of `integrate(..., data_source="cells"|"points")`.
  Compatibility wrappers remain available for this release and emit
  `LegacyFeatureWarning`.

### Removed

### Fixed

- Multinomial index sampling now uses one shared `weighted_multinomial`
  functional across datapipes, DoMINO, and remeshing. Its core API follows
  `torch.multinomial`, adds allocation-free integer input for uniform
  populations, and supports sampling with or without replacement. Exact
  sampling without replacement uses `torch.randperm` or a chunked exponential
  race, with an explicit low-memory Poisson-gap approximation for uniform
  sampling. This removes the `torch.multinomial` `2^24` category limit for
  sampling without replacement, consolidates duplicated Poisson index
  samplers, and fixes incorrect chunk-local indices and biased per-chunk quotas
  in DoMINO.
- Unified external aerodynamics recipe: the aggregate metrics reported for
  vector fields under the bare field name (e.g. `wss_l2`, likewise `_l1` /
  `_mae`) were computed on per-point vector magnitudes, so direction errors
  were invisible — a prediction with the correct magnitude but wrong direction
  at every point scored 0. The bare-name aggregate is now computed over all
  components jointly (whole-field relative norms, Frobenius for `l2`). The
  broken magnitude-only aggregate is not retained under a separate key.
  Per-component metrics (`wss_x_l2`, ...) were always direction-sensitive and
  are unchanged, and training/checkpoints are unaffected (the training
  objective goes through `LossCalculator`, not this metric path) — only
  reported aggregate vector metrics were misleading.
- Unified external aerodynamics recipe: model templates can now carry
  known-good training overrides (`train.yaml`'s `_self_` merges before the
  model template; all existing templates resolve identically). The GLOBE
  example now uses the recipe's default `compile` and learning-rate
  settings — both measured equivalent or better than the previously
  documented overrides on the DrivAerML surface case — and no longer sets
  `training.field_weights={pressure: 1.0, wss: 100.0}`, which was redundant
  with `NormalizeMeshFields` normalization and starved the pressure field
  of gradient signal (~2x worse converged pressure L2 at equal WSS L2).
- Datapipe contiguous-block subsampling now wraps cyclically, giving boundary
  and interior elements equal inclusion probability.
- Cell-subsampled GLOBE inputs now retain their effective integration measure,
  preventing area-weighted outputs and gradients from collapsing.
- `physicsnemo.mesh.io.from_pyvista(..., force_copy=True)` now copies attached
  point, cell, and global data as well as geometry. The matching new
  `to_pyvista(..., force_copy=True)` option prevents exported PyVista geometry
  and data from mutating the source `Mesh` through shared CPU storage.
- `physicsnemo.mesh.sampling.sample_data_at_points` now handles integer and
  boolean fields by returning `float64`, so NaN sentinels and non-integral
  interpolation or multi-cell means are representable (subject to the usual
  `float64` precision limits). Point-data interpolation now promotes field and
  geometry dtypes consistently, and accumulation uses fewer full-sized
  temporaries and CUDA host synchronizations.
- `physicsnemo.mesh.projections.extrude` now produces a *conforming* (crack-free)
  simplicial complex for multi-cell inputs. Each prism was previously tessellated
  using the per-cell local vertex order, so adjacent cells that listed a shared
  edge's endpoints in different orders split the shared quad face along opposite
  diagonals; the resulting non-manifold volume leaked interior crack faces into
  `get_boundary_mesh` (boundary edges shared by 4 faces — e.g. an extruded L-shape
  or any multi-column grid, which also broke `repair.fix_orientation`). Parent-cell
  vertices are now sorted into a global order before tessellation (the
  Freudenthal-Kuhn subdivision), a no-op for already-sorted inputs.
- `physicsnemo.mesh.generate.marching_cubes` now accepts `bfloat16` fields by
  converting them to `float32` before crossing the NumPy boundary.
- `physicsnemo.mesh.projections.extrude` now returns consistently oriented cells
  for full-dimensional (codimension-0) output.
- `physicsnemo.mesh.remeshing.remesh` now preserves the input mesh's device and
  floating dtype instead of dropping them to CPU/float32.
- `physicsnemo.mesh.io.to_pyvista` now preserves supported dtypes for attached
  point, cell, and global data instead of narrowing every array to `float32`.
  Reduced-precision floating-point values are promoted only as needed for VTK.
- `physicsnemo.mesh.io.from_pyvista` and `to_pyvista` now preserve `float64`
  point coordinates instead of unconditionally narrowing geometry to `float32`,
  which could collapse small features on meshes with large coordinate offsets.
  Existing `float32` geometry remains `float32`.
- `physicsnemo.mesh`: `Mesh.to(<float dtype>)` and `DomainMesh.to(<float dtype>)`
  raised `TypeError: cells must have an int-like dtype` because the cast was applied
  to the integer `cells` tensor. A floating/complex dtype is now applied only to
  floating tensors; the integer `cells` (and any integer data) are preserved. Device
  moves are unchanged.
- `physicsnemo.mesh`: fixed several silent-wrong-result bugs — `slice_cells`
  carried stale point-level and non-local (`gaussian_curvature`) caches onto the
  sliced mesh; the intrinsic LSQ gradient returned all-zeros for codimension >= 2
  manifolds (now estimates the tangent space via local PCA); `smooth_laplacian`
  returned stale geometry caches after its in-place point update; `transform`
  propagated an incorrect point-normals cache under anisotropic/shear maps; and the
  derived-mesh methods (`compute_point_derivatives`, `compute_cell_derivatives`,
  `cell_data_to_point_data`, `point_data_to_cell_data`) aliased the source mesh's
  mutable `_cache`.
- `physicsnemo.mesh.spatial.ClusterTree.compute_source_aggregates` now
  normalizes with its call-time area weights instead of the weights used when
  constructing the tree, preserving correct aggregates when weights change.
- `physicsnemo.mesh`: fixed crash / data-integrity bugs — `project(...)` with
  `transform_point_data`/`transform_cell_data=True` mutated the input mesh in
  place; visualization and `to_pyvista` crashed on autograd-tracked tensors (now
  detached before `.numpy()`); and integer/bool data crashed (`safe_eps` on an
  integer dtype) or truncated via integer division during facet/scatter
  aggregation (now computed in a floating dtype).
- `physicsnemo.mesh` Morton-code quantization now handles empty inputs, tiny
  extents, half-precision coordinates, and one-dimensional endpoints correctly.
- `physicsnemo.mesh`: fixed Loop subdivision pulling open boundaries inward (now
  applies the boundary/crease mask); subdivision zero-filling integer/bool
  `point_data` at new edge vertices (now inherits a parent label);
  non-deterministic orientation flips and over-counted component sizes in
  `repair.fix_orientation`; random point sampling drawing barycentric weights in
  float32 for float64 meshes; and `Mesh.merge` not validating `point_data` /
  `global_data` key consistency.
- Fixed `DefaultTrainingLoop` reading `DistributedManager.device` at the class
  level (a `property` descriptor) instead of `DistributedManager().device`, which
  left the loop's device set to a `property` object under an initialized
  `DistributedManager` (`physicsnemo/active_learning/loop.py`).
- Replaced three plain-string regex / docstring literals containing invalid
  escape sequences with raw-string equivalents
  (`physicsnemo/utils/logging/launch.py`,
  `physicsnemo/metrics/general/calibration.py`,
  `physicsnemo/metrics/general/crps.py`); these were `SyntaxWarning`s today
  and become `SyntaxError`s in Python 3.16.
- Various test cleanups to remove self-inflicted warnings in CI output:
  disabled pytest collection for `TestModelA`/`TestModelB` helpers in
  `test/core/test_registry.py` via `__test__ = False`; migrated
  `test/nn/module/test_interpolation.py` to call the non-deprecated
  `grid_to_point_interpolation` and added a dedicated test for the
  deprecation alias; scoped a `lr_scheduler.step()`-before-`optimizer.step()`
  `UserWarning` filter to a single test in
  `test/optim/test_combined_optimizer.py`; guarded the
  `DistributedManager.initialize()` calls in `test/utils/test_checkpoint.py`
  with `is_initialized()`; and suppressed the import-time
  `ExperimentalFeatureWarning` in `test/datapipes/healda/test_features.py`
  via `warnings.catch_warnings()`.
- Fixed `physicsnemo.utils.get_checkpoint_dir` returning paths with `\`
  separators on Windows (e.g. `.\checkpoints_model`), which was inconsistent
  with the `/`-based paths used elsewhere in the checkpoint utilities and
  broke the `test_get_checkpoint_dir` CI test on Windows. The function now
  always joins with `/`, working uniformly for local paths and `fsspec`
  URIs (`msc://`, etc.) across operating systems.

### Security

### Dependencies

- Removes `pyacvd` from the `mesh-extras` optional dependencies. Remeshing now
  uses NVIDIA Warp.
- Updates the minimum supported `warp-lang` version to 1.14.0.

## [2.1.0] - 2026-05-26

### Added

- Adds GLOBE model (`physicsnemo.experimental.models.globe.model.GLOBE`),
  including new variant that uses a dual tree traversal algorithm to reduce the
  complexity of the kernel evaluations from O(N^2) to O(N).
- Adds GLOBE AirFRANS example case (`examples/cfd/external_aerodynamics/globe/airfrans`)
- Adds GLOBE DrivAerML example case (`examples/cfd/external_aerodynamics/globe/drivaer`)
- Adds drop-test dynamics recipe.
- Adds concrete dropout uncertainty quantification for GeoTransolver. Learnable
  per-layer dropout rates enable MC-Dropout inference for uncertainty
  estimates. Disabled by default (`concrete_dropout: false`).
- Adds automatic support for `FSDP` and/or `ShardTensor` models in checkpoint save/load
  functionality
- PhysicsNeMo-Mesh now supports conversion from PyVista/VTK/VTU meshes that may
  contain polyhedral cells.
- In PhysicsNeMo-Mesh, adds `Mesh.to_point_cloud()`, `.to_edge_graph()`, and
  `.to_dual_graph()` methods. These allow Mesh conversion to 0D point clouds, 1D
  edge graphs, and 1D dual graphs, respectively, when connectivity information
  is not needed.
- Adds `physicsnemo.mesh.generate` subpackage with `marching_cubes` for
  isosurface extraction from 3D scalar fields, returning a `Mesh` object.
  Supports the NVIDIA Warp backend.
- Adds a type system to PhysicsNeMo-Mesh, allowing annotation of Mesh dimensions
  using notation like `Mesh[2, 3]` for a 2D manifold in 3D space.
- Adds adjacency caching to PhysicsNeMo-Mesh `Mesh` objects, allowing efficient
  reuse of neighbor information.
- Adds `DomainMesh` class for grouping an interior mesh with named boundary
  meshes and domain-level metadata, with passthrough geometric transforms
  (translate, rotate, scale, transform) and data operations.
- Allows selective per-field transformation of `Mesh` objects: `transform_point_data`,
  `transform_cell_data`, and `transform_global_data` now accept `bool | TensorDict`
  (or plain `dict` for convenience).
- Adds `physicsnemo.mesh.remeshing` subpackage with `partition_cells()` for
  creating Voronoi regions around seed points. BVH-accelerated.
- Added support for 1D, 2D, and 3D neighborhood attention (natten) via
  `physicsnemo.nn.functional` interface, with full `ShardTensor` support.
- Added derivative functionals in `physicsnemo.nn.functional` for
  `uniform_grid_gradient`, `rectilinear_grid_gradient`,
  `spectral_grid_gradient`, `meshless_fd_derivatives`, `mesh_lsq_gradient`,
  and `mesh_green_gauss_gradient`.
- Adds `physicsnemo.sym` module for symbolic PDE residual computation
  (`PhysicsInformer`). Users define PDEs via SymPy and select a gradient method
  (`autodiff`, `finite_difference`, `spectral`, `meshless_finite_difference`,
  `least_squares`); spatial derivatives are computed automatically using the
  `nn.functional.derivatives` functionals.
- Ports all physics-informed examples (LDC PINNs, Darcy, Stokes MGN, DoMINO,
  datacenter, xaeronet, MHD/SWE PINO) to the new `physicsnemo.sym` interface,
  replacing the separate `physicsnemo-sym` package dependency. Geometry is now
  handled via `physicsnemo.mesh` and PyVista.
- Added geometry functionals in `physicsnemo.nn.functional` for
  `mesh_poisson_disk_sample`, `mesh_to_voxel_fraction`, and
  `signed_distance_field`.
- Added rendering functionals in `physicsnemo.nn.functional` for isosurface,
  mesh, volume, LIC, point cloud, wireframe, and RGBA transfer rendering, with
  Warp kernels for rendering and PyTorch fallbacks for transfer functions.
- Adds embedded OOD guardrail `OODGuard` at
  `physicsnemo.experimental.guardrails.embedded`, optionally
  wired into `GeoTransolver` via a new `guard_config` constructor argument.
  The guard calibrates per-channel global bounds and a geometry-latent
  kNN threshold during training, and emits warnings on out-of-distribution
  inputs at inference.
- In PhysicsNeMo-Mesh, `physicsnemo.mesh.geometry` now publicly exposes
  `stable_angle_between_vectors` and `compute_triangle_angles` (previously
  only available via the private `physicsnemo.mesh.curvature._utils`).
- PhysicsNeMo Datapipes enables reproducability through `torch.generator`
  utilities.
- PhysicsNeMo Datapipes now supports `physicsnemo.mesh.Mesh` and
  `physicsnemo.mesh.DomainMesh` objects for deserialization, with
  transformations and utilities for mesh-based datasets.
- PhysicsNeMo Datapipes now support `MultiDataset` construction,
  allowing on-the-fly construction of multi-source composite datasets
  that can be sampled and processed efficiently and coherently
  as one dataset.
- PhysicsNeMo Datapipes also support random augmentations for
  mesh-based datapipes, leveraging `torch.distributions` for
  broad random distribution support. Mesh and DomainMesh
  datasets allow random translation, scaling, and rotation
  of mesh data in coherent ways, compatible with reproducability
  features of physicsnemo datapipes.
- Adds a new *unified* training recipe for external aerodynamics
  that supports training on multiple datasets (DrivaerML, ShiftSUV,
  HighLiftAeroML, or more, bring your own, mix and match), supports
  training several different models (Domino, Transolver, GeoTransolver,
  Flare, GeoTransolver with Flare-attention, bring your own!).  Leverages
  mesh datasets and non-dimensionalization to enable dataset mixing and
  matching at runtime.  Train with surface or volume data.
- Adds a new `physicsnemo.diffusion.multi_diffusion` subpackage that
  scales 2D diffusion models to large domains via patch-based training
  and inference. Provides `MultiDiffusionModel2D` (wraps a base model and
  handles state patching, conditioning preprocessing, positional-embedding
  injection, and per-patch output fusion), the
  `MultiDiffusionMSEDSMLoss` / `MultiDiffusionWeightedMSEDSMLoss` losses
  for patch-based DSM training, and `MultiDiffusionPredictor` for
  sampling (plugs straight into `sample()` / `get_denoiser()` and the
  standard solvers). Patching primitives (`BasePatching2D`,
  `GridPatching2D`, `RandomPatching2D`) are exposed under the same
  subpackage and are `torch.compile`-friendly with `fullgraph=True`.
  `MultiDiffusionPredictor` supports memory-efficient inference on
  large domains via `chunk_size` and `use_checkpointing`. The
  subpackage also ships patch-local DPS guidance:
  `MultiDiffusionDPSScorePredictor` (drop-in score predictor that plugs
  into the standard sampling stack),
  `MultiDiffusionDataConsistencyDPSGuidance` for inpainting and sparse
  data assimilation, and `MultiDiffusionModelConsistencyDPSGuidance` for
  generic patch-local observation operators. Use these instead of the
  global `DPSScorePredictor` to run guided sampling on domains that
  would otherwise OOM.
- Adds `"epsilon"` as a supported prediction type throughout the diffusion
  framework, alongside the existing `"x0"` and `"score"` modes. A new
  `PredictorType = Literal["x0", "score", "epsilon"]` alias in
  `physicsnemo.diffusion.base` is wired through losses (`MSEDSMLoss`,
  `WeightedMSEDSMLoss`, and the multi-diffusion losses), preconditioners,
  samplers / solvers, DPS guidance, and noise schedulers, enabling
  end-to-end training and sampling of epsilon-parameterized models.
  Losses gain an `epsilon_to_x0_fn` kwarg used for the epsilon-to-x0
  conversion required during DSM training.
- Adds `DiffusionUNet3D` 3D U-Net diffusion backbone for volumetric data at
  `physicsnemo.experimental.models.diffusion_unets`. Implements the
  `DiffusionModel` protocol. Exposes reusable 3D building blocks
  (`Conv3D`, `GroupNorm3D`, `UNetAttention3D`, `UNetBlock3D`) at
  `physicsnemo.experimental.nn`.
- Added support for Batched radius search, which enables Domino
  and GeoTransolver with local features and batch size > 1.
- Added the underfill recipe.

### Changed

- Improved crash recipe with configurable stats directory.
- `physicsnemo.mesh.sampling.find_nearest_cells` uses a KNN-backed
  implementation, and no longer accepts the `bvh=`, `chunk_size=`,
  `max_rounds=`, or `max_candidates_per_point=` parameters.
- &#9888;&#65039; **BC-impact (deep imports):** internal `physicsnemo.nn.functional`
  modules were reorganized by category. Public top-level functional imports are
  unchanged, but code importing internal module paths directly (for example
  `physicsnemo.nn.functional.knn` or
  `physicsnemo.nn.functional.radius_search`) should migrate to
  `physicsnemo.nn.functional.neighbors.*`.
- Consolidated Warp interpolation kernels for grid-to-point and point-to-grid
  backends, and added missing kernel/helper docstrings.
- In PhysicsNeMo-Mesh, dual-mesh primitives gained closed-form fast paths
  for triangle meshes embedded in 3D. `compute_circumcenters` is up to
  ~10000x faster (e.g. 11 s -> ~1 ms on a 360 K-triangle AirFRANS mesh,
  RTX 4090) by replacing batched `torch.linalg.lstsq` over (2, 3) systems
  with a closed-form cross product, and `compute_vertex_angles` is up to
  ~15x faster on the same meshes by replacing the dimension-agnostic
  Gram-determinant formula with an `atan2(||cross||, dot)` formulation.
  Anything that depends on these (Gaussian curvature, FEM Laplacian,
  cotangent weights, Voronoi areas, smoothing) inherits the speedup. See
  `perf.md` for the full audit.
- In PhysicsNeMo-Mesh, BVH construction is faster on GPU.
  `_compute_morton_codes` has a CUDA-specific fused-bits path that
  eliminates the `n_bits` sequential kernel launches of the previous
  bit-loop (5-8x speedup on small / medium meshes), and `BVH.from_mesh`
  reuses the cached `Mesh.cell_centroids` instead of recomputing.
  End-to-end `BVH.from_mesh` is ~2x faster on a 162 K-tet `cube_volume`
  mesh.
- In PhysicsNeMo-Mesh, the topology-dedup APIs
  (`categorize_facets_by_count`, `find_edges_in_reference`,
  `remove_duplicate_cells`, `build_adjacency_from_pairs`) gained optional
  `index_bound` / `n_targets` parameters. When the caller passes a strict
  upper bound (typically `mesh.n_points` or `mesh.n_cells`), the implicit
  `tensor.max().item()` GPU sync is avoided and the dedup uses a packed
  int64 unique (via the new internal `unique_index_tuples`) and a single
  composite-key argsort. End-to-end `get_boundary_edges` and
  `cell_to_cells_adjacency` are ~2x faster on practical-size unstructured
  meshes (e.g. 360 K-triangle AirFRANS).
- &#9888;&#65039; **BC-impact (deep imports):** in PhysicsNeMo-Mesh,
  `stable_angle_between_vectors` and `compute_triangle_angles` moved from
  `physicsnemo.mesh.curvature._utils` to `physicsnemo.mesh.geometry._angles`.
  The old private path is no longer available; use the
  `physicsnemo.mesh.geometry` re-export instead.
- &#9888;&#65039; **BC-impact (pre-release rename):** in PhysicsNeMo-Mesh,
  `DomainMesh.apply` was renamed to `DomainMesh.apply_to_meshes`. The
  original name shadowed the recursive `Tensor -> Tensor` `apply` method
  that `@tensorclass` auto-injects, breaking duck-type symmetry with
  `Mesh.apply` for any code that handled both classes. After the rename,
  `dm.apply(tensor_fn)` works as expected (recurses through every leaf
  tensor in `interior`, `boundaries`, and `global_data`); the original
  Mesh-to-Mesh broadcast is now `dm.apply_to_meshes(mesh_fn)`. Early
  adopters of the unreleased `DomainMesh` API should rename their
  `.apply(...)` callsites to `.apply_to_meshes(...)`.
- Refactored the patching utilities under
  `physicsnemo.diffusion.multi_diffusion.patching`. Patching and fusion
  operations are now more performant and `torch.compile`-friendly (e.g.
  `fullgraph=True`,`error_on_recompile=True`).
- Refactored the `examples/geophysics/diffusion_fwi` full-waveform
  inversion example to use the consolidated `physicsnemo.diffusion` API
  (preconditioners, samplers, losses, DPS guidance) and removed the
  recipe-local copies of these utilities under `utils/`.
- Refactored the `examples/generative/topodiff` recipe to use the
  consolidated `physicsnemo.diffusion` API (`MSEDSMLoss` with
  `prediction_type="epsilon"`, `sample()`, `DPSScorePredictor`) plus a
  recipe-local DDPM scheduler, solver, and classifier guidance. Removed
  the now-unused `Diffusion`, `DatasetTopoDiff`, and `load_data_topodiff`
  abstractions from `physicsnemo.models.topodiff`.
- Significantly expanded CI test coverage for `physicsnemo.diffusion`,
  including new tests for samplers, solvers, preconditioners, losses,
  DPS guidance, multi-diffusion, and patching utilities, plus
  combined-workflow and from-checkpoint round-trip tests. Most tests
  run with `fullgraph=True` and `error_on_recompile` to catch
  `torch.compile` regressions.
- Internal weight initialization in the distributed AFNO layers and the
  `EarthAttention` blocks of `physicsnemo.nn.module.attention_layers` now
  dispatches to `torch.nn.init.trunc_normal_` directly instead of going
  through frozen in-tree copies of the pre-PyTorch-2.12 inverse-CDF
  implementation. PyTorch 2.12 reimplemented `trunc_normal_` as a
  rejection-sampling loop on top of `normal_()` (see
  [pytorch/pytorch#174997](https://github.com/pytorch/pytorch/pull/174997)),
  so seeded from-scratch initialization consumes the RNG stream
  differently on 2.12+ vs older versions. Existing trained checkpoints
  are unaffected (loading bypasses init). Forward-accuracy reference
  outputs for `AFNO`, `ModAFNO`, `Transolver`, `FLARE`, and `Pangu` were
  regenerated against the new algorithm. Rather than wiring per-model
  skips, `test.common.validate_forward_accuracy` now uniformly skips on
  `torch < 2.12` (the reference data is locked to that floor via a single
  `_REFERENCE_DATA_MIN_TORCH` constant; bump it when a PyTorch
  release next changes an init/RNG algorithm any forward-accuracy model
  depends on, and regenerate the `.pth` files at the same time).

### Deprecated

- `physicsnemo.utils.mesh` is deprecated and will be removed in v2.2.0. For
  isosurface extraction, use `physicsnemo.mesh.generate.marching_cubes` instead
  of `sdf_to_stl`. For VTP/OBJ/STL file conversion (`combine_vtp_files`,
  `convert_tesselated_files_in_directory`), use VTK or PyVista directly.
- `physicsnemo.nn.module.utils.trunc_normal_` (and its submodule path
  `physicsnemo.nn.module.utils.weight_init.trunc_normal_`) is deprecated
  and will be removed in v2.2.0. It is now a thin wrapper around
  `torch.nn.init.trunc_normal_` that emits a `DeprecationWarning` on
  call, replacing the frozen in-tree copy of the legacy inverse-CDF
  implementation. Use `torch.nn.init.trunc_normal_` directly.

### Removed

- The legacy in-tree `trunc_normal_` implementation that lived in
  `physicsnemo/models/afno/distributed/layers.py` (`_trunc_normal_` /
  `_no_grad_trunc_normal_`) is removed. These names were private; all
  in-tree call sites now use `torch.nn.init.trunc_normal_`.

### Fixed

- Fixed functional benchmark plot fallback labeling so unlabeled ASV results use
  the same key ordering as the benchmark runner.
- Fixed graph break caused by `FunctionSpec` dispatch (`max(key=)` is not supported by `torch.compile`)
- Fixed bug in Pangu, FengWu attention window shift for asymmetric longitudes
- Fixed a bug in `mesh.sampling.find_nearest_cells`, where a mixup between L2 and L-inf norms
  could cause slightly incorrect nearest-neighbor assignments in highly skewed meshes.
- Fixed TensorDict key-ordering bug in GLOBE's Barnes-Hut kernel that caused
  incorrect results when `tensordict >= 0.12` reordered leaves during
  TensorDict construction from dict literals mixing plain and nested keys.
- In PhysicsNeMo-Mesh, `from_pyvista` now correctly handles
  `UnstructuredGrid` inputs in newer `pyvista` versions, looking up
  cell-type buckets in `cells_dict` with `np.uint8(pv.CellType.X)` keys
  rather than the `IntEnum` value, and skipping non-numeric VTK arrays
  (strings, objects) when copying point / cell / field data into the
  `Mesh` `TensorDict`s instead of failing the conversion.
- In PhysicsNeMo-Mesh, the `Mesh` constructor now preserves data when
  `point_data` / `cell_data` / `global_data` are passed as a non-dict
  `Mapping` (notably PyVista's `DataSetAttributes`). Previously, with
  `tensordict >= 0.12`, the `@tensorclass(tensor_only=True)` auto-init
  silently wrapped such Mappings as `NonTensorData` and dropped every key,
  so e.g. `Mesh(cell_data=pv_mesh.cell_data, ...)` produced an empty
  `cell_data`. `Mesh.__post_init__` now detects this wrapping and unwraps
  the original Mapping before coercing to `TensorDict`. The `tensor_only`
  fast path is preserved, so internal Mesh constructions (slicing,
  transforms, `from_pyvista`) keep their full speed. Backed by new direct-
  construction regression tests, a `cell_data` / `global_data` memmap
  round-trip test, and a committed `.pmsh` golden fixture that locks the
  on-disk format against silent breakage in future changes.
- In PhysicsNeMo-Mesh, `safe_eps(dtype)` is now capped at
  `torch.finfo(dtype).eps`, which fixes a float16 corner case where the
  previous `tiny ** 0.25` floor exceeded machine epsilon and could
  corrupt fp16 mesh quantities. Ad-hoc `+ 1e-10` denominators in
  `smooth_laplacian` and `compute_quality_metrics` have been replaced
  with the dtype-aware `.clamp(min=safe_eps(dtype))` to avoid silently
  zeroing fp16 weights.
- Fixed a silent bug in loading state from checkpoint for
  FSDP-backed models with `use_orig_params=False` and channels last
  memory format.
- Fixed issues with physicsnemo.nn.functional's `radius_search` that
  caused crashes when used with torch.compile.
- Fixed the sinusoidal positional embeddings formula in `SongUNet` and
  `MultiDiffusionModel2D` so it now follows the standard `sin / cos`
  convention. Affected reference data was regenerated.
- Constructing a `Mesh` (or `DomainMesh`) inside a `torch.compile`-traced
  function no longer raises `AttributeError` / `KeyError` or silently
  produces wrong output. The breakage came from two regressions in
  `tensordict >= 0.12.0` (PR `pytorch/tensordict#1552`), where the
  `@tensorclass` init wrapper's bypass branch silently skipped both
  field-default normalization and `__post_init__` under
  `torch.compile`. We pin `tensordict < 0.12` until the upstream fix
  (`pytorch/tensordict#1708`, `pytorch/tensordict#1709`) ships, and add
  a regression test (`test/mesh/mesh/test_compile.py`) that constructs
  a `Mesh` inside `torch.compile` and reads cached properties, so the
  same bug cannot return on a future pin bump unnoticed.

### Dependencies

- Increments minimum viable PyTorch version to `torch>=2.5.0` to support FSDP better
- Upper-bounds `tensordict < 0.12` to avoid the `torch.compile` regressions
  in `tensordict >= 0.12.0` (see corresponding entry under Fixed).

## [2.0.0] - 2026-03-09

### Added

- Refactored diffusion preconditioners in
  `physicsnemo.diffusion.preconditioners` relying on a new abstract base class
  `BaseAffinePreconditioner` for preconditioning schemes using affine
  transformations. Existing preconditioners (`VPPrecond`, `VEPrecond`,
  `iDDPMPrecond`, `EDMPrecond`) reimplemented based on this new interface.
- New `physicsnemo.experimental.nn.symmetry` module that implements building
  blocks that preserve 2D and 3D rotational equivariance using a
  grid-based layout for efficient GPU parallelization, and an emphasis on
  compact `einsum` operations.
- Flare attention support for both Transolver and GeoTransolver models.

### Changed

- PhysicsNemo v2.0 contains significant reorganization of tools.  Please see
  the v2.0-MIGRATION-GUIDE.md to understand what has changed and why.
- DiT (Diffusion Transformer) has been moved from `physicsnemo.experimental.models.dit`
  to `physicsnemo.models.dit`.

### Fixed

- Shape mistmatch bug in the Lennard Jones example

### Dependencies

- CUDA backend is now selected via orthogonal `cu12` / `cu13` extras rather
  than being hardcoded to CUDA 13. Feature extras (`nn-extras`, `utils-extras`,
  etc.) are now CUDA-agnostic and can be combined with either backend, e.g.
  `pip install "nvidia-physicsnemo[cu13,nn-extras]"`. When neither `cu12` nor
  `cu13` is specified, PyTorch is installed from PyPI using its default build
  (currently CUDA 12.8 on Linux). For development with `uv`, use
  `uv sync --extra cu13` (or `--extra cu12`) to select the backend.

## [1.3.0] - 2025-11-17

### Added

- Added mixture_of_experts for weather example in physicsnemo.examples.weather.
  **⚠️Warning:** - It uses experimental DiT model subject to future API changes.
  Added some modifications to DiT architecture in physicsnemo.experimental.models.dit.
  Added learnable option to PositionalEmbedding in physicsnemo.models.diffusion.layers.
- Added lead-time aware training support to the StormCast example.
- Add a device aware kNN method to physicsnemo.utils.neighbors. Works with CPU or GPU
  by dispatching to the proper optimized library, and torch.compile compatible.
- Added additional testing of the DoMINO datapipe.
- Examples: added a new example for full-waveform inversion using diffusion
  models. Accessible in `examples/geophysics/diffusion_fwi`.
- Domain Parallelism: Domain Parallelism is now available for kNN, radius_search,
  and torch.nn.functional.pad.
- Unified recipe for crash modeling, supporting Transolver and MeshGraphNet,
  and three transient schemes.
- Added a check to `stochastic_sampler` that helps handle the `EDMPrecond` model,
  which has a specific `.forward()` signature
- Examples: added a new example for reservoir simulation using X-MeshGraphNet.
  Accessible in `examples/reservoir_simulation`
- Added abstract interfaces for constructing active learning workflows, contained
  under the `physicsnemo.active_learning` namespace. A preliminary example of how
  to compose and define an active learning workflow is provided in `examples/active_learning`.
  The `moons` example provides a minimal (pedagogical) composition that is meant to
  illustrate how to define the necessary parts of the workflow.
- Added a new example for temporal interpolation of weather forecasts using ModAFNO.
  Accessible in `examples/weather/temporal_interpolation`.

### Changed

- Migrated Stokes MGN example to PyTorch Geometric.
- Migrated Lennard Jones example to PyTorch Geometric.
- Migrated physicsnemo.utils.sdf.signed_distance_field to a static return,
  torch-only interface.  It also now works on distributed meshes and input fields.
- Refactored DiTBlock to be more modular
- Added NATTEN 2D neighborhood attention backend for DiTBlock
- Migrated blood flow example to PyTorch Geometric.
- Refactored DoMINO model code and examples for performance optimizations and improved readability.
- Migrated HydroGraphNet example to PyTorch Geometric.
- Support for saving and loading nested `physicsnemo.Module`s. It is now
  possible to create nested modules with `m = Module(submodule, ...)`, and save
  and load them with `Module.save` and `Module.from_checkpoint`.
  **⚠️Warning:** - The modules have to be `physicsnemo.Module`s, and not
  `torch.nn.Module`s.
- Support passing custom tokenizer, detokenizer, and attention `Module`s in
  experimental DiT architecture
- Improved Transolver training recipe's configuration for checkpointing and normalization.
- Bumped `multi-storage-client` version to 0.33.0 with rust client.
- Improved configuration for DLWP Healpix (checkpoint directory) and GraphCast (W&B settings).

### Fixed

- Set `skip_scale` to Python float in U-Net to ensure compilation works.
- Ensure stream dependencies are handled correctly in physicsnemo.utils.neighbors
- Fixed the issue with incorrect handling of files with consecutive runs of
  `combine_stl_solids.py` in the X-MGN recipe.
- Fixed the `RuntimeError: Worker data receiving interrupted` error in the datacenter example.

## [1.2.0] - 2025-08-26

### Added

- Diffusion Transformer (DiT) model. The DiT model can be accessed in
 `physicsnemo.experimental.models.dit.DiT`. **⚠️Warning:** - Experimental feature
  subject to future API changes.
- Improved documentation for diffusion models and diffusion utils.
- Safe API to override `__init__`'s arguments saved in checkpoint file with
  `Module.from_checkpoint("chkpt.mdlus", override_args=set(...))`.
- PyTorch Geometric MeshGraphNet backend.
- Functionality in DoMINO to take arbitrary number of `scalar` or `vector`
  global parameters and encode them using `class ParameterModel`
- TopoDiff model and example.
- Added ability for DoMINO model to return volume neighbors.
- Added functionality in DoMINO recipe to introduce physics residual losses.
- Diffusion models, metrics, and utils: implementation of Student-t
  distribution for EDM-based diffusion models (t-EDM). This feature is adapted
  from the paper [Heavy-Tailed Diffusion Models, Pandey et al.](https://arxiv.org/abs/2410.14171>).
  This includes a new EDM preconditioner (`tEDMPrecondSuperRes`), a loss
  function (`tEDMResidualLoss`), and a new option in corrdiff `diffusion_step`.
  &#9888;&#65039; This is an experimental feature that can be accessed through the
  `physicsnemo.experimental` module; it might also be subjected to API changes
  without notice.
- Bumped Ruff version from 0.0.290 to 0.12.5. Replaced Black with `ruff-format`.
- Domino improvements with Unet attention module and user configs
- Hybrid MeshGraphNet for modeling structural deformation
- Enabled TransformerEngine backend in the `transolver` model.
- Inference code for x-meshgraphnet example for external aerodynamics.
- Added a new example for external_aerodynamics: training `transolver` on
  irregular mesh data for DrivaerML surface data.
- Added a new example for external aerodynamics for finetuning pretrained models.

### Changed

- Diffusion utils: `physicsnemo.utils.generative` renamed into `physicsnemo.utils.diffusion`
- Diffusion models: in CorrDiff model wrappers (`EDMPrecondSuperResolution` and
  `UNet`), the arguments `profile_mode` and `amp_mode` cannot be overriden by
  `from_checkpoint`. They are now properties that can be dynamically changed
  *after* the model instantiation with, for example, `model.amp_mode = True`
  and `model.profile_mode = False`.
- Updated healpix data module to use correct `DistributedSampler` target for
  test data loader
- Existing DGL-based vortex shedding example has been renamed to `vortex_shedding_mgn_dgl`.
  Added new `vortex_shedding_mgn` example that uses PyTorch Geometric instead.
- HEALPixLayer can now use earth2grid HEALPix padding ops, if desired
- Migrated Vortex Shedding Reduced Mesh example to PyTorch Geometric.
- CorrDiff example: fixed bugs when training regression `UNet`.
- Diffusion models: fixed bugs related to gradient checkpointing on non-square
  images.
- Diffusion models: created a separate class `Attention` for clarity and
  modularity. Updated `UNetBlock` accordingly to use the `Attention` class
  instead of custom attention logic. This will update the model architecture
  for `SongUNet`-based diffusion models. Changes are not BC-breaking and are
  transparent to the user.
- &#9888;&#65039; **BC-breaking:** refactored the automatic mixed precision
  (AMP) API in layers and models defined in `physicsnemo/models/diffusion/` for
  improved usability. Note: it is now, not only possible, but *required* to
  explicitly set `model.amp_mode = True` in order to use the model in a
  `torch.autocast` clause. This applies to all `SongUNet`-based models.
- Diffusion models: fixed and improved API to enable fp16 forward pass in
  `UNet` and `EDMPrecondSuperResolution` model wrappers; fp16 forward pass can
  now be toggled/untoggled by setting `model.use_fp16 = True`.
- Diffusion models: improved API for Apex group norm. `SongUNet`-based models
  will automatically perform conversion of the input tensors to
  `torch.channels_last` memory format when `model.use_apex_gn` is `True`. New
  warnings are raised when attempting to use Apex group norm on CPU.
- Diffusion utils: systematic compilation of patching operations in `stochastic_sampler`
  for improved performance.
- CorrDiff example: added option for Student-t EDM (t-EDM) in `train.py` and
  `generate.py`. When training a CorrDiff diffusion model, this feature can be
  enabled with the hydra overrides `++training.hp.distribution=student_t` and
  `++training.hp.nu_student_t=<nu_value>`. For generation, this feature can be
  enabled with similar overrides: `++generation.distribution=student_t` and
  `++generation.nu_student_t=<nu_value>`.
- CorrDiff example: the parameters `P_mean` and `P_std` (used to compute the
  noise level `sigma`) are now configurable. They can be set with the hydra
  overrides `++training.hp.P_mean=<P_mean_value>` and
  `++training.hp.P_std=<P_std_value>` for training (and similar ones with
  `training.hp` replaced by `generation` for generation).
- Diffusion utils: patch-based inference and lead time support with
  deterministic sampler.
- Existing DGL-based XAeroNet example has been renamed to `xaeronet_dgl`.
  Added new `xaeronet` example that uses PyTorch Geometric instead.
- Updated the deforming plate example to use the Hybrid MeshGraphNet model.
- &#9888;&#65039; **BC-breaking:** Refactored the `transolver` model to improve
  readability and performance, and extend to more use cases.
- Diffusion models: improved lead time support for `SongUNetPosLtEmbd` and
  `EDMLoss`. Lead-time embeddings can now be used with/without positional
  embeddings.
- Diffusion models: consolidate `ApexGroupNorm` and `GroupNorm` in
  `models/diffusion/layers.py` with a factory `get_group_norm` that can
  be used to instantiate either one of them. `get_group_norm` is now the
  recommended way to instantiate a GroupNorm layer in `SongUNet`-based and
  other diffusion models.
- Physicsnemo models: improved checkpoint loading API in
  `Module.from_checkpoint` that now exposes a `strict` parameter to raise error
  on missing/unexpected keys, similar to that used in
  `torch.nn.Module.load_state_dict`.
- Migrated Hybrid MGN and deforming plate example to PyTorch Geometric.

### Fixed

- Bug fixes in DoMINO model in sphere sampling and tensor reshaping
- Bug fixes in DoMINO utils random sampling and test.py
- Optimized DoMINO config params based on DrivAer ML

## [1.1.1] - 2025-06-16

### Fixed

- Fixed an inadvertent change to the deterministic sampler 2nd order correction
- Bug Fix in Domino model ball query layer
- Fixed bug models/unet/unet.py: setting num_conv_layers=1 gives errors

## [1.1.0] - 2025-06-05

### Added

- Added ReGen score-based data assimilation example
- General purpose patching API for patch-based diffusion
- New positional embedding selection strategy for CorrDiff SongUNet models
- Added Multi-Storage Client to allow checkpointing to/from Object Storage
- Added a new aerodynamics example using DoMINO to compute design sensitivities
  (e.g., drag adjoint) with respect to underlying input geometry.

### Changed

- Simplified CorrDiff config files, updated default values
- Refactored CorrDiff losses and samplers to use the patching API
- Support for non-square images and patches in patch-based diffusion
- ERA5 download example updated to use current file format convention and
  restricts global statistics computation to the training set
- Support for training custom StormCast models and various other improvements for StormCast
- Updated CorrDiff training code to support multiple patch iterations to amortize
  regression cost and usage of `torch.compile`
- Refactored `physicsnemo/models/diffusion/layers.py` to optimize data type
  casting workflow, avoiding unnecessary casting under autocast mode
- Refactored Conv2d to enable fusion of conv2d with bias addition
- Refactored GroupNorm, UNetBlock, SongUNet, SongUNetPosEmbd to support usage of
  Apex GroupNorm, fusion of activation with GroupNorm, and AMP workflow.
- Updated SongUNetPosEmbd to avoid unnecessary HtoD Memcpy of `pos_embd`
- Updated `from_checkpoint` to accommodate conversion between Apex optimized ckp
  and non-optimized ckp
- Refactored CorrDiff NVTX annotation workflow to be configurable
- Refactored `ResidualLoss` to support patch-accumlating training for
  amortizing regression costs
- Explicit handling of Warp device for ball query and sdf
- Merged SongUNetPosLtEmb with SongUNetPosEmb, add support for batch>1
- Add lead time embedding support for `positional_embedding_selector`. Enable
arbitrary positioning of probabilistic variables
- Enable lead time aware regression without CE loss
- Bumped minimum PyTorch version from 2.0.0 to 2.4.0, to minimize
  support surface for `physicsnemo.distributed` functionality.

### Dependencies

- Made `nvidia.dali` an optional dependency

## [1.0.1] - 2025-03-25

### Added

- Added version checks to ensure compatibility with older PyTorch for distributed
  utilities and ShardTensor

### Fixed

- `EntryPoint` error that occured during physicsnemo checkpoint loading

## [1.0.0] - 2025-03-18

### Added

- DoMINO model architecture, datapipe and training recipe
- Added matrix decomposition scheme to improve graph partitioning
- DrivAerML dataset support in FIGConvNet example.
- Retraining recipe for DoMINO from a pretrained model checkpoint
- Prototype support for domain parallelism of using ShardTensor (new).
- Enable DeviceMesh initialization via DistributedManager.
- Added Datacenter CFD use case.
- Add leave-in profiling utilities to physicsnemo, to easily enable torch/python/nsight
  profiling in all aspects of the codebase.

### Changed

- Refactored StormCast training example
- Enhancements and bug fixes to DoMINO model and training example
- Enhancement to parameterize DoMINO model with inlet velocity
- Moved non-dimensionaliztion out of domino datapipe to datapipe in domino example
- Updated utils in `physicsnemo.launch.logging` to avoid unnecessary `wandb` and `mlflow`
  imports
- Moved to experiment-based Hydra config in Lagrangian-MGN example
- Make data caching optional in `MeshDatapipe`
- The use of older `importlib_metadata` library is removed

### Deprecated

- ProcessGroupConfig is tagged for future deprecation in favor of DeviceMesh.

### Fixed

- Update pytests to skip when the required dependencies are not present
- Bug in data processing script in domino training example
- Fixed NCCL_ASYNC_ERROR_HANDLING deprecation warning

### Dependencies

- Remove the numpy dependency upper bound
- Moved pytz and nvtx to optional
- Update the base image for the Dockerfile
- Introduce Multi-Storage Client (MSC) as an optional dependency.
- Introduce `wrapt` as an optional dependency, needed when using
  ShardTensor's automatic domain parallelism

## [0.9.0] - 2024-12-04

### Added

- Graph Transformer processor for GraphCast/GenCast.
- Utility to generate STL from Signed Distance Field.
- Metrics for CAE and CFD domain such as integrals, drag, and turbulence invariances and
  spectrum.
- Added gradient clipping to StaticCapture utilities.
- Bistride Multiscale MeshGraphNet example.
- FIGConvUNet model and example.
- The Transolver model.
- The XAeroNet model.
- Incoporated CorrDiff-GEFS-HRRR model into CorrDiff, with lead-time aware SongUNet and
  cross entropy loss.
- Option to offload checkpoints to further reduce memory usage
- Added StormCast model training and simple inference to examples
- Multi-scale geometry features for DoMINO model.

### Changed

- Refactored CorrDiff training recipe for improved usability
- Fixed timezone calculation in datapipe cosine zenith utility.
- Refactored EDMPrecondSRV2 preconditioner and fixed the bug related to the metadata
- Extended the checkpointing utility to store metadata.
- Corrected missing export of loggin function used by transolver model

## [0.8.0] - 2024-09-24

### Added

- Graph Transformer processor for GraphCast/GenCast.
- Utility to generate STL from Signed Distance Field.
- Metrics for CAE and CFD domain such as integrals, drag, and turbulence invariances and
  spectrum.
- Added gradient clipping to StaticCapture utilities.
- Bistride Multiscale MeshGraphNet example.

### Changed

- Refactored CorrDiff training recipe for improved usability
- Fixed timezone calculation in datapipe cosine zenith utility.

## [0.7.0] - 2024-07-23

### Added

- Code logging for CorrDiff via Wandb.
- Augmentation pipeline for CorrDiff.
- Regression output as additional conditioning for CorrDiff.
- Learnable positional embedding for CorrDiff.
- Support for patch-based CorrDiff training and generation (stochastic sampling only)
- Enable CorrDiff multi-gpu generation
- Diffusion model for fluid data super-resolution (CMU contribution).
- The Virtual Foundry GraphNet.
- A synthetic dataloader for global weather prediction models, demonstrated on GraphCast.
- Sorted Empirical CDF CRPS algorithm
- Support for history, cos zenith, and downscaling/upscaling in the ERA5 HDF5 dataloader.
- An example showing how to train a "tensor-parallel" version of GraphCast on a
Shallow-Water-Equation example.
- 3D UNet
- AeroGraphNet example of training of MeshGraphNet on Ahmed body and DrivAerNet datasets.
- Warp SDF routine
- DLWP HEALPix model
- Pangu Weather model
- Fengwu model
- SwinRNN model
- Modulated AFNO model

### Changed

- Raise `PhysicsNeMoUndefinedGroupError` when querying undefined process groups
- Changed Indexing error in `examples/cfd/swe_nonlinear_pino` for `physicsnemo` loss function
- Safeguarding against uninitialized usage of `DistributedManager`

### Removed

- Remove mlflow from deployment image

### Fixed

- Fixed bug in the partitioning logic for distributing graph structures
intended for distributed message-passing.
- Fixed bugs for corrdiff diffusion training of `EDMv1` and `EDMv2`
- Fixed bug when trying to save DDP model trained through unified recipe

### Dependencies

- Update DALI to CUDA 12 compatible version.
- Update minimum python version to 3.10

## [0.6.0] - 2024-04-17

### Added

- The citation file.
- Link to the CWA dataset.
- ClimateDatapipe: an improved datapipe for HDF5/NetCDF4 formatted climate data
- Performance optimizations to CorrDiff.
- Physics-Informed Nonlinear Shallow Water Equations example.
- Warp neighbor search routine with a minimal example.
- Strict option for loading PhysicsNeMo checkpoints.
- Regression only or diffusion only inference for CorrDiff.
- Support for organization level model files on NGC file system
- Physics-Informed Magnetohydrodynamics example.

### Changed

- Updated Ahmed Body and Vortex Shedding examples to use Hydra config.
- Added more config options to FCN AFNO example.
- Moved posiitonal embedding in CorrDiff from the dataloader to network architecture

### Deprecated

- `physicsnemo.models.diffusion.preconditioning.EDMPrecondSR`. Use `EDMPecondSRV2` instead.

### Removed

- Pickle dependency for CorrDiff.

### Fixed

- Consistent handling of single GPU runs in DistributedManager
- Output location of objects downloaded with NGC file system
- Bug in scaling the conditional input in CorrDiff deterministic sampler

### Dependencies

- Updated DGL build in Dockerfile
- Updated default base image
- Moved Onnx from optional to required dependencies
- Optional Makani dependency required for SFNO model.

## [0.5.0] - 2024-01-25

### Added

- Distributed process group configuration mechanism.
- DistributedManager utility to instantiate process groups based on a process group config.
- Helper functions to faciliate distributed training with shared parameters.
- Brain anomaly detection example.
- Updated Frechet Inception Distance to use Wasserstein 2-norm with improved stability.
- Molecular Dynamics example.
- Improved usage of GraphPartition, added more flexible ways of defining a partitioned graph.
- Physics-Informed Stokes Flow example.
- Profiling markers, benchmarking and performance optimizations for CorrDiff inference.
- Unified weather model training example.

### Changed

- MLFLow logging such that only proc 0 logs to MLFlow.
- FNO given seperate methods for constructing lift and spectral encoder layers.

### Removed

- The experimental SFNO

### Dependencies

- Removed experimental SFNO dependencies
- Added CorrDiff dependencies (cftime, einops, pyspng, nvtx)
- Made tqdm a required dependency

## [0.4.0] - 2023-11-20

### Added

- Added Stokes flow dataset
- An experimental version of SFNO to be used in unified training recipe for
weather models
- Added distributed FFT utility.
- Added ruff as a linting tool.
- Ported utilities from PhysicsNeMo Launch to main package.
- EDM diffusion models and recipes for training and sampling.
- NGC model registry download integration into package/filesystem.
- Denoising diffusion tutorial.

### Changed

- The AFNO input argument `img_size` to `inp_shape`
- Integrated the network architecture layers from PhysicsNeMo-Sym.
- Updated the SFNO model, and the training and inference recipes.

### Fixed

- Fixed physicsnemo.Module `from_checkpoint` to work from custom model classes

### Dependencies

- Updated the base container to PyTorch 23.10.
- Updated examples to use Pydantic v2.

## [0.3.0] - 2023-09-21

### Added

- Added ability to compute CRPS(..., dim: int = 0).
- Added EFI for arbitrary climatological CDF.
- Added Kernel CRPS implementation (kcrps)
- Added distributed utilities to create process groups and orthogonal process groups.
- Added distributed AFNO model implementation.
- Added distributed utilities for communication of buffers of varying size per rank.
- Added distributed utilities for message passing across multiple GPUs.
- Added instructions for docker build on ARM architecture.
- Added batching support and fix the input time step for the DLWP wrapper.

### Changed

- Updating file system cache location to physicsnemo folder

### Fixed

- Fixed physicsnemo uninstall in CI docker image

### Security

- Handle the tar ball extracts in a safer way.

### Dependencies

- Updated the base container to latest PyTorch 23.07.
- Update DGL version.
- Updated require installs for python wheel
- Added optional dependency list for python wheel

## [0.2.1] - 2023-08-08

### Fixed

- Added a workaround fix for the CUDA graphs error in multi-node runs

### Security

- Update `certifi` package version

## [0.2.0] - 2023-08-07

### Added

- Added a CHANGELOG.md
- Added build support for internal DGL
- 4D Fourier Neural Operator model
- Ahmed body dataset
- Unified Climate Datapipe

### Changed

- DGL install changed from pypi to source
- Updated SFNO to add support for super resolution, flexible checkpoining, etc.

### Fixed

- Fixed issue with torch-harmonics version locking
- Fixed the PhysicsNeMo editable install
- Fixed AMP bug in static capture

### Security

- Fixed security issues with subprocess and urllib in `filesystem.py`

### Dependencies

- Updated the base container to latest PyTorch base container which is based on torch 2.0
- Container now supports CUDA 12, Python 3.10

## [0.1.0] - 2023-05-08

### Added

- Initial public release.
