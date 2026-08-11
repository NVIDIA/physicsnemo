# Snapshot: domain-parallel mesh phase (2026-08-10)

Handoff state for transferring this work to another system. Branch:
`geotransolver_flare_domain_parallel` (based directly on origin/main;
HEAD = b207d6f3 "Enable domain parallelism for GeoTransolver"). Everything
below the "Committed" line is already in git history; everything under
"Uncommitted" must travel with the snapshot.

## Committed (model phase — done, green on 2 and 4 GPUs)

- FLARE + GeoTransolver domain parallelism, both attention backends
  (GALE/GALE_FA), unified sdpa_wrapper mixed placements, F.linear
  uneven-4D handler (`shard_utils/linear_patches.py`), grad_ops
  consolidation, GALE/context_projector Partial→Replicate resolves.
- Tests: `test/domain_parallel/models/{test_flare,test_geotransolver}.py`,
  `test/domain_parallel/ops/{test_sdpa,test_linear,test_grad_ops}.py`.

## Uncommitted work in this tree

Modified (step 1 of the mesh plan — verified green on CPU:
116 passed / 42 cuda-skipped):

- `test/mesh/conftest.py` — new `make_test_mesh(n_spatial_dims,
  n_manifold_dims, device, *, n_points=None, n_cells=None, seed=42,
  point_data=None, cell_data=None, global_data=None)` factory
  (delegates to the existing literal meshes when `n_points is None`;
  seeded-random otherwise, draw order points → cells → point_data →
  cell_data → global_data) plus a `mesh_factory` fixture.
- `test/mesh/mesh/test_merge.py`, `test_padding.py` — local
  `create_simple_mesh` now thin wrappers over `make_test_mesh`.
- `test/mesh/mesh/test_data_conversion.py` — duplicate factory and
  `assert_on_device` deleted; imports both from `test.mesh.conftest`.
- NOT refactored on purpose: `test/mesh/transformations/
  test_transformations.py::create_mesh_with_caches` (different
  coordinates than the conftest meshes; swapping would change behavior —
  it is monkeypatched instead, see below).

New: `test/domain_parallel/mesh/`

- `conftest.py` — `device` fixture override returning the rank-local
  device-type string; `shard_queries` (scatter_tensor → Shard(0));
  `gather_full` (full_tensor over a named tuple's fields).
- `test_imported_transformations.py` — subclasses the base suite's
  `TestTranslation/TestRotation/TestScale/TestTransform`; an autouse
  fixture monkeypatches `base.create_mesh_with_caches` to warm caches on
  the full mesh (cell quantities stay plain) then rebuild the Mesh with
  Shard(0) points. No feature code written yet — the bet is that
  DTensor's automatic op-level propagation carries the pointwise ops;
  the GPU run decides.
- `test_point_cloud_mesh.py` — placement-contract tests (construction,
  CenterMesh, NormalizeMeshFields, translate) from the earlier
  test-first session; unbaselined.

Retirable debris (untracked, not part of the PR): `gale_debug.py`,
`domain_parallel_geotransolver_flare_plan.md` (original scoping plan —
keep), `benchmarks/physicsnemo/domain_parallel/`,
`examples/minimal/ShardTensorExamples/5_vit_training_loop/results/`.

## Design rulings from this session (binding)

1. **No sharding logic inside library functions.** The sanctioned
   pattern is: computational core = `torch.library.custom_op` (+
   `register_fake`); the ShardTensor handler lives in
   `physicsnemo/domain_parallel/shard_utils/` and is registered with
   `ShardTensor.register_named_function_handler("<ns>.<op>.default",
   wrapper)` — exemplar: `shard_utils/mesh_ops.py:197` for the Warp SDF.
   A duck-typed `hasattr(x, "_spec")` branch inside
   `mesh/spatial/sdf.py` was rejected and reverted; that file is
   pristine at HEAD.
2. **Mesh-native SDF is out of this PR entirely.** The Triton
   mesh-native SDF being on the loader path via `ComputeSDF`
   (`datapipes/transforms/geometric.py:159`) is itself wrong: that
   transform should call the functional Warp op
   (`nn/functional/geometry/sdf.py`,
   `torch.ops.physicsnemo.signed_distance_field`), which already has a
   shard handler. Re-wiring ComputeSDF happens in a separate PR.
   `test_mesh_native_sdf.py` and the SDF import module were deleted.
3. **Rely on automatic Partial resolution.** No manual `redistribute()`
   calls sprinkled into mesh code (a translate offset-resolve was
   rejected). Manual resolves are only for paths that bypass op-level
   propagation. Fix only what GPU-run evidence shows red.
4. Datapipes = separate PR. All mesh deformation = out of scope.
   Corey runs all GPU tests; no CPU pytest runs by the assistant.

## Open architectural question (raised, undecided)

The rest of the planned query-sharded import tier
(`find_nearest_cells`, sampling/integration, implicit functions) is
plain-Python orchestration with no custom-op core — same problem the
mesh-native SDF had. Decide per ruling 2's logic: which of these remain
on the loader path once ComputeSDF moves to the Warp op, and whether
any deserve op-ification here vs. the follow-up PR. The
pointwise/reduction tier (transformations, CenterMesh, integrate,
Gauss-Bonnet) does not have this problem.

## Next actions on the new system

1. Baseline GPU run (closes the tensordict-ShardTensor unknown and
   tests the transformations import in one shot):
   `uv run torchrun --nproc-per-node 2 -m pytest --multigpu-static
   test/domain_parallel/mesh/ -x`
   Repeat with `--nproc-per-node 4` (uneven splits differ).
2. Fix whatever that run shows red, following rulings 1 and 3.
3. Resolve the open question above, then continue the import table from
   the plan: `test_mesh_spec.py` isinstance classes,
   `calculus/test_integration.py`, `curvature/test_curvature_gauss_bonnet.py`,
   `test_domain_mesh_transforms.py` (translate/rotate/scale/transform +
   `TestApplyToMeshes` + global-data groups only).
4. Global-topology guardrails (~13 entry points: `get_facet_mesh`,
   `get_boundary_mesh`, the 3 adjacency getters, `clean`, `subdivide`,
   `merge`, `slice_points`, `is_manifold`, `is_watertight`,
   `to_edge_graph`, `to_dual_graph`, `BVH.from_mesh`,
   `ClusterTree.from_points`) — one shared duck-typed helper raising
   "not supported on sharded meshes; gather first", plus one
   parametrized test. Pattern must follow ruling 1's spirit (checks live
   at entry points is fine here since these raise, not compute).
5. Base-suite neutrality re-check on GPU:
   `uv run pytest test/mesh/ -q`.
