<!-- markdownlint-disable MD013 -->
# PhysicsNeMo Mesh-Zarr Schema

Version: **1** (`schema_version` attr). Reference implementation:
`physicsnemo.datapipes.save_mesh_to_zarr` / `save_domain_mesh_to_zarr`;
validation: `physicsnemo.datapipes.validate_mesh_zarr`. Emitted by
PhysicsNeMo-Curator's `DomainMeshZarrSink`; consumed by `ZarrMeshReader` /
`ZarrDomainMeshReader`.

## Mesh group (`format = "physicsnemo-mesh-zarr"`)

One zarr group per mesh:

```
<group>/                      attrs (required): format, schema_version,
    points                    #   n_points, n_cells; (when n_cells > 0):
    cells                     #   nodes_per_cell, layout
    point_data/<field>        # optional subgroup, one array per field
    cell_data/<field>         # optional subgroup
    global_data/<field>       # optional subgroup (0-d and small arrays)
```

| Array | Shape | Notes |
|---|---|---|
| `points` | `(n_points, n_spatial_dims)` | required |
| `cells` | `(n_cells, nodes_per_cell)` | absent for point clouds (`n_cells == 0`); integer dtype; indices into `points` |
| `point_data/<f>` | `(n_points, ...)` | |
| `cell_data/<f>` | `(n_cells, ...)` | |
| `global_data/<f>` | any (typically 0-d) | |

### Attributes

- `format` (str, required): `"physicsnemo-mesh-zarr"`.
- `schema_version` (int, required): this document's version.
- `n_points`, `n_cells` (int, required): must equal the array shapes.
- `nodes_per_cell` (int, required when `n_cells > 0`).
- `layout` (str, required when `n_cells > 0`): `"soup"` or `"indexed"`.
  **Writers MUST verify** `layout="soup"` (cells exactly
  `arange(n_points).reshape(n_cells, nodes_per_cell)`) before writing it;
  readers rely on it to skip the `cells` read and synthesize indices.
  `"soup"` groups sacrifice shared-vertex connectivity for contiguous
  block reads -- valid only for cell-centric consumers. Consumers that
  need connectivity (e.g. graph networks) MUST require `"indexed"`.

## DomainMesh group (`format = "physicsnemo-domainmesh-zarr"`)

One zarr group per case, mirroring the `.pdmsh` tree:

```
<case>.zarr/                  attrs (required): format, schema_version,
    global_data/<field>       #   boundary_names (list[str])
    interior/                 # a mesh-schema group
    boundaries/<name>/        # mesh-schema groups, one per boundary_names entry
```

Case-level `global_data` is **single-sourced at the root** and is not
copied into member meshes; readers merge it at read time
(`merge_global_data_from`), and a key present both case-level and on a
member mesh is a data error (readers raise). Member meshes keep their own
mesh-local `global_data` (e.g. `TimeValue`).

## Versioning policy

- `schema_version` is a single integer; any change that could make a
  correct v(N) reader misread the data increments it.
- Readers MUST reject groups with `schema_version` greater than the
  version they implement (raise, never best-effort). Groups without the
  attr are treated as version 1.
- Writers always stamp the current version.

## Reserved for version 2

- **Ragged / mixed-element connectivity**: v1 requires fixed-width
  `cells (n_cells, nodes_per_cell)` (homogeneous simplices). v2 will add
  an offsets-based encoding (`connectivity` + `offsets` [+ `types`]),
  following VTKHDF/UGRID practice, for tetrahedra/hexahedra/prism volume
  meshes; the fixed-width form remains the fast path.
- **Connectivity dtype**: v1 preserves the writer's dtype; a v2 writer
  MAY downcast to the smallest safe integer type.

## Design notes (non-normative)

- Chunks are sized along the cell/point axes; align `chunk_cells` with
  the training reader's `subsample_n_cells` so one sample read touches at
  most two chunks per field.
- Default compression is blosc-zstd (bitshuffle); uncompressed stores
  trade bytes for zero decode cost and only win when the working set fits
  the page cache.
- Groups are self-contained (e.g. SDF source geometry stored as an
  additional full-resolution `indexed` boundary) so cases remain portable
  across filesystems and object stores.
