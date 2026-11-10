# PhysicsNeMo-Mesh Primitives

`physicsnemo.mesh.primitives` is a collection of canonical meshes for
tutorials, testing, and experimentation. Each mesh module generates its mesh
at runtime with a `load()` function. The modules are grouped by family and by
dimensional configuration. The configuration is the manifold dimension of the
cells and the dimension of the space they live in, written `2D→3D` for a
surface in 3D. For the API reference, refer to the
[primitives page](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/mesh/primitives.html).

![Ten primitives in five columns, one for each family. Curves: a helix and a
spiral. Planar: an annulus and an L-shape. Surfaces: a torus and a Möbius
strip. Volumes: a cube and a ball, cut open to show their tetrahedra.
Procedural: a lumpy sphere and a jittered planar
grid.](../../../docs/img/mesh/primitives_gallery.png)

## Quick Start

Every mesh module has a `load()` function. Most take resolution and size
parameters, and all but `procedural.noisy_mesh` take a `device`:

```python
import torch
from physicsnemo.mesh import primitives

# A sphere, coarse or fine, on the CPU or a GPU
sphere = primitives.surfaces.sphere_icosahedral.load(radius=1.0, subdivisions=2)
fine = primitives.surfaces.sphere_icosahedral.load(subdivisions=4, device="cpu")

# The Stanford bunny
mesh = primitives.pyvista_datasets.bunny.load()

# A sphere with radial noise, or noise added to any mesh
lumpy = primitives.procedural.lumpy_sphere.load(noise_amplitude=0.1, seed=42)
noisy = primitives.procedural.noisy_mesh.load(
    base_mesh=sphere, noise_scale=0.05, seed=42
)
```

The examples on this page run in order, so later blocks reuse `mesh` and
`primitives` from this one.

The `pyvista_datasets` modules, `volumes.sphere_volume`, and
`volumes.cylinder_volume` need the optional `pyvista` package. The `text`
module needs `matplotlib`.

## Categories

### `basic/`: Minimal Test Meshes

The `basic` module holds meshes of one to three cells for unit tests.
`single_point`, `three_points`, `single_edge`, `three_edges`,
`single_triangle`, and `two_triangles` each come in a `_2d` and a `_3d`
version, for example `single_triangle_2d`. `single_tetrahedron` and
`two_tetrahedra` complete the set. The point clouds among the primitives
(0D→2D and 0D→3D) are all in this module.

### `curves/`: 1D Manifolds

The following curves are embedded in 1D, 2D, and 3D spaces:

| Function | Dimensions | Description | Properties |
| ---------- | ------------ | ------------- | ------------ |
| `line_segment_1d` | 1D→1D | Line segment on the real line | Open |
| `line_segments_1d` | 1D→1D | Multiple disconnected segments | Disconnected |
| `straight_line_2d` | 1D→2D | Straight line in 2D | Open |
| `circular_arc_2d` | 1D→2D | Circular arc in 2D | Open, constant curvature |
| `circle_2d` | 1D→2D | Closed circle in 2D | Closed, constant curvature |
| `ellipse_2d` | 1D→2D | Closed ellipse in 2D | Closed, variable curvature |
| `polyline_2d` | 1D→2D | Zigzag polyline in 2D | Open, piecewise linear |
| `spiral_2d` | 1D→2D | Archimedean spiral in 2D | Open |
| `straight_line_3d` | 1D→3D | Straight line in 3D | Open |
| `helix_3d` | 1D→3D | Helical curve in 3D | Open, constant curvature |
| `circle_3d` | 1D→3D | Circle in 3D (any orientation) | Closed |
| `trefoil_knot_3d` | 1D→3D | Trefoil knot in 3D | Closed, knotted |
| `spline_3d` | 1D→3D | PyVista's `load_spline` curve, built without PyVista | Open |

### `planar/`: 2D Manifolds in 2D Space

The following triangulated 2D shapes lie in the plane:

| Function | Dimensions | Description | Properties |
| ---------- | ------------ | ------------- | ------------ |
| `unit_square` | 2D→2D | Unit square with subdivisions | Has boundary |
| `rectangle` | 2D→2D | Rectangular domain | Has boundary |
| `equilateral_triangle` | 2D→2D | Equilateral triangle | Has boundary |
| `regular_polygon` | 2D→2D | Regular n-sided polygon | Has boundary |
| `circle_2d` | 2D→2D | Filled disk | Has boundary |
| `annulus_2d` | 2D→2D | Ring (annulus) | Has inner and outer boundaries |
| `l_shape` | 2D→2D | L-shaped domain | Non-convex, has boundary |
| `structured_grid` | 2D→2D | Regular triangular grid | Has boundary |

### `surfaces/`: 2D Manifolds in 3D Space

The following surface meshes are embedded in 3D:

<!-- markdownlint-disable MD013 -->
| Function | Dimensions | Description | Properties |
| ---------- | ------------ | ------------- | ------------ |
| **Spheres** | | | |
| `sphere_icosahedral` | 2D→3D | Sphere from icosahedron subdivision | Closed, uniform triangulation |
| `sphere_uv` | 2D→3D | Sphere from latitude-longitude (UV) parametrization | Closed, polar singularities |
| **Cylinders** | | | |
| `cylinder` | 2D→3D | Cylinder with caps | Closed |
| `cylinder_open` | 2D→3D | Cylinder without caps | Has boundary circles |
| **Other Shapes** | | | |
| `torus` | 2D→3D | Torus (donut shape) | Closed, genus=1 |
| `plane` | 2D→3D | Flat plane | Has boundary |
| `cone` | 2D→3D | Cone with a base cap | Closed |
| `disk` | 2D→3D | Flat disk | Has boundary circle |
| `hemisphere` | 2D→3D | Half sphere | Has boundary circle |
| **Platonic Solids** | | | |
| `cube_surface` | 2D→3D | Cube surface (triangulated) | Closed |
| `tetrahedron_surface` | 2D→3D | Regular tetrahedron | Closed |
| `octahedron_surface` | 2D→3D | Regular octahedron | Closed |
| `icosahedron_surface` | 2D→3D | Regular icosahedron | Closed |
| **Special Surfaces** | | | |
| `mobius_strip` | 2D→3D | Möbius strip | Non-orientable, has boundary |
<!-- markdownlint-enable MD013 -->

### `volumes/`: 3D Manifolds in 3D Space

This module has the following tetrahedral volume meshes:

| Function | Dimensions | Description | Properties |
| ---------- | ------------ | ------------- | ------------ |
| `cube_volume` | 3D→3D | Tetrahedral cube mesh | Structured |
| `sphere_volume` | 3D→3D | Tetrahedral sphere mesh | Unstructured |
| `cylinder_volume` | 3D→3D | Tetrahedral cylinder mesh | Unstructured |
| `tetrahedron_volume` | 3D→3D | Single tetrahedron | Minimal volume mesh |

### `procedural/`: Mesh Variations and Noise Generation

This module has functions for creating modified versions of meshes and for
standalone noise generation.

The following functions create mesh variations:

<!-- markdownlint-disable MD013 -->
| Function | Description | Use Case |
| ---------- | ------------- | ---------- |
| `lumpy_sphere` | Sphere surface with radial noise (2D→3D) | Testing robustness to irregular geometry |
| `lumpy_ball` | Tetrahedral ball with radial noise (3D→3D) | Irregular volume meshes |
| `noisy_mesh` | Any mesh with added Gaussian noise | Generic perturbation utility |
| `perturbed_grid` | Structured grid with random perturbations (2D→2D) | Testing on nearly regular grids |
<!-- markdownlint-enable MD013 -->

The following procedural noise functions are in
`physicsnemo.mesh.primitives.procedural.noise`:

| Function | Description | Dimensions | GPU |
| ---------- | ------------- | ------------ | ----- |
| `perlin_noise_nd` | Dimension-agnostic Perlin noise | 1D–nD | ✓ |
| `perlin_noise_1d` | 1D Perlin noise | 1D | ✓ |
| `perlin_noise_2d` | 2D Perlin noise | 2D | ✓ |
| `perlin_noise_3d` | 3D Perlin noise | 3D | ✓ |

```python
# Generate noise at mesh centroids
from physicsnemo.mesh.primitives.procedural.noise import perlin_noise_nd

centroids = mesh.cell_centroids
noise = perlin_noise_nd(centroids, scale=1.0, seed=42)
mesh.cell_data["noise"] = noise

# Works on any dimensional mesh
points_4d = torch.randn(100, 4)
noise_4d = perlin_noise_nd(points_4d, scale=2.0, seed=123)
```

### `pyvista_datasets/`: PyVista Examples

PyVista downloads some of its example datasets on first use and caches them.
The following functions wrap PyVista's example datasets:

| Function | Dimensions | Description |
| ---------- | ------------ | ------------- |
| `airplane` | 2D→3D | Classic airplane surface mesh |
| `bunny` | 2D→3D | Stanford bunny (computer graphics classic) |
| `ant` | 2D→3D | Ant surface mesh |
| `cow` | 2D→3D | Cow mesh (mixed cell types, automatically triangulated) |
| `globe` | 2D→3D | Earth globe surface |
| `tetbeam` | 3D→3D | Tetrahedral beam (finite element analysis test case) |
| `hexbeam` | 3D→3D | Hexahedral beam (automatically tessellated to tetrahedra) |

### `text`: Text as Meshes

`physicsnemo.mesh.primitives.text` renders a string into a mesh with
Matplotlib's fonts, through the following functions:

- `text_1d_2d`: outlines
- `text_2d_2d`: filled glyphs
- `text_2d_3d`: the surface of extruded glyphs
- `text_3d_3d`: extruded volume

```python
from physicsnemo.mesh.primitives.text import text_2d_2d

label = text_2d_2d("mesh")
```

## Design Principles

The primitives follow these design principles:

- One file per mesh: each mesh type is in its own module.
- Consistent interface: every mesh module has a `load()` function. The
  `text` module and `procedural.noise` provide functions instead.
- Runtime generation: no stored mesh files (PyVista caches its own
  downloads).
- Parametric: most meshes accept resolution and size parameters.
- Device-aware: every `load()` accepts `device`, except
  `procedural.noisy_mesh`, which keeps the device of its `base_mesh`.
- Documented: each module states its manifold and spatial dimensions.

## Testing

The `physicsnemo.mesh` test suite uses these primitives throughout. You can also
use them in your own tests:

```python
def test_my_algorithm():
    from physicsnemo.mesh import primitives

    # Test on various mesh types
    for mesh_loader in [
        primitives.surfaces.sphere_icosahedral.load,
        primitives.pyvista_datasets.bunny.load,
        primitives.surfaces.torus.load,
    ]:
        mesh = mesh_loader()
        result = my_algorithm(mesh)
        assert result is not None
```
