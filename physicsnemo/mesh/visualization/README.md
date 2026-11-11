# Mesh Visualization

`Mesh.draw()` plots a mesh with Matplotlib or PyVista, optionally colored by
point or cell data. For the API reference, refer to the
[visualization page](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/mesh/visualization.html).

![Four drawings, each titled with its draw() call. An annulus colored by cell
pressure, and the same annulus colored by point temperature. A sphere drawn
with PyVista and colored by height, and the annulus colored by the magnitude
of a velocity field.](../../../docs/img/mesh/draw_gallery.png)

The examples on this page run in order. The first block defines the mesh that
the later blocks use.

## Quick Start

```python
import torch
from physicsnemo.mesh import Mesh

# Create a simple mesh: two triangles in 2D
points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
mesh = Mesh(points=points, cells=cells)

# Draw with automatic backend selection
mesh.draw()
```

## Backend Selection

With `backend="auto"` (the default), the backend depends on the spatial
dimension of the mesh (`n_spatial_dims`):

- 0D to 2D space: Matplotlib.
- 3D space: PyVista.

If the preferred backend is not installed, `draw()` uses the other one.
Meshes in more than three spatial dimensions raise a `ValueError`.

You can also select the backend explicitly:

```python
# Force matplotlib (supports 3D via mplot3d)
mesh.draw(backend="matplotlib")

# Force PyVista
mesh.draw(backend="pyvista")
```

## Scalar Data Visualization

### Basic Usage

Color points or cells using scalar data:

```python
# Color cells by pressure
mesh.cell_data["pressure"] = torch.rand(mesh.n_cells)
mesh.draw(cell_scalars="pressure", cmap="coolwarm")

# Color points by temperature
mesh.point_data["temperature"] = torch.rand(mesh.n_points)
mesh.draw(point_scalars="temperature", cmap="plasma")
```

### Direct Tensor Values

Pass scalar values directly without storing them:

```python
# Generate scalar values on the fly
scalars = torch.randn(mesh.n_cells)
mesh.draw(cell_scalars=scalars)
```

### Vector Fields (Automatic Norming)

`draw()` colors by the L2 norm of multidimensional data:

```python
# 3D velocity field at each point
mesh.point_data["velocity"] = torch.randn(mesh.n_points, 3)

# Displays the norm: sqrt(vx² + vy² + vz²)
mesh.draw(point_scalars="velocity")
```

### Nested TensorDict Keys

Access nested data with tuple keys:

```python
from tensordict import TensorDict

mesh.cell_data["flow"] = TensorDict(
    {
        "temperature": torch.rand(mesh.n_cells),
        "pressure": torch.rand(mesh.n_cells),
    },
    batch_size=[mesh.n_cells],
)

# Access nested data
mesh.draw(cell_scalars=("flow", "temperature"))
```

## Visualization Parameters

### Colormap Control

```python
mesh.draw(
    cell_scalars="pressure",
    cmap="viridis",  # Colormap name
    vmin=0.0,  # Minimum value
    vmax=1.0,  # Maximum value
)
```

### Transparency Control

```python
mesh.draw(
    alpha_points=1.0,  # Point opacity (0-1)
    alpha_cells=0.3,  # Cell opacity (0-1)
    alpha_edges=0.7,  # Edge opacity (0-1); matplotlib only
    show_edges=True,  # Display cell edges
)
```

The PyVista backend ignores `alpha_edges` and warns when it is not 1.0.

### Drawing on Existing Axes or Plotters

Pass `ax` to draw on an existing Matplotlib `Axes` or PyVista `Plotter`, for
example to overlay several meshes in one scene. With Matplotlib:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
mesh.draw(backend="matplotlib", ax=ax, show=False)
ax.set_title("Custom Title")
plt.show()
```

## Colors

`point_scalars` and `cell_scalars` are mutually exclusive, so that one
colormap describes the plot. Passing both raises a `ValueError`. The neutral
colors depend on which scalars are active:

<!-- markdownlint-disable MD013 -->
| Active Scalars | Points | Cells |
| -------------- | ------ | ----- |
| None | Black | Light blue |
| `point_scalars` | Colored by the scalars | Matplotlib fills each triangle with the mean of its vertex colors. PyVista interpolates across each cell. |
| `cell_scalars` | Black | Colored by the scalars |
<!-- markdownlint-enable MD013 -->

Line segments that Matplotlib draws in 1D or 3D space stay light gray.

## Return Values

`draw()` returns the axes or plotter whether or not it shows the plot:

```python
# Matplotlib: returns a matplotlib Axes
ax = mesh.draw(backend="matplotlib", show=False)
ax.set_title("My Mesh")

# PyVista: returns a pyvista Plotter
plotter = mesh.draw(backend="pyvista", show=False)
plotter.camera_position = "xy"
plotter.show()
```

The full signature of `Mesh.draw()`, with every default, is on the
[visualization page](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/mesh/visualization.html).
The Matplotlib backend ignores `backend_options` and warns when it is set.

## Examples

### 2D Triangle Mesh

```python
# Create mesh
points = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32)
cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
mesh = Mesh(points=points, cells=cells)

# Add data and visualize
mesh.cell_data["pressure"] = torch.tensor([1.0, 1.5])
mesh.point_data["temperature"] = points[:, 0] + points[:, 1]
mesh.draw(cell_scalars="pressure", cmap="coolwarm", show_edges=True)
```

### Multiple Visualizations

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: point scalars
mesh.draw(
    backend="matplotlib",
    ax=axes[0],
    show=False,
    point_scalars="temperature",
)
axes[0].set_title("Point Scalars")

# Right: cell scalars
mesh.draw(
    backend="matplotlib",
    ax=axes[1],
    show=False,
    cell_scalars="pressure",
)
axes[1].set_title("Cell Scalars")

plt.tight_layout()
plt.show()
```

### 3D Surface Mesh

```python
from physicsnemo.mesh.io import from_pyvista
import pyvista as pv

# Load an example surface
pv_mesh = pv.examples.load_airplane()
surface = from_pyvista(pv_mesh)

# Visualize with PyVista, with extra arguments for add_mesh()
surface.draw(
    alpha_cells=0.8,
    show_edges=True,
    backend_options={"smooth_shading": True},
)
```
