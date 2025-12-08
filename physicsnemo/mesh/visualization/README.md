# Mesh Visualization

The `physicsnemo.mesh` library now includes comprehensive visualization capabilities
through the `Mesh.draw()` method, supporting both matplotlib and PyVista backends.

## Quick Start

```python
import torch
from physicsnemo.mesh import Mesh

# Create a simple mesh
points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
mesh = Mesh(points=points, cells=cells)

# Draw with automatic backend selection
mesh.draw()
```

## Backend Selection

The visualization backend is automatically selected based on spatial dimensions:

- **0D, 1D, 2D meshes**: Uses matplotlib (fast, lightweight)
- **3D meshes**: Uses PyVista (GPU-accelerated, interactive)

You can also explicitly specify the backend:

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
# Generate scalar values on-the-fly
scalars = torch.randn(mesh.n_cells)
mesh.draw(cell_scalars=scalars)
```

### Vector Fields (Automatic Norming)

Multi-dimensional data is automatically L2-normed:

```python
# 3D velocity field at each point
mesh.point_data["velocity"] = torch.randn(mesh.n_points, 3)

# Automatically displays norm: sqrt(vx² + vy² + vz²)
mesh.draw(point_scalars="velocity")
```

### Nested TensorDict Keys

Access nested data structures using tuple keys:

```python
from tensordict import TensorDict

mesh.cell_data["flow"] = TensorDict({
    "temperature": torch.rand(mesh.n_cells),
    "pressure": torch.rand(mesh.n_cells),
}, batch_size=[mesh.n_cells])

# Access nested data
mesh.draw(cell_scalars=("flow", "temperature"))
```

## Visualization Parameters

### Colormap Control

```python
mesh.draw(
    cell_scalars="data",
    cmap="viridis",      # Colormap name
    vmin=0.0,            # Minimum value
    vmax=1.0,            # Maximum value
)
```

### Transparency Control

```python
mesh.draw(
    alpha_points=1.0,    # Point opacity (0-1)
    alpha_cells=0.3,     # Cell opacity (0-1)
    alpha_edges=0.7,     # Edge opacity (0-1)
    show_edges=True,     # Display cell edges
)
```

### Custom Axes (Matplotlib Only)

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
mesh.draw(backend="matplotlib", ax=ax, show=False)
ax.set_title("Custom Title")
plt.show()
```

## Color Logic

The visualization system uses intelligent neutral coloring:

- **Points**: Always black when `point_scalars=None`
- **Cells**:
  - No scalars (`point_scalars=None`, `cell_scalars=None`): Light blue
  - Point scalars active: Light gray (cells are background)
  - Cell scalars active: Colored by scalar data using colormap

## Mutual Exclusivity

`point_scalars` and `cell_scalars` are mutually exclusive to avoid colormap ambiguity:

```python
# ✓ Valid
mesh.draw(point_scalars="temperature")
mesh.draw(cell_scalars="pressure")
mesh.draw()  # No scalars

# ✗ Invalid - raises ValueError
mesh.draw(point_scalars="temp", cell_scalars="pressure")
```

## Return Values

```python
# Matplotlib: returns axes object
ax = mesh.draw(backend="matplotlib", show=False)
ax.set_title("My Mesh")

# PyVista: returns plotter object
plotter = mesh.draw(backend="pyvista", show=False)
plotter.camera_position = "xy"
plotter.show()
```

## Complete API Reference

```python
mesh.draw(
    backend="auto",                  # "auto", "matplotlib", "pyvista"
    show=True,                       # Display immediately
    point_scalars=None,              # Point scalar data
    cell_scalars=None,               # Cell scalar data
    cmap="viridis",                  # Colormap name
    vmin=None,                       # Colormap min value
    vmax=None,                       # Colormap max value
    alpha_points=1.0,                # Point opacity
    alpha_cells=0.3,                 # Cell opacity
    alpha_edges=0.7,                 # Edge opacity
    show_edges=True,                 # Display edges
    ax=None,                         # Matplotlib axes (matplotlib only)
    **kwargs                         # Backend-specific arguments
)
```

## Examples

### 2D Triangle Mesh

```python
# Create mesh
points = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32)
cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
mesh = Mesh(points=points, cells=cells)

# Add data and visualize
mesh.cell_data["pressure"] = torch.tensor([1.0, 1.5])
mesh.draw(cell_scalars="pressure", cmap="coolwarm", show_edges=True)
```

### 3D Surface Mesh

```python
from physicsnemo.mesh.io.io_pyvista import from_pyvista
import pyvista as pv

# Load example
pv_mesh = pv.examples.load_airplane()
mesh = from_pyvista(pv_mesh)

# Visualize with PyVista
mesh.draw(alpha_cells=0.8, show_edges=True)
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

## Implementation Details

The visualization system is organized into:

- `physicsnemo/mesh/visualization/draw_mesh.py`: Main dispatcher with backend selection
- `physicsnemo/mesh/visualization/_matplotlib_impl.py`: Matplotlib backend (0D-3D)
- `physicsnemo/mesh/visualization/_pyvista_impl.py`: PyVista backend (3D)
- `physicsnemo/mesh/visualization/_scalar_utils.py`: Scalar data processing utilities

Key features:

- Automatic L2 norm computation for multi-dimensional data
- Support for nested TensorDict key lookup
- Intelligent neutral color selection
- Comprehensive parameter validation
- Full matplotlib 3D support via `mpl_toolkits.mplot3d`
