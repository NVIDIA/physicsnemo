# Discrete Calculus on Simplicial Meshes

`physicsnemo.mesh.calculus` computes gradients, divergence, curl, the
Laplacian, and integrals of fields on simplicial meshes of any dimension. Every
operator runs on the device of the mesh and works with autograd. For the API
reference, refer to the [calculus page](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/mesh/calculus.html).

## Choosing Between DEC and LSQ

Two methods compute the derivatives. Weighted least squares (LSQ) fits a linear
function to the values at the neighbors of each vertex, as computational fluid
dynamics (CFD) codes do. Discrete exterior calculus (DEC) works on the
circumcentric dual mesh, after Desbrun et al. (2005) and Hirani (2003):

![Two panels on the same triangle mesh. Left, labeled method="lsq", a vertex
with arrows to its six neighbors. Right, labeled method="dec", the
circumcenters of the triangles, the shaded dual cell of the vertex, and one
primal edge crossed by its dual edge.](../../../docs/img/mesh/dec_primal_dual.png)

*The stencils behind the two methods. LSQ fits a linear function to the values
at the neighbors of a vertex. In DEC, the dual cell of a vertex joins the
circumcenters of its triangles, and its area is the dual volume. Each primal
edge has a dual edge that joins the circumcenters of its two triangles, and the
cotangent weight is the ratio of their lengths.*

Pick the method by what you need:

<!-- markdownlint-disable MD013 -->
| You Need | Method |
| -------- | ------ |
| A general-purpose gradient, divergence, or curl | LSQ, the default |
| Derivatives on irregular or poor-quality meshes | LSQ |
| Derivatives of tensor fields | LSQ |
| Values at cell centers | LSQ (`compute_gradient_cells_lsq` and related functions) |
| A Laplacian on a manifold | DEC (`compute_laplacian_points_dec`) |
| Discrete structure, such as $d \circ d = 0$ and the discrete Stokes theorem | DEC |
| Differential geometry, such as curvature | DEC |
| Divergence or a Laplacian in more than three spatial dimensions | DEC |
<!-- markdownlint-enable MD013 -->

Both methods are first-order accurate, $O(h)$, for gradients of smooth fields.
Their errors are larger near the boundary, where the neighborhood is one-sided:

![Maps of the gradient error of LSQ and DEC on a jittered grid. A log-log plot
shows the error of both methods falling as O(h) with the mesh spacing h, in
the interior and on the boundary.](../../../docs/img/mesh/calculus_lsq_vs_dec.png)

On irregular meshes, the pointwise error of the DEC divergence and of the DEC
Laplacian does not shrink as the mesh refines. The gradients of both methods
converge.

## Setup

The examples on this page run in order. This block defines the mesh and the
fields that the later blocks use. The mesh is a sphere (a 2D surface in 3D)
with a scalar field and a vector field at its vertices.

```python
import torch
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

mesh = sphere_icosahedral.load(subdivisions=3).to(torch.float64)
x, y, z = mesh.points.unbind(dim=-1)

scalar_field = x * y + z  # (n_points,)
vector_field = torch.stack([-y, x, z], dim=-1)  # (n_points, 3)
cell_values = mesh.cell_centroids[:, 0]  # (n_cells,)

mesh.point_data["pressure"] = scalar_field
mesh.point_data["temperature"] = z**2
```

## The Mesh Methods

For one field, the `Mesh` methods `gradient`, `divergence`, `curl`, and
`laplacian` return the result tensor. They accept a data key or a tensor:

```python
grad_p = mesh.gradient("pressure", gradient_type="extrinsic")  # (n_points, 3)
div_v = mesh.divergence(vector_field, method="dec")  # (n_points,)
curl_v = mesh.curl(vector_field)  # (n_points, 3)
lap_p = mesh.laplacian("pressure")  # (n_points,)
```

`compute_point_derivatives` computes gradients of several `point_data` fields
at once and stores them as `<key>_gradient`:

```python
# Unified interface for gradients of point_data fields
mesh_with_grad = mesh.compute_point_derivatives(
    keys=["pressure", "temperature"],
    method="lsq",  # or "dec"
    gradient_type="extrinsic",  # "intrinsic" (the default), "extrinsic", or "both"
)

# Access results
grad_p = mesh_with_grad.point_data["pressure_gradient"]  # (n_points, n_spatial_dims)
```

`mesh.compute_cell_derivatives` does the same for `cell_data` with LSQ. The
sections below call each operator by name.

## Discrete Exterior Calculus (DEC)

DEC builds discrete operators that satisfy discrete versions of continuous
theorems. The exterior derivative satisfies the discrete Stokes theorem
exactly, and the angle-defect Gaussian curvature satisfies Gauss-Bonnet
exactly on closed surfaces.

### Laplace-Beltrami Operator

```python
from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

# Intrinsic Laplacian:
#   Lap_f(v) = (1 / |star v|) sum (|star e| / |e|) * (f_neighbor - f_v)
laplacian = compute_laplacian_points_dec(mesh, scalar_field)
```

The sign convention is the analyst's: the Laplacian of $x^2$ on a line is
$+2$, and the operator is negative semidefinite.

Properties:

- Uses cotangent weights derived from the finite element method (FEM)
  stiffness matrix, as described below.
- In 2D: equivalent to $\tfrac{1}{2}(\cot \alpha + \cot \beta)$[^meyer2003-eq5].
- In 1D: gives $1/|e|$ (standard finite-difference second derivative).
- In 3D and higher: gives the dihedral-angle cotangent weights, computed from
  the inverse Gram matrix.
- Normalized by the dual 0-cell volumes $|\star v|$: circumcentric (Voronoi)
  volumes in 1D and 2D, a barycentric approximation in 3D and higher (refer to
  [Circumcentric Dual Volumes](#circumcentric-dual-volumes-voronoi-cells)).
- Exact for linear functions at interior vertices of flat meshes
  (codimension 0, where the manifold dimension equals the spatial dimension).
  Boundary vertices have an incomplete 1-ring and give large errors.
- Works on manifolds of any dimension embedded in any ambient space.

<!-- markdownlint-disable MD033 -->
<details>
<summary>Cotangent Weights from the FEM Stiffness Matrix</summary>
<!-- markdownlint-enable MD033 -->

For an $n$-simplex with vertices $v_0, \ldots, v_n$, the cotangent weight
for edge $(i, j)$ is

$$
w_{ij} = -|\sigma| \, (\nabla \lambda_i \cdot \nabla \lambda_j),
$$

where $\lambda_i$ are barycentric coordinate functions and $|\sigma|$ is
the cell volume. The inverse Gram matrix gives the gradient dot products:

$$
\begin{aligned}
E &= [v_1 - v_0, \ldots, v_n - v_0]
   \quad (n \times d \text{ edge matrix}) \\
G &= E E^\top
   \quad (n \times n \text{ Gram matrix}) \\
\nabla \lambda_k \cdot \nabla \lambda_l &= (G^{-1})_{k-1,\, l-1}
   \quad \text{for } k, l \geq 1.
\end{aligned}
$$

This generalizes the classical 2D cotangent formula to arbitrary
dimensions. Refer to [^hirani2003-eq642] and [^meyer2003-eq8].

<!-- markdownlint-disable MD033 -->
</details>
<!-- markdownlint-enable MD033 -->

### Gradient and Divergence

```python
from physicsnemo.mesh.calculus.gradient import compute_gradient_points_dec

# Computes: grad(f) = sharp(df)
grad_f = compute_gradient_points_dec(mesh, scalar_field)  # (n_points, 3)
```

This composes the exterior derivative and the sharp operator. It is exact for
linear functions on flat meshes (codimension 0), including at boundary
vertices.

```python
from physicsnemo.mesh.calculus.divergence import compute_divergence_points_dec

div_v = compute_divergence_points_dec(mesh, vector_field)  # (n_points,)
```

This computes $\star_0^{-1}\, d^{\ast}\, \star_1\, \flat(X)$: the net flux
through the boundary of each dual cell, per unit dual volume. It is exact for
linear vector fields at interior vertices of uniform grids and of 1D meshes.
On irregular triangle or tetrahedral meshes, it is not exact, and its pointwise
error does not shrink as the mesh refines.

### Exterior Derivative, Hodge Star, Sharp, and Flat

The DEC gradient and divergence are built from four lower-level operators in
private modules. Most code never needs them.

<!-- markdownlint-disable MD033 -->
<details>
<summary>The DEC building blocks, with their properties</summary>
<!-- markdownlint-enable MD033 -->

#### Exterior Derivative

```python
from physicsnemo.mesh.calculus._exterior_derivative import (
    exterior_derivative_0,
    exterior_derivative_1,
)

# d: Omega^0 -> Omega^1 (0-forms to 1-forms)
#   df([vi, vj]) = f(vj) - f(vi)
edge_1form, edges = exterior_derivative_0(mesh, scalar_field)

# d: Omega^1 -> Omega^2 (1-forms to 2-forms; circulation around faces)
# Requires n_manifold_dims >= 2.
face_2form, faces = exterior_derivative_1(mesh, edge_1form, edges)
```

Properties:

- $d \circ d = 0$ (exact by construction).
- Discrete Stokes theorem:
  $\langle d\alpha, c\rangle = \langle \alpha, \partial c\rangle$
  (true by definition).

Refer to [^desbrun2005-s5] and [^hirani2003-ch3].

#### Hodge Star

```python
from physicsnemo.mesh.calculus._hodge_star import hodge_star_0, hodge_star_1

# star: Omega^0 -> Omega^n (vertex values to dual n-cells)
#   star_f(star v) = f(v) * |star v|
star_f = hodge_star_0(mesh, scalar_field)

# star: Omega^1 -> Omega^(n-1) (edge values to dual (n-1)-cells)
star_df = hodge_star_1(mesh, edge_1form, edges)
```

Properties:

- Preserves averages between primal and dual cells, as the following equation
  shows.
- $\star\star\alpha = (-1)^{k(n-k)} \alpha$.
- Uses circumcentric (Voronoi) dual cells in 1D and 2D, with mixed areas for
  obtuse triangles, and a barycentric approximation in 3D and higher.

$$
\frac{\langle \alpha, \sigma\rangle}{|\sigma|}
  = \frac{\langle \star\alpha, \star\sigma\rangle}{|\star\sigma|}
$$

Refer to [^hirani2003-def411] and [^desbrun2005-s6].

#### Sharp and Flat Operators

```python
from physicsnemo.mesh.calculus._sharp_flat import sharp, flat

# sharp: Omega^1 -> X (1-forms to vector fields)
grad_vector = sharp(mesh, edge_1form, edges)  # (n_points, n_spatial_dims)

# flat: X -> Omega^1 (vector fields to 1-forms)
one_form = flat(mesh, vector_field, edges)  # (n_edges,)
```

Implementation:

- Sharp ($\sharp$): equation 5.8.1 with support volume intersections and
  barycentric gradients[^hirani2003-eq581].
- Flat ($\flat$): PDP-flat with averaged endpoint vectors[^hirani2003-s56].

Sharp and flat are not exact inverses in
DEC[^hirani2003-prop553]. This is a property of the discrete theory, not a
bug.

Refer to [^hirani2003-ch5].

<!-- markdownlint-disable MD033 -->
</details>
<!-- markdownlint-enable MD033 -->

## Weighted Least-Squares (LSQ) Methods

LSQ methods fit a local linear model at each vertex or cell. They work on
irregular meshes and on tensor-valued fields.

### Gradient

```python
from physicsnemo.mesh.calculus.gradient import (
    compute_gradient_points_lsq,
    compute_gradient_cells_lsq,
)

# At vertices
grad = compute_gradient_points_lsq(
    mesh,
    scalar_field,
    weight_power=2.0,  # Inverse-distance weighting exponent (the default)
    intrinsic=False,  # Set True for tangent-space gradients on manifolds
)

# At cell centers
grad_cells = compute_gradient_cells_lsq(mesh, cell_values)  # (n_cells, 3)
```

Properties:

- Exact for constant and linear fields on flat meshes (codimension 0),
  including at boundary vertices.
- First-order accurate, $O(h)$, for smooth fields.
- Supports intrinsic (tangent-space) gradients for curves and surfaces in any
  spatial dimension. Codimension-1 meshes (curves in 2D, surfaces in 3D) use
  the vertex normals. Higher codimensions use a principal component analysis
  (PCA) estimate of the tangent space.
- Works for scalar and tensor fields: a field of shape `(n_points, ...)` gives
  a gradient of shape `(n_points, n_spatial_dims, ...)`.
- Extrinsic LSQ (the default, `intrinsic=False`) requires
  `n_spatial_dims <= 3`.

### Divergence

```python
from physicsnemo.mesh.calculus.divergence import compute_divergence_points_lsq

div_v = compute_divergence_points_lsq(mesh, vector_field)  # (n_points,)
```

It computes the divergence

$$
\operatorname{div}(v) = \partial_x v_x + \partial_y v_y + \partial_z v_z
$$

as the trace of the LSQ Jacobian. `compute_divergence_cells_lsq` does the same
at cell centers.

### Curl (3D only)

```python
from physicsnemo.mesh.calculus.curl import compute_curl_points_lsq

curl_v = compute_curl_points_lsq(mesh, vector_field)  # Requires n_spatial_dims = 3
```

It computes the curl from the antisymmetric part of the LSQ Jacobian.
`compute_curl_cells_lsq` does the same at cell centers.

## Integration

```python
from physicsnemo.mesh.calculus import integrate

# P1 quadrature of a point field; exact for piecewise-linear fields
total = integrate(mesh, "pressure", data_source="points")
```

`integrate_flux`, `integrate_moment`, and the measure helpers in
`physicsnemo.mesh.calculus.measure` complete the integration API.

## Circumcentric Dual Volumes (Voronoi Cells)

DEC divides by the dual volume $|\star v|$ of each vertex, the area of its dual
cell in the figure above:

```python
from physicsnemo.mesh.geometry.dual_meshes import compute_dual_volumes_0

dual_vols = compute_dual_volumes_0(mesh)  # |star v| for each vertex
```

In every dimension, the dual volumes tile the mesh:
$\sum_v |\star v| = |M|$ up to floating-point rounding.

<!-- markdownlint-disable MD033 -->
<details>
<summary>The formula in each dimension</summary>
<!-- markdownlint-enable MD033 -->

Algorithm (dimension-specific):

1D manifolds (edges):

- Each vertex gets half the length of each incident edge.
- Exact for piecewise-linear 1-manifolds.

2D manifolds (triangles):

- Acute triangles: circumcentric Voronoi formula[^meyer2003-eq7],

$$
|\star v| = \tfrac{1}{8} \sum_{e \ni v} \|e\|^2 \cot \theta_{\mathrm{opp}}(e).
$$

- Obtuse triangles: mixed-area subdivision[^meyer2003-fig4],

$$
|\star v| = \begin{cases}
  \mathrm{area}(T) / 2 & \text{if obtuse at } v, \\
  \mathrm{area}(T) / 4 & \text{otherwise}.
\end{cases}
$$

3D and higher manifolds (such as tetrahedra):

- Barycentric approximation: $|\star v| = \sum_{\sigma \ni v} |\sigma| / (n + 1)$.
- A rigorous circumcentric dual requires "well-centered"
  meshes[^desbrun2005].

Refer to [^meyer2003-s32_34], [^desbrun2005-s3], and [^hirani2003-def245].

<!-- markdownlint-disable MD033 -->
</details>
<!-- markdownlint-enable MD033 -->

## Usage Examples

### Example 1: Laplace-Beltrami on a Curved Surface

```python
from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

# Scalar field on the sphere from the setup block
temperature = mesh.point_data["temperature"]

# Compute the intrinsic Laplacian
laplacian = compute_laplacian_points_dec(mesh, temperature)

# Use for diffusion: dT/dt = kappa * Lap T
mesh.point_data["laplacian_T"] = laplacian
```

### Example 2: Gradient on a Manifold (Intrinsic)

```python
from physicsnemo.mesh.calculus.gradient import compute_gradient_points_lsq

# Compute the gradient in the tangent space of the surface
grad_intrinsic = compute_gradient_points_lsq(
    mesh,
    scalar_field,
    intrinsic=True,  # Solves in tangent space
)

# The result is perpendicular to the vertex normals
assert torch.allclose(
    (grad_intrinsic * mesh.point_normals).sum(dim=-1),
    torch.zeros(mesh.n_points, dtype=mesh.points.dtype),
    atol=1e-6,
)
```

### Example 3: Vector Calculus Identities

The continuous identities $\nabla \times \nabla f = 0$ and
$\nabla \cdot (\nabla \times v) = 0$ hold for LSQ operators only
approximately. They hold to rounding error for linear fields. For the
quadratic fields below on `cube_volume.load(subdivisions=4)`, a 5 x 5 x 5
grid of vertices, the residuals are at rounding level at the center vertex and
at a few other vertices. They reach about 0.10 one layer in from the boundary
and 0.35 at the boundary.

```python
from physicsnemo.mesh.calculus import (
    compute_gradient_points_lsq,
    compute_divergence_points_lsq,
    compute_curl_points_lsq,
)
from physicsnemo.mesh.primitives.volumes import cube_volume

volume = cube_volume.load(subdivisions=4).to(torch.float64)
vx, vy, vz = volume.points.unbind(dim=-1)

# curl(grad(f)) for a linear f: zero to rounding error
f_linear = vx + 2 * vy + 3 * vz
curl_grad_f = compute_curl_points_lsq(
    volume, compute_gradient_points_lsq(volume, f_linear)
)
assert curl_grad_f.abs().max() < 1e-10

# div(curl(v)) for a quadratic v: not zero near the boundary
v_quadratic = torch.stack([vy * vz, vx**2, vx * vy], dim=-1)
div_curl_v = compute_divergence_points_lsq(
    volume, compute_curl_points_lsq(volume, v_quadratic)
)
print(f"max |div(curl v)| = {div_curl_v.abs().max():.2f}")  # 0.35
```

## Dimension Support

The following table shows which operators run, by manifold dimension $n$
(columns) and spatial dimension $d$. Codimension is $d - n$. Each entry lists
the supported spatial dimensions, and a "-" means that the call raises an
error:

<!-- markdownlint-disable MD013 -->
| Operator | $n = 1$ | $n = 2$ | $n = 3$ | $n \geq 4$ |
| -------- | ------- | ------- | ------- | ---------- |
| Gradient (extrinsic) and divergence (LSQ) | $d \leq 3$ | $d \leq 3$ | $d = 3$ | - |
| Gradient (LSQ, `intrinsic=True`) | any $d$ | any $d$ | $d = 3$ | - |
| Curl (LSQ) | $d = 3$ | $d = 3$ | $d = 3$ | - |
| Gradient (DEC) | any $d$ | $d \leq 3$ | $d = 3$ | - |
| Divergence (DEC) | any $d$ | any $d$ | any $d$ | any $d$ |
| Laplacian (DEC) | any $d$ | any $d$ | any $d$ | any $d$ |
| Hodge star, dual volumes | any $d$ | any $d$ | any $d$\* | any $d$\* |
| `exterior_derivative_1` | - | any $d$ | any $d$ | any $d$ |
<!-- markdownlint-enable MD013 -->

\*Dual volumes use the barycentric approximation for $n \geq 3$.

The extrinsic LSQ operators require $d \leq 3$. The intrinsic LSQ gradient
fits in the tangent space instead, so curves and surfaces work in any $d$.
The DEC gradient needs barycentric gradients, which exist for segments in any
$d$, triangles in $d \leq 3$, and tetrahedra in $d = 3$. The DEC Laplacian
uses the FEM cotangent weights in every dimension.

## Known Behavior

$\operatorname{div}(\nabla f)$ is close to $\Delta f$, but not equal to it:

- In DEC, sharp ($\sharp$) and flat ($\flat$) are not exact
  inverses[^hirani2003-prop553].
- Therefore, the composed operator differs from
  `compute_laplacian_points_dec`. Away from the boundary of a regular mesh,
  both converge to $\Delta f$ as the mesh refines. On irregular meshes, the
  pointwise error of either one need not go to zero.
- To compute a Laplacian, call `compute_laplacian_points_dec` directly.

3D dual volumes use a barycentric approximation:

- A rigorous circumcentric dual requires "well-centered" meshes[^desbrun2005].
- No mixed-volume formula for obtuse tetrahedra exists in the literature.
- The barycentric approximation is standard practice.

Boundary vertices have incomplete 1-ring neighborhoods:

- The cotangent Laplacian gives large errors there.
- `mean_curvature_vertices` returns `NaN` at boundary vertices by default
  (`include_boundary=False`).

## References

The three primary works are:

- Meyer, M., Desbrun, M., Schröder, P., & Barr, A. H. (2003).
  *Discrete Differential-Geometry Operators for Triangulated 2-Manifolds*.
  In: Visualization and Mathematics III, pp. 35–57.
- Desbrun, M., Hirani, A. N., Leok, M., & Marsden, J. E. (2005).
  *Discrete Exterior Calculus*. arXiv:math/0508341v2.
- Hirani, A. N. (2003). *Discrete Exterior Calculus*. PhD thesis,
  California Institute of Technology.

Footnotes throughout this document point to specific sections and equations
in these works.

[^meyer2003-eq5]: Meyer et al. (2003), *Discrete Differential-Geometry
Operators for Triangulated 2-Manifolds*, Eq. 5 (cotangent weights).
[^meyer2003-eq7]: Meyer et al. (2003), Eq. 7 (circumcentric Voronoi formula
for acute triangles).
[^meyer2003-eq8]: Meyer et al. (2003), Eq. 8 (cotangent Laplacian for mean
curvature).
[^meyer2003-fig4]: Meyer et al. (2003), Fig. 4 (mixed-area subdivision for
obtuse triangles).
[^meyer2003-s32_34]: Meyer et al. (2003), §3.2–3.4 (mixed Voronoi areas).
[^desbrun2005]: Desbrun et al. (2005), *Discrete Exterior Calculus*,
arXiv:math/0508341v2.
[^desbrun2005-s3]: Desbrun et al. (2005), §3 (Primal Simplicial Complex and
Dual Cell Complex; circumcentric duals).
[^desbrun2005-s5]: Desbrun et al. (2005), §5 (Differential Forms and
Exterior Derivative).
[^desbrun2005-s6]: Desbrun et al. (2005), §6 (Hodge Star and Codifferential).
[^hirani2003-def245]: Hirani (2003), Definition 2.4.5 (Circumcentric Dual
Cell).
[^hirani2003-ch3]: Hirani (2003), Chapter 3 (Discrete Forms and Exterior
Derivative).
[^hirani2003-def411]: Hirani (2003), Definition 4.1.1 (Hodge star via volume
ratios).
[^hirani2003-ch5]: Hirani (2003), Chapter 5 (Forms and Vector Fields;
sharp and flat operators).
[^hirani2003-s56]: Hirani (2003), §5.6 (PDP-flat operator).
[^hirani2003-prop553]: Hirani (2003), Proposition 5.5.3 (sharp and flat are
not exact inverses).
[^hirani2003-eq581]: Hirani (2003), Eq. 5.8.1 (PP-sharp formula).
[^hirani2003-eq642]: Hirani (2003), Eq. 6.4.2 (Laplace-Beltrami).
