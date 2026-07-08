# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simplex mesh generation for implicit domains, in any dimension.

``mesh_implicit_domain`` meshes ``{x : phi(x) < 0}`` for an arbitrary
implicit function ``phi`` (signed distance or any level set with a usable
gradient), in 2D, 3D, or higher, on CPU or CUDA, using pure PyTorch tensor
ops. The pipeline is *stamp -> erode -> inflate -> optimize -> repair*:

1. stamp a Kuhn simplicial lattice over the bounding box (``d!`` simplices
   per hypercube, valid in any dimension);
2. keep cells safely inside (``phi(centroid) < -0.2 h``) -- the initial
   mesh is a staircase of perfect lattice simplices, valid by construction;
3. split pinched vertices (thin concave features can make the staircase
   non-manifold at a point);
4. optimize: batched ODT vertex updates gated so no step ever creates an
   inverted or below-floor cell, with boundary vertices projected onto
   ``phi = 0`` each iteration, interleaved with quality-greedy bistellar
   flips;
5. peel residual boundary pancakes.

Every update is validity-gated and the iteration budget is fixed, so the
generator cannot fail to emit a valid mesh; input difficulty degrades
element quality (reported in the diagnostics), never existence. Features
smaller than ``h`` cannot be represented; the coverage guard measures the
worst boundary gap and (by default) raises rather than silently dropping
geometry.
"""

import math
import time
from typing import TYPE_CHECKING, Literal

import torch

from physicsnemo.mesh.generate._flips import flip_until_done
from physicsnemo.mesh.generate._lattice import kuhn_lattice
from physicsnemo.mesh.generate._repair import (
    peel_boundary_slivers,
    pin_feature_points,
    split_pinched_vertices,
)
from physicsnemo.mesh.generate._simplex_ops import (
    boundary_is_closed_manifold,
    boundary_vertex_mask,
    compact_mesh,
    facet_census,
    orient_positive,
    signed_volumes,
    volume_length_quality,
)
from physicsnemo.mesh.generate.implicit_functions import (
    ImplicitFunction,
    project_to_zero_set,
)

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh

__all__ = ["mesh_implicit_domain", "refit_mesh_to_implicit"]


def _clip(phi, points, cells, h=None):
    keep = phi(points[cells].mean(dim=1)) < 0
    if h is not None:
        d = points.shape[1]
        vol = signed_volumes(points, cells).abs()
        keep &= vol > 1e-6 * h**d / math.factorial(d)
    return compact_mesh(points, cells[keep])


def _odt_targets(points, cells, h):
    """Volume-weighted circumcenter average per vertex (Chen-Xu ODT)."""
    d = points.shape[1]
    p0 = points[cells[:, 0]]
    rel = points[cells[:, 1:]] - p0[:, None, :]
    rhs = 0.5 * (rel * rel).sum(-1)
    centroid_rel = rel.sum(dim=1) / (d + 1)
    vol = signed_volumes(points, cells)
    good = vol.abs() > 1e-8 * h**d / math.factorial(d)
    cc_rel = centroid_rel.clone()
    cc_rel[good] = torch.linalg.solve(rel[good], rhs[good])
    off = cc_rel - centroid_rel
    dist = off.norm(dim=-1, keepdim=True)
    cc_rel = centroid_rel + off * (2.0 * h / dist.clamp_min(2.0 * h))
    cc = p0 + cc_rel
    w = vol.clamp_min(1e-300)[:, None]

    num = torch.zeros_like(points)
    den = torch.zeros(points.shape[0], 1, dtype=points.dtype, device=points.device)
    idx = cells.reshape(-1)
    num.index_add_(0, idx, (w * cc).repeat_interleave(cells.shape[1], dim=0))
    den.index_add_(0, idx, w.repeat_interleave(cells.shape[1], dim=0))
    has = den[:, 0] > 0
    target = points.clone()
    target[has] = num[has] / den[has]
    return target


def _gated_update(points, cells, target, vol_floor, max_halvings: int = 6):
    """Move toward ``target`` without creating volumes <= ``vol_floor``.

    Already-bad cells may improve (else they would freeze), but no move may
    make any volume non-positive. The revert fallback iterates to a fixed
    point because reverting one cell's vertices shifts neighbors' volumes.
    """
    vol_old = signed_volumes(points, cells)
    scale = torch.ones(points.shape[0], 1, dtype=points.dtype, device=points.device)
    delta = target - points

    def bad_cells(trial):
        vol = signed_volumes(trial, cells)
        return (vol <= vol_floor) & ((vol <= vol_old) | (vol <= 0))

    trial = target
    for _ in range(max_halvings):
        trial = points + scale * delta
        bad = bad_cells(trial)
        if not bad.any():
            return trial
        scale[torch.unique(cells[bad].reshape(-1))] *= 0.5
    while True:
        trial = points + scale * delta
        bad = bad_cells(trial)
        if not bad.any():
            return trial
        bad_verts = torch.unique(cells[bad].reshape(-1))
        if bool((scale[bad_verts] == 0).all()):
            return trial
        scale[bad_verts] = 0.0


def _coverage_gap(phi, points, cells, bounds, h, n_probes=4096, seed=0):
    """Worst distance (in units of h) from the zero set to the mesh boundary.

    Probes the bounding box, projects onto ``phi = 0``, and measures each
    projected sample's distance to the nearest boundary vertex or
    boundary-facet centroid. Large gaps mean the mesh silently failed to
    cover part of the surface (features below the lattice resolution).
    """
    lo, hi = bounds
    d = points.shape[1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    probe = torch.rand(n_probes, d, generator=g, dtype=points.dtype).to(points.device)
    probe = lo + probe * (hi - lo)
    surf = project_to_zero_set(phi, probe, iters=6)
    surf = surf[phi(surf).abs() < 1e-6 * max(1.0, float(h))]
    if surf.shape[0] == 0:
        return 0.0
    uniq, counts, _, _ = facet_census(cells)
    bfacets = uniq[counts == 1]
    targets = torch.cat(
        [points[torch.unique(bfacets.reshape(-1))], points[bfacets].mean(dim=1)]
    )
    gap = torch.cdist(surf, targets).min(dim=1).values.max()
    return float(gap / h)


def mesh_implicit_domain(
    phi: ImplicitFunction,
    bounds: tuple,
    h: float,
    *,
    reconnect: Literal["flips", "none"] = "flips",
    iters: int = 60,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    max_coverage_gap_h: float | None = 1.5,
    feature_points: torch.Tensor | None = None,
    erode: float = 0.2,
    reconnect_every: int = 10,
    flip_q_focus: float = 0.5,
    tol: float = 5e-3,
    peel: bool = True,
    seed: int = 0,
    full_output: bool = False,
) -> "Mesh":
    r"""Generate a simplex mesh of the implicit domain ``{x : phi(x) < 0}``.

    Works in any spatial dimension (set by ``len(bounds[0])``), entirely in
    PyTorch tensor ops on the requested device. The generator is
    *structurally robust*: every optimization step is gated to preserve
    validity, so it always returns a valid mesh; difficult inputs degrade
    element quality (see the ``full_output`` diagnostics), not existence.

    Parameters
    ----------
    phi : callable
        Implicit function ``phi(x: (..., d)) -> (...)``: negative inside the
        domain, positive outside, differentiable (almost everywhere) under
        torch autograd. An exact signed-distance function conditions the
        boundary projection best, but any level-set function with a
        non-vanishing gradient near its zero set works. See
        :mod:`physicsnemo.mesh.generate.implicit_functions` for building
        blocks.
    bounds : tuple of (array-like, array-like)
        ``(lo, hi)`` corners of an axis-aligned box that contains the
        domain. Each of length ``d``.
    h : float
        Target edge length. Geometric features smaller than ``h`` cannot be
        represented (see ``max_coverage_gap_h``).
    reconnect : {"flips", "none"}, optional
        Topology-improvement strategy interleaved with smoothing.
        ``"flips"`` (default): quality-greedy bistellar flips,
        recommended everywhere. ``"none"``: fixed lattice topology
        (fastest; quality suffers on concave domains).
    iters : int, optional
        Optimization iteration budget. The loop exits early when the
        largest vertex displacement falls below ``tol * h``.
    device : str or torch.device, optional
        Device for all tensor work. Default ``"cpu"``.
    dtype : torch.dtype, optional
        Floating dtype for the geometry. Default: ``float64`` on CPU,
        ``float32`` on CUDA.
    max_coverage_gap_h : float or None, optional
        Coverage guard, in units of ``h``. After generation, the zero set
        is probed and the worst gap between it and the mesh boundary is
        measured; a gap above this threshold raises :class:`ValueError`
        (typical causes: domain features thinner than ``h``, or sharp
        convex corners -- see ``feature_points``). ``None`` disables the
        guard. Default ``1.5``.
    feature_points : torch.Tensor, optional
        ``(n_features, d)`` locations (corners, crease points) that the
        mesh must interpolate exactly. One vertex is pinned to each;
        pinned vertices are exempt from smoothing and boundary projection.
        Without this, sharp convex corners are rounded off at the ``h``
        scale (implicit projection has nothing pulling a vertex *into* a
        corner tip).
    erode : float, optional
        Initialization margin: keep lattice cells with
        ``phi(centroid) < -erode * h``. Default ``0.2``.
    reconnect_every : int, optional
        Reconnection cadence in iterations. Default ``10`` (the flips tier
        fires at half this cadence).
    flip_q_focus : float, optional
        Flip candidates are harvested only from neighborhoods whose worst
        cell quality is below this. Default ``0.5``.
    tol : float, optional
        Early-stop displacement tolerance, relative to ``h``.
    peel : bool, optional
        Remove residual zero-volume boundary pancakes at the end.
    seed : int, optional
        Seed for the flip independent-set priorities and coverage probes.
    full_output : bool, optional
        When ``True``, return ``(mesh, diagnostics)`` where ``diagnostics``
        is a dict of quality/validity/coverage metrics and phase timings.

    Returns
    -------
    Mesh
        Volume mesh with ``points (n_points, d)`` in the requested dtype
        and ``cells (n_cells, d+1)`` (int64), positively oriented, with a
        closed-manifold boundary. With ``full_output=True``, the tuple
        ``(mesh, diagnostics)``.

    Raises
    ------
    ValueError
        If the eroded domain is empty at this resolution (``h`` too coarse
        or ``bounds`` not containing the domain), if bounds are invalid, or
        if the coverage guard trips.

    Notes
    -----
    Determinism: runs are seeded and deterministic on CPU. On CUDA,
    ``index_add_``/``scatter_reduce_`` use non-deterministic atomics, so
    bitwise reproducibility across runs is not guaranteed.

    This function is not differentiable end-to-end (topology decisions are
    discrete). For gradients of the *geometry* at fixed topology — e.g.
    d(mesh)/d(shape parameters) in a shape-optimization loop — see
    :func:`refit_mesh_to_implicit`.

    Examples
    --------
    Mesh a 2D disk and a 3D shell with the same code path:

    >>> import torch
    >>> from physicsnemo.mesh.generate import (
    ...     mesh_implicit_domain, sdf_sphere, sdf_difference,
    ... )
    >>> disk = mesh_implicit_domain(
    ...     sdf_sphere([0.0, 0.0], 0.7), ([-1, -1], [1, 1]), h=0.1)
    >>> disk.n_spatial_dims, disk.n_manifold_dims
    (2, 2)
    >>> shell = mesh_implicit_domain(
    ...     sdf_difference(sdf_sphere([0.0] * 3, 0.7), sdf_sphere([0.0] * 3, 0.3)),
    ...     ([-1] * 3, [1] * 3), h=0.15)
    >>> shell.n_manifold_dims
    3
    """
    from physicsnemo.mesh.mesh import Mesh

    if dtype is None:
        dtype = torch.float32 if torch.device(device).type == "cuda" else torch.float64
    lo = torch.as_tensor(bounds[0], dtype=dtype, device=device)
    hi = torch.as_tensor(bounds[1], dtype=dtype, device=device)
    if lo.ndim != 1 or lo.shape != hi.shape:
        raise ValueError(
            f"bounds must be two length-d vectors; got shapes "
            f"{tuple(lo.shape)} and {tuple(hi.shape)}"
        )
    if not bool((hi > lo).all()):
        raise ValueError("bounds must satisfy hi > lo componentwise")
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")
    if reconnect not in ("flips", "none"):
        raise ValueError(f"unknown reconnect strategy: {reconnect!r}")

    t0 = time.perf_counter()
    points, cells = kuhn_lattice(lo, hi, h, device=device, dtype=dtype)
    keep = phi(points[cells].mean(dim=1)) < -erode * h
    if not bool(keep.any()):
        raise ValueError(
            "no lattice cell lies inside the domain at this resolution: "
            "h may be too coarse for the geometry, phi may be positive "
            "everywhere in bounds, or bounds may not contain the domain"
        )
    points, cells = compact_mesh(points, cells[keep])
    cells = orient_positive(points, cells)
    points, cells, n_split = split_pinched_vertices(points, cells)
    t_init = time.perf_counter() - t0

    fixed_targets = None
    fixed_idx = None  # stable vertex identity; reset on renumbering
    if feature_points is not None:
        fixed_targets = torch.as_tensor(
            feature_points, dtype=dtype, device=device
        ).reshape(-1, points.shape[1])
        # Insert a vertex AT each feature (cell split or boundary tent):
        # walking an existing vertex there deadlocks against the gate.
        points, cells, fixed_idx = pin_feature_points(points, cells, fixed_targets, h)

    d = points.shape[1]
    vol_floor = 1e-3 * h**d / math.factorial(d)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    diag = {
        "iters_run": 0,
        "reconnects": 0,
        "flips": 0,
        "pinch_splits": n_split,
    }
    bnd = boundary_vertex_mask(points, cells)
    phi_cache = phi(points)  # for the escape-check band
    t_reconnect = 0.0
    # Flip scheduling: first invocation once points have moved off-grid,
    # then a productive cadence of reconnect_every/2, with exponential
    # backoff after flip-free invocations (a flip pass costs a full-mesh
    # scan; measured GPU wall-time at 7.5e5 cells is dominated by pass
    # COUNT, while small meshes need the frequent cadence for quality).
    productive_interval = max(1, reconnect_every // 2)
    next_flip_it = reconnect_every
    flip_interval = productive_interval

    t0 = time.perf_counter()
    for it in range(iters):
        tr = time.perf_counter()
        if reconnect == "flips" and it >= next_flip_it:
            cells, n_flips = flip_until_done(
                points,
                cells,
                h,
                max_passes=6,
                generator=gen,
                q_focus=flip_q_focus,
            )
            diag["reconnects"] += 1
            diag["flips"] += n_flips
            if n_flips:
                bnd = boundary_vertex_mask(points, cells)
                flip_interval = productive_interval
            else:
                flip_interval *= 2  # flip-quiescent: back off
            next_flip_it = it + flip_interval
        t_reconnect += time.perf_counter() - tr

        target = _odt_targets(points, cells, h)
        target[bnd] = project_to_zero_set(phi, target[bnd])
        if fixed_targets is not None:
            # Pinned vertices sit exactly at their features; the pin
            # overrides smoothing and projection (a corner tip is on the
            # zero set, but projection would slide a vertex off the tip).
            if fixed_idx is None:
                fixed_idx = torch.cdist(fixed_targets, points).argmin(dim=1)
            target[fixed_idx] = fixed_targets
        # Escape check only in the near-boundary band: interior vertices
        # more than ~3h inside cannot exit the domain in one gated step.
        band = (~bnd) & (phi_cache > -3.0 * h)
        if band.any():
            phi_band = phi(target[band])
            esc = torch.zeros_like(band)
            esc[band] = phi_band > 0
            if esc.any():
                target[esc] = project_to_zero_set(phi, target[esc])
            phi_cache = phi_cache.clone()
            phi_cache[band] = torch.where(
                esc[band], torch.zeros_like(phi_band), phi_band
            )
        new_points = _gated_update(points, cells, target, vol_floor)
        disp = (new_points - points).norm(dim=-1).max() / h
        points = new_points
        diag["iters_run"] = it + 1
        if disp < tol:
            break
    t_optimize = time.perf_counter() - t0 - t_reconnect

    target = points.clone()
    target[bnd] = project_to_zero_set(phi, points[bnd], iters=5)
    if fixed_targets is not None:
        if fixed_idx is None:
            fixed_idx = torch.cdist(fixed_targets, points).argmin(dim=1)
        target[fixed_idx] = fixed_targets
    points = _gated_update(points, cells, target, vol_floor)

    diag["peeled"] = 0
    if peel:
        points, cells, diag["peeled"] = peel_boundary_slivers(
            points, cells, phi, h, protect_vertices=fixed_idx
        )

    q = volume_length_quality(points, cells)
    diag.update(
        n_points=int(points.shape[0]),
        n_cells=int(cells.shape[0]),
        all_volumes_positive=bool((signed_volumes(points, cells) > 0).all()),
        boundary_closed_manifold=boundary_is_closed_manifold(cells),
        q_min=float(q.min()),
        q_p01=float(q.quantile(0.01)),
        q_median=float(q.median()),
        sliver_fraction=float((q < 0.2).double().mean()),
        time_init_s=t_init,
        time_optimize_s=t_optimize,
        time_reconnect_s=t_reconnect,
    )
    diag["coverage_gap_h"] = _coverage_gap(phi, points, cells, (lo, hi), h, seed=seed)
    if max_coverage_gap_h is not None and diag["coverage_gap_h"] > max_coverage_gap_h:
        raise ValueError(
            f"coverage guard tripped: the zero set has a point "
            f"{diag['coverage_gap_h']:.2f}h away from the mesh boundary "
            f"(threshold {max_coverage_gap_h}h). The domain likely has "
            f"features thinner than h={h}; decrease h, or pass "
            f"max_coverage_gap_h=None to accept the loss."
        )

    mesh = Mesh(points=points, cells=cells)
    return (mesh, diag) if full_output else mesh


def refit_mesh_to_implicit(
    mesh: "Mesh",
    phi: ImplicitFunction,
    iters: int = 3,
) -> "Mesh":
    r"""Differentiably re-project a mesh's boundary onto ``phi = 0``.

    The topology is held fixed; boundary vertices take graph-preserving
    Newton steps onto the zero set, so gradients flow from the returned
    ``points`` to any parameters inside ``phi`` (and to ``mesh.points``).
    This is the differentiable half of implicit meshing: generate once with
    :func:`mesh_implicit_domain`, then refit inside an optimization loop as
    the shape parameters move.

    Parameters
    ----------
    mesh : Mesh
        Volume mesh (``n_manifold_dims == n_spatial_dims``), e.g. from
        :func:`mesh_implicit_domain` at nearby shape parameters.
    phi : callable
        Implicit function; may close over tensors that require grad.
    iters : int, optional
        Newton projection steps. Default ``3``.

    Returns
    -------
    Mesh
        Mesh with the same cells and refit points (same dtype/device),
        connected to the autograd graph of ``phi``'s parameters.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh.generate import (
    ...     mesh_implicit_domain, refit_mesh_to_implicit, sdf_sphere,
    ... )
    >>> base = mesh_implicit_domain(
    ...     sdf_sphere([0.0, 0.0], 0.7), ([-1, -1], [1, 1]), h=0.1)
    >>> r = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    >>> refit = refit_mesh_to_implicit(base, lambda x: x.norm(dim=-1) - r)
    >>> area = (refit.points[refit.cells],)  # differentiable geometry
    """
    from physicsnemo.mesh.mesh import Mesh

    points, cells = mesh.points, mesh.cells
    bnd = boundary_vertex_mask(points, cells)
    x = points.clone()
    # The Newton step needs autograd for grad(phi) even in inference, so
    # grad mode is enabled locally; callers under torch.no_grad() still work.
    with torch.enable_grad():
        for _ in range(iters):
            xb = x[bnd]
            if not xb.requires_grad:
                xb = xb.clone().requires_grad_(True)
            f = phi(xb)
            (g,) = torch.autograd.grad(f.sum(), xb, create_graph=True)
            step = f[:, None] * g / (g * g).sum(-1, keepdim=True).clamp_min(1e-30)
            step = torch.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
            x = x.clone()
            x[bnd] = xb - step
    return Mesh(points=x, cells=cells)
