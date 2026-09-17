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
ops. The meshed set is the intersection of ``{phi < 0}`` with the bounding
box: phi is clipped by the box's own signed distance, so the box faces are
first-class boundary wherever the domain reaches them. The pipeline is
*stamp -> erode -> inflate -> optimize -> repair*:

1. stamp a Kuhn simplicial lattice over the bounding box (``d!`` simplices
   per hypercube, valid in any dimension);
2. keep cells safely inside (``phi(centroid) < -0.2 h``) -- the initial
   mesh is a staircase of perfect lattice simplices, valid by construction;
3. split pinched vertices (thin concave features can make the staircase
   non-manifold at a point);
4. optimize: batched optimal-Delaunay-triangulation (ODT) vertex updates
   -- each interior vertex moves to the volume-weighted average of its
   incident cells' circumcenters (Chen & Xu 2004) -- gated so no step ever creates an
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
from typing import TYPE_CHECKING, Any, Literal, overload

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
    sdf_box,
)

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh

__all__ = ["mesh_implicit_domain", "refit_mesh_to_implicit"]


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
        # Non-finite volumes (a NaN target, e.g. from a phi that is NaN in
        # part of the box) compare False in every branch below, which would
        # classify them as good and let them replace valid vertices.
        return ((vol <= vol_floor) & ((vol <= vol_old) | (vol <= 0))) | ~vol.isfinite()

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


def _coverage_gap(phi, points, cells, bounds, h, seed=0):
    """Worst distance (in units of h) from the zero set to the mesh boundary.

    Probes the bounding box, projects onto ``phi = 0``, and measures each
    projected sample's distance to the nearest boundary vertex or
    boundary-facet centroid. The projection probe count scales with the
    box-to-h ratio (capped at 65536; the sign census below always uses the
    full 65536), but detection of a dropped feature is still
    probabilistic: its projection basin must be hit by a probe. Projection
    convergence is judged in DISTANCE units, |phi| / |grad phi| < 0.05 h,
    so noisy or scaled level sets cannot silently empty the probe set.

    The guard's one job is out-of-contract input, so failure to measure IS
    failure (+inf, never perfect coverage): phi NaN anywhere in the box
    and phi with a sign change but no autograd gradient both return
    +inf, and a Monte-Carlo volume cross-check (from the probe signs)
    returns +inf when the mesh is missing a macroscopic chunk of
    ``{phi < 0}`` -- the two-sided complement to the zero-set gap, which
    only sees boundary the mesh already has. Individual stalled
    projections ARE dropped from the gap max (after a retry): a probe
    oscillating in a positive CSG dead zone certifies nothing and hides
    nothing, and hard-failing on it would reject in-contract composites.
    """
    lo, hi = bounds
    d = points.shape[1]
    span = float((hi - lo).max())
    # The projection stage scales its probe count with the box-to-h ratio;
    # the sign census (NaN detection + volume cross-check) always uses the
    # full budget -- it costs only one batched phi evaluation, and a small
    # census would inflate the Monte-Carlo sigma until the volume check
    # went blind at coarse resolution.
    n_proj = int(min(65536, max(4096, (span / h) ** d)))
    n_probes = 65536
    g = torch.Generator(device="cpu").manual_seed(seed)
    probe = torch.rand(n_probes, d, generator=g, dtype=points.dtype).to(points.device)
    probe = lo + probe * (hi - lo)
    f_probe = phi(probe)
    if bool(torch.isnan(f_probe).any()):
        # NaN phi inside the box (canonical case: a neural field queried
        # outside its training range) carries no sign, so the guard has
        # nothing to measure there -- and the mesher will have dropped
        # that region. Report failure, never perfect coverage. (An
        # inf-valued phi is fine: its sign still classifies the probe.)
        return float("inf")
    signs = torch.sign(f_probe)
    # Volume cross-check (the two-sided half of the guard): the probe signs
    # give a Monte-Carlo estimate of vol({phi < 0} in the box). A mesh
    # missing a macroscopic chunk of the domain fails here even when every
    # converging probe lands on covered zero set -- the zero-set gap alone
    # is blind to boundary the mesh SHOULD have but does not. The 20%
    # floor keeps this a MACROSCOPIC detector: the sub-h boundary rind of
    # an honestly coarse mesh measures up to ~15% (4D sphere at h=0.25:
    # 11.4%; adversarial 3D CSG at h=0.12: 15%) and policing it is the
    # pointwise gap threshold's job, while the failures this exists for --
    # a hollowed interior, a boundary sheet detaching wholesale -- sit at
    # 25-50%.
    box_vol = float(torch.prod(hi - lo))
    p_in = float((signs < 0).double().mean())
    est_vol = box_vol * p_in
    mc_sigma = box_vol * math.sqrt(max(p_in * (1.0 - p_in), 0.0) / n_probes)
    mesh_vol = float(signed_volumes(points, cells).abs().sum())
    if est_vol - mesh_vol > max(0.20 * est_vol, 5.0 * mc_sigma):
        return float("inf")
    if bool((signs >= 0).all()) or bool((signs <= 0).all()):
        # No zero crossing inside the box: either the domain fills the box
        # (boundary = box faces, not phi's zero set) or it is empty (already
        # rejected earlier). Nothing for the guard to measure.
        return 0.0

    def dist_to_zero(pts):
        """``|phi| / |grad phi|`` at ``pts``; None if phi has no gradient."""
        xg = pts.detach().requires_grad_(True)
        f = phi(xg)
        if not f.requires_grad:
            return None
        (grads,) = torch.autograd.grad(f.sum(), xg, allow_unused=True)
        if grads is None:
            return None
        return f.detach().abs() / grads.norm(dim=-1).clamp_min(1e-30)

    surf = project_to_zero_set(phi, probe[:n_proj], iters=6)
    dist_est = dist_to_zero(surf)
    if dist_est is None:
        # A sign change exists but phi exposes no autograd gradient (e.g. a
        # step function): the boundary cannot even be located, let alone
        # certified. Reporting 0.0 here blessed a staircase boundary as
        # perfect coverage.
        return float("inf")
    tol_dist = 0.05 * h
    converged = torch.nan_to_num(dist_est, nan=float("inf")) < tol_dist
    # The zero set can extend beyond the bounding box (a domain clipped by
    # bounds); only its in-box portion is meshable, so out-of-box
    # projections must not count against coverage.
    in_box = ((surf >= lo - 0.5 * h) & (surf <= hi + 0.5 * h)).all(dim=1)
    # In-box projections that fail to converge certify nothing; give the
    # slow-but-sound cases (Newton overshoot near interior critical points
    # of a non-SDF level set, oscillation across CSG kinks in a positive
    # dead zone with no zero set nearby) more iterations, then drop any
    # still-stalled probe. Dropping is safe here because the dangerous
    # failure classes are caught elsewhere: NaN regions at the probe
    # stage, gradient-free boundaries by the no-gradient path, and a
    # macroscopic uncertified region by the volume cross-check.
    retry = (~converged) & in_box
    if bool(retry.any()):
        surf_r = project_to_zero_set(phi, surf[retry], iters=24)
        dist_r = dist_to_zero(surf_r)
        if dist_r is None:
            return float("inf")
        conv_r = torch.nan_to_num(dist_r, nan=float("inf")) < tol_dist
        in_box_r = ((surf_r >= lo - 0.5 * h) & (surf_r <= hi + 0.5 * h)).all(dim=1)
        surf = surf.clone()
        surf[retry] = surf_r
        converged = converged.clone()
        converged[retry] = conv_r
        in_box = in_box.clone()
        in_box[retry] = in_box_r
    surf = surf[converged & in_box]
    if surf.shape[0] == 0:
        if bool(in_box.any()):
            return float("inf")  # in-box zero set exists; projection failed
        return 0.0
    uniq, counts, _, _ = facet_census(cells)
    bfacets = uniq[counts == 1]
    targets = torch.cat(
        [points[torch.unique(bfacets.reshape(-1))], points[bfacets].mean(dim=1)]
    )
    # Chunk the probe rows: a single cdist would materialize an
    # (n_probes, n_targets) matrix, which exceeds GPU memory for fine
    # meshes (n_targets grows with the boundary).
    gap = torch.zeros((), dtype=points.dtype, device=points.device)
    for s0 in range(0, surf.shape[0], 256):
        chunk_min = torch.cdist(surf[s0 : s0 + 256], targets).min(dim=1).values
        gap = torch.maximum(gap, chunk_min.max())
    return float(gap / h)


@overload
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
    full_output: Literal[False] = ...,
) -> "Mesh":
    """Overload: ``full_output=False`` returns just the ``Mesh``."""
    ...


@overload
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
    full_output: Literal[True],
) -> tuple["Mesh", dict[str, Any]]:
    """Overload: ``full_output=True`` returns ``(Mesh, diagnostics)``."""
    ...


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

    The meshed set is ``{phi < 0}`` *intersected with the bounds box*:
    where the domain reaches the box, the box faces are honored as
    boundary, so external-flow "box minus obstacle" domains work directly
    -- pass the obstacle's negated SDF and let the box provide the
    farfield.

    Parameters
    ----------
    phi : callable
        Implicit function ``phi(x: (..., d)) -> (...)``: negative inside the
        domain, positive outside, differentiable (almost everywhere) under
        torch autograd. An exact signed-distance function conditions the
        boundary projection best, but any level-set function with a
        non-vanishing gradient near its zero set works. phi must not be
        NaN anywhere in ``bounds``: NaN values (e.g. a neural field
        queried outside its training range) trip the coverage guard, since
        coverage of that region cannot be certified. See
        :mod:`physicsnemo.mesh.generate.implicit_functions` for building
        blocks.
    bounds : tuple of (array-like, array-like)
        ``(lo, hi)`` corners of an axis-aligned box. Each of length ``d``.
        The domain is clipped to the box (faces become boundary where it
        reaches them). Keep the box reasonably tight: the full lattice over
        ``bounds`` is materialized before clipping, so memory scales with
        ``prod((hi - lo) / h)`` regardless of the domain's own size.
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
        if the coverage guard trips -- including when coverage cannot be
        *certified* because phi is NaN inside bounds, exposes no autograd
        gradient despite a sign change (e.g. a step function), or the mesh
        volume falls short of the Monte-Carlo estimate of the domain
        volume.

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
    centroids = points[cells].mean(dim=1)
    centroid_phi = phi(centroids)
    # Normalize phi-unit thresholds by the gradient magnitude near the zero
    # set, so any level set with a usable gradient works -- not only unit-
    # gradient SDFs (phi = c * sdf must behave identically for any c > 0).
    band = torch.topk(
        centroid_phi.abs(), k=min(1024, centroid_phi.shape[0]), largest=False
    ).indices
    bg = centroids[band].detach().requires_grad_(True)
    f_bg = phi(bg)
    grads = (
        torch.autograd.grad(f_bg.sum(), bg, allow_unused=True)[0]
        if f_bg.requires_grad
        else None
    )
    gscale = float(grads.norm(dim=-1).median()) if grads is not None else 1.0
    if not (gscale > 0 and gscale < float("inf")):
        gscale = 1.0
    # The meshable set is {phi < 0} INTERSECTED with the box, so clip phi
    # by the box's own signed distance (scaled into phi's units): where the
    # domain touches the bounds, the faces become first-class zero set.
    # Without this, face vertices Newton-project onto the only zero set the
    # raw phi has -- the interior one -- and the mesh silently detaches
    # from the faces and collapses inward (``.clamp(lo, hi)`` keeps
    # vertices in the box, not on its faces). Where the domain is strictly
    # interior, ``max(phi, box)`` equals phi near the zero set and nothing
    # changes. The clip also gives every phi a projectable gradient -- even
    # a constant field ("the domain fills the box") -- so boundary vertices
    # always have an anchor (unanchored ODT is a shrinking flow).
    user_phi = phi
    box_phi = sdf_box(lo, hi)

    def phi(x):
        return torch.maximum(user_phi(x), gscale * box_phi(x))

    # Erosion uses the UNCLIPPED phi: its margin exists for the user
    # field's unknown boundary, while the lattice conforms to the box
    # exactly -- face-adjacent cells need no margin from the box term (and
    # would be spuriously eroded by it whenever ceil-rounding makes the
    # actual lattice spacing smaller than the nominal h in the threshold).
    keep = centroid_phi < -erode * h * gscale
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
    # Face-membership tolerance: h-relative, widened for coarse dtypes at
    # large coordinates (float32 at |x|~1e3 rounds worse than 1e-4*h), and
    # capped so it can never capture genuinely off-face vertices.
    coord_scale = float(torch.maximum(lo.abs(), hi.abs()).max())
    face_eps = min(max(1e-4 * h, 16.0 * torch.finfo(dtype).eps * coord_scale), 0.25 * h)

    def pin_face_coords(target_b, current_b):
        """Vertices on a box face slide only within that face.

        Newton projection onto the clipped phi targets the *nearest*
        zero-set branch, which near a box edge or corner is a single face;
        without per-coordinate pinning, edge and corner vertices migrate
        onto one face and the box's own edges chamfer at the h scale.
        Snapping (rather than just keeping the coordinate) makes face
        membership exact, so the pin is a maintained invariant.
        """
        on_lo = (current_b - lo).abs() <= face_eps
        on_hi = (hi - current_b).abs() <= face_eps
        target_b = torch.where(on_lo, lo.expand_as(target_b), target_b)
        return torch.where(on_hi, hi.expand_as(target_b), target_b)

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
        if it > 0 and it % reconnect_every == 0:
            # Full refresh bounds escape-band staleness: the cache stores
            # phi at proposed targets, not at the gated positions actually
            # accepted, and drift can otherwise accumulate unchecked.
            phi_cache = phi(points)
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
        # phi is box-clipped, so its zero set includes the box faces where
        # the domain touches them; the clamp is float-epsilon hygiene for
        # projections that land a hair outside.
        proj = project_to_zero_set(phi, target[bnd]).clamp(lo, hi)
        # A projection that fails to land near the zero set (in gradient-
        # normalized distance units) anchors its vertex instead: with a
        # discontinuous or plateaued phi, the Newton step can strand a
        # vertex in a region with no zero set at all, and an unanchored
        # boundary then creeps outward unchecked.
        near0 = phi(proj).abs() <= 0.5 * h * gscale
        target[bnd] = pin_face_coords(
            torch.where(near0[:, None], proj, points[bnd]), points[bnd]
        )
        if fixed_targets is not None:
            # Pinned vertices sit exactly at their features; the pin
            # overrides smoothing and projection (a corner tip is on the
            # zero set, but projection would slide a vertex off the tip).
            target[fixed_idx] = fixed_targets
        # Escape check only in the near-boundary band: interior vertices
        # more than ~3h inside cannot exit the domain in one gated step.
        band = (~bnd) & (phi_cache > -3.0 * h * gscale)
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
    proj = project_to_zero_set(phi, points[bnd], iters=5).clamp(lo, hi)
    near0 = phi(proj).abs() <= 0.5 * h * gscale
    target[bnd] = pin_face_coords(
        torch.where(near0[:, None], proj, points[bnd]), points[bnd]
    )
    if fixed_targets is not None:
        target[fixed_idx] = fixed_targets
    points = _gated_update(points, cells, target, vol_floor)

    diag["peeled"] = 0
    if peel:
        points, cells, diag["peeled"] = peel_boundary_slivers(
            points, cells, phi, h * gscale, protect_vertices=fixed_idx
        )
        # Deleting cells can expose a vertex-pinched boundary (two closed
        # surface sheets touching at a point), which the ridge-pairing
        # manifoldness check inside the peel cannot see; the split is a
        # no-op when no pinch exists, and only appends vertices, so
        # earlier vertex indices (e.g. pinned features) stay valid.
        points, cells, n_post_split = split_pinched_vertices(points, cells)
        diag["pinch_splits"] += n_post_split

    q = volume_length_quality(points, cells)
    diag.update(
        n_points=int(points.shape[0]),
        n_cells=int(cells.shape[0]),
        all_volumes_positive=bool((signed_volumes(points, cells) > 0).all()),
        boundary_closed_manifold=boundary_is_closed_manifold(cells),
        q_min=float(q.min()),
        q_p01=float(torch.kthvalue(q, max(1, int(0.01 * q.shape[0]))).values),
        q_median=float(q.median()),
        sliver_fraction=float((q < 0.2).double().mean()),
        time_init_s=t_init,
        time_optimize_s=t_optimize,
        time_reconnect_s=t_reconnect,
    )
    # The guard certifies the USER phi's zero set: the box faces the
    # clipped phi adds are covered by construction (face pinning), and
    # running on the raw field keeps its no-gradient and non-finite
    # detection intact (the box clip would lend a step function a healthy
    # gradient near the faces and mask it).
    diag["coverage_gap_h"] = _coverage_gap(
        user_phi, points, cells, (lo, hi), h, seed=seed
    )
    if max_coverage_gap_h is not None and diag["coverage_gap_h"] > max_coverage_gap_h:
        if math.isinf(diag["coverage_gap_h"]):
            raise ValueError(
                "coverage guard tripped: coverage could not be certified. "
                "phi is NaN somewhere inside bounds (e.g. a neural field "
                "queried outside its training range), exposes no usable "
                "autograd gradient despite a sign change (e.g. a step "
                "function), or the mesh volume falls short of the "
                "Monte-Carlo estimate of the domain volume. Clean up phi, "
                "or pass max_coverage_gap_h=None to accept a best-effort "
                "mesh of the well-defined region."
            )
        raise ValueError(
            f"coverage guard tripped: the zero set has a point "
            f"{diag['coverage_gap_h']:.2f}h away from the mesh boundary "
            f"(threshold {max_coverage_gap_h}h). The domain likely has "
            f"features thinner than h={h}; decrease h, or pass "
            f"max_coverage_gap_h=None to accept the loss."
        )

    if fixed_targets is not None:
        diag["pinned_vertex_indices"] = torch.cdist(fixed_targets, points).argmin(dim=1)
    mesh = Mesh(points=points, cells=cells)
    return (mesh, diag) if full_output else mesh


def refit_mesh_to_implicit(
    mesh: "Mesh",
    phi: ImplicitFunction,
    iters: int = 3,
    bounds: tuple | None = None,
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
    bounds : tuple of (array-like, array-like), optional
        The ``(lo, hi)`` box the mesh was generated with. Pass it whenever
        the domain touches the box (an external-flow "box minus obstacle"
        mesh): phi is then clipped by the box exactly as in
        :func:`mesh_implicit_domain`, so vertices on the box faces stay on
        them instead of being dragged onto phi's interior zero set. Face
        vertices receive zero gradient from ``phi``'s parameters (the box
        is not parameterized). Default ``None`` (no clipping).

    Returns
    -------
    Mesh
        Mesh with the same cells and refit points (same dtype/device),
        connected to the autograd graph of ``phi``'s parameters. The
        projection is ungated (gating would break differentiability):
        large shape changes can invert cells, which raises a
        ``UserWarning`` — regenerate instead of refitting in that case.

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
    >>> verts = refit.points[refit.cells]  # differentiable geometry
    """
    from physicsnemo.mesh.mesh import Mesh

    points, cells = mesh.points, mesh.cells
    if bounds is not None:
        lo = torch.as_tensor(bounds[0], dtype=points.dtype, device=points.device)
        hi = torch.as_tensor(bounds[1], dtype=points.dtype, device=points.device)
        user_phi, box_phi = phi, sdf_box(lo, hi)

        def phi(x):
            # No gscale here: the Newton step normalizes by |grad|^2 per
            # point, and the zero set of max(f, g) is sign-determined, so
            # mixed gradient scales cannot move the boundary.
            return torch.maximum(user_phi(x), box_phi(x))

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
    with torch.no_grad():
        n_bad = int((signed_volumes(x.detach(), cells) <= 0).sum())
    if n_bad:
        import warnings

        warnings.warn(
            f"refit_mesh_to_implicit inverted {n_bad} cells: the projection "
            f"is ungated (to stay differentiable) and intended for small "
            f"shape perturbations at fixed topology. Regenerate the mesh "
            f"with mesh_implicit_domain for large shape changes.",
            stacklevel=2,
        )
    return Mesh(points=x, cells=cells)
