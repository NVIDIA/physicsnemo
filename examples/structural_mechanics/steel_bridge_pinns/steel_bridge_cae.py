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

"""Reference CAE model for a steel plate-girder bridge (2D plane stress).

The bridge deck is idealised as the web of a simply supported steel plate girder of
span ``L`` and depth ``H``, modelled as a two-dimensional plane-stress continuum with
out-of-plane thickness ``t``.  The model is intentionally self-contained (NumPy +
SciPy only) so that the same reference solution can be produced on a laptop, used to
validate the PINN, and used to generate sparse "sensor" measurements.

Physics
-------
Static equilibrium of a linear elastic solid (Navier-Cauchy form)::

    d(sigma_xx)/dx + d(sigma_xy)/dy = 0
    d(sigma_xy)/dx + d(sigma_yy)/dy = rho * g

with plane-stress constitutive law::

    sigma_xx = E/(1 - nu^2) * (eps_xx + nu * eps_yy)
    sigma_yy = E/(1 - nu^2) * (eps_yy + nu * eps_xx)
    sigma_xy = G * gamma_xy,      G = E / (2 (1 + nu))

Boundary conditions
-------------------
* Left bearing pad (``x = 0``, ``y <= bearing_length``): pinned, ``u_x = u_y = 0``.
* Right bearing pad (``x = L``, ``y <= bearing_length``): roller, ``u_y = 0``.
* Wheel patches on the deck (``y = H``): imposed normal traction ``q = F_patch / (l_patch * t)``.
* All other edges are traction free.

Discretisation
--------------
Four-node bilinear isoparametric elements (Q4) with 2x2 Gauss integration, assembled
into a sparse system and solved with a direct sparse solver.  Element stresses are
recovered at the Gauss points, extrapolated to the element nodes and averaged, which
gives a smooth nodal stress field suitable for plotting and for comparing with the PINN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

__all__ = [
    "SteelBridge",
    "BeamTheory",
    "QuadMesh",
    "PlaneStressSolution",
    "solve_plane_stress",
]


@dataclass(frozen=True)
class BeamTheory:
    """Closed-form (Euler-Bernoulli) estimates used for scaling and verification."""

    max_moment: float
    max_bending_stress: float
    midspan_deflection: float


@dataclass(frozen=True)
class SteelBridge:
    """Geometry, material and loading of the reference bridge.

    Parameters
    ----------
    span : float
        Distance between bearings ``L`` in metres.
    depth : float
        Girder depth ``H`` in metres.
    thickness : float
        Out-of-plane thickness ``t`` of the plane-stress domain in metres.
    youngs_modulus, poisson_ratio, density, gravity
        Steel properties and gravity.
    bearing_length : float
        Length of the bearing pads measured up from the bottom flange.
    load_centres : tuple of float
        Wheel patch centres, as fractions of the span.
    load_patch_length : float
        Length of each wheel patch in metres.
    load_per_patch : float
        Total vertical load carried by one wheel patch, in newtons.
    """

    span: float = 12.0
    depth: float = 1.2
    thickness: float = 0.02
    youngs_modulus: float = 210.0e9
    poisson_ratio: float = 0.3
    density: float = 7850.0
    gravity: float = 9.81
    bearing_length: float = 0.10
    load_centres: tuple[float, ...] = (0.35, 0.55)
    load_patch_length: float = 0.6
    load_per_patch: float = 60.0e3

    # ------------------------------------------------------------------ material
    @property
    def shear_modulus(self) -> float:
        """Shear modulus ``G`` in Pa."""
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def constitutive_matrix(self) -> np.ndarray:
        """Plane-stress stiffness matrix ``D`` such that ``sigma = D @ strain``."""
        e, nu = self.youngs_modulus, self.poisson_ratio
        factor = e / (1.0 - nu**2)
        g = self.shear_modulus
        return np.array(
            [
                [factor, factor * nu, 0.0],
                [factor * nu, factor, 0.0],
                [0.0, 0.0, g],
            ]
        )

    @property
    def second_moment_of_area(self) -> float:
        """Second moment of area of the girder section about the bending axis."""
        return self.thickness * self.depth**3 / 12.0

    # ------------------------------------------------------------------ loading
    @property
    def load_patch_bounds(self) -> list[tuple[float, float]]:
        """Start/end abscissae of the wheel patches in metres."""
        half = 0.5 * self.load_patch_length
        return [(c * self.span - half, c * self.span + half) for c in self.load_centres]

    @property
    def patch_traction(self) -> float:
        """Normal traction [Pa] applied under one wheel patch."""
        return self.load_per_patch / (self.load_patch_length * self.thickness)

    @property
    def total_load(self) -> float:
        """Total vertical load applied to the structure, in newtons."""
        return self.load_per_patch * len(self.load_centres)

    # ------------------------------------------------------------- beam theory
    def beam_theory(self, n_samples: int = 20001) -> BeamTheory:
        """Euler-Bernoulli reference values (bending only, shear neglected).

        Deck loads and self weight are both included so that the result is directly
        comparable with the finite element solution.  Deflections are positive
        downwards.
        """
        x = np.linspace(0.0, self.span, n_samples)
        self_weight = self.density * self.gravity * self.thickness  # N/m

        # Reaction at the left bearing from global moment balance.
        reaction_left = self_weight * self.span / 2.0
        for start, stop in self.load_patch_bounds:
            reaction_left += self.load_per_patch * (self.span - 0.5 * (start + stop))
        reaction_left /= self.span

        # Bending moment: bearing reaction minus the loads already transferred.
        moment = reaction_left * x + self_weight * x * (self.span - x) / 2.0
        for start, stop in self.load_patch_bounds:
            intensity = self.load_per_patch / (stop - start)
            loaded = np.clip(x - start, 0.0, stop - start)
            moment -= intensity * (loaded * (x - start) - 0.5 * loaded**2)

        # Unit-load moment for a unit vertical load at midspan (virtual work).
        half = 0.5 * self.span
        unit_moment = np.where(x <= half, 0.5 * x, 0.5 * (self.span - x))
        compliance = 1.0 / (self.youngs_modulus * self.second_moment_of_area)
        midspan = np.trapezoid(moment * unit_moment, x) * compliance

        return BeamTheory(
            max_moment=float(np.max(np.abs(moment))),
            max_bending_stress=float(np.max(np.abs(moment)))
            * 0.5
            * self.depth
            / self.second_moment_of_area,
            midspan_deflection=float(midspan),
        )


@dataclass(frozen=True)
class QuadMesh:
    """Structured quad mesh of the rectangular bridge domain."""

    nx: int
    ny: int
    span: float
    depth: float

    @property
    def element_size(self) -> tuple[float, float]:
        return self.span / self.nx, self.depth / self.ny

    @property
    def nodes(self) -> np.ndarray:
        """Node coordinates, shape ``(n_nodes, 2)``, ordered row-major in ``y``."""
        dx, dy = self.element_size
        j, i = np.meshgrid(
            np.arange(self.ny + 1), np.arange(self.nx + 1), indexing="ij"
        )
        return np.stack([i.ravel() * dx, j.ravel() * dy], axis=-1)

    @property
    def elements(self) -> np.ndarray:
        """Element connectivity, shape ``(n_elements, 4)``, counter-clockwise."""
        j, i = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing="ij")
        n0 = j * (self.nx + 1) + i
        return np.stack(
            [n0, n0 + 1, n0 + self.nx + 2, n0 + self.nx + 1], axis=-1
        ).reshape(-1, 4)


def _shape_derivatives(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Derivatives of the four bilinear shape functions w.r.t. the reference axes."""
    d_xi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
    d_eta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])
    return d_xi, d_eta


def _gauss_points() -> tuple[np.ndarray, np.ndarray]:
    """2x2 Gauss points and weights on the reference element."""
    a = 1.0 / np.sqrt(3.0)
    points = np.array([[-a, -a], [a, -a], [a, a], [-a, a]])
    weights = np.ones(4)
    return points, weights


def _element_matrices(
    xy_e: np.ndarray, thickness: float
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Stiffness matrices, determinants and B-operators for a batch of elements.

    Parameters
    ----------
    xy_e : np.ndarray
        Element coordinates, shape ``(n_elements, 4, 2)``.
    thickness : float
        Out-of-plane thickness.

    Returns
    -------
    determinant : np.ndarray
        Jacobian determinant at each Gauss point, shape ``(n_elements, 4)``.
    operators : list of np.ndarray
        Strain-displacement matrices at each Gauss point, shape ``(n_elements, 3, 8)``.
    """
    gauss, _ = _gauss_points()
    determinant = np.empty((xy_e.shape[0], 4))
    operators = []
    for k, (xi, eta) in enumerate(gauss):
        d_xi, d_eta = _shape_derivatives(xi, eta)
        jac_xi = np.einsum("i,eij->ej", d_xi, xy_e)
        jac_eta = np.einsum("i,eij->ej", d_eta, xy_e)
        det = jac_xi[:, 0] * jac_eta[:, 1] - jac_xi[:, 1] * jac_eta[:, 0]
        if np.any(det <= 0.0):
            raise ValueError("Degenerate element: check node ordering.")
        determinant[:, k] = det
        inv = np.stack(
            [
                np.stack([jac_eta[:, 1] / det, -jac_xi[:, 1] / det], axis=-1),
                np.stack([-jac_eta[:, 0] / det, jac_xi[:, 0] / det], axis=-1),
            ],
            axis=-2,
        )
        # dN/dx, dN/dy from the reference derivatives and the inverse Jacobian.
        d_x = inv[:, 0, 0, None] * d_xi[None, :] + inv[:, 0, 1, None] * d_eta[None, :]
        d_y = inv[:, 1, 0, None] * d_xi[None, :] + inv[:, 1, 1, None] * d_eta[None, :]
        b = np.zeros((xy_e.shape[0], 3, 8))
        b[:, 0, 0::2] = d_x
        b[:, 1, 1::2] = d_y
        b[:, 2, 0::2] = d_y
        b[:, 2, 1::2] = d_x
        operators.append(b)
    return determinant, operators


@dataclass
class PlaneStressSolution:
    """Reference solution sampled on the finite element mesh."""

    bridge: SteelBridge
    mesh: QuadMesh
    nodes: np.ndarray
    elements: np.ndarray
    displacement: np.ndarray
    stress: np.ndarray
    reactions: np.ndarray

    # ------------------------------------------------------------- interpolation
    def _locate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Element indices and bilinear weights for arbitrary points."""
        dx, dy = self.mesh.element_size
        x = np.clip(points[:, 0], 0.0, self.bridge.span)
        y = np.clip(points[:, 1], 0.0, self.bridge.depth)
        i = np.clip(np.floor(x / dx).astype(int), 0, self.mesh.nx - 1)
        j = np.clip(np.floor(y / dy).astype(int), 0, self.mesh.ny - 1)
        r = (x - i * dx) / dx
        s = (y - j * dy) / dy
        weights = np.stack(
            [(1 - r) * (1 - s), r * (1 - s), r * s, (1 - r) * s], axis=-1
        )
        element = j * self.mesh.nx + i
        return element, weights

    def displacement_at(self, points: np.ndarray) -> np.ndarray:
        """Displacement ``(u_x, u_y)`` in metres at ``points`` of shape ``(N, 2)``."""
        element, weights = self._locate(np.asarray(points, dtype=float))
        corner = self.displacement[self.elements[element]]  # (N, 4, 2)
        return np.einsum("nk,nkc->nc", weights, corner)

    def stress_at(self, points: np.ndarray) -> np.ndarray:
        """Stress ``(sigma_xx, sigma_yy, sigma_xy)`` in Pa at ``points``."""
        element, weights = self._locate(np.asarray(points, dtype=float))
        corner = self.stress[self.elements[element]]
        return np.einsum("nk,nkc->nc", weights, corner)

    def von_mises_at(self, points: np.ndarray) -> np.ndarray:
        """Von Mises stress in Pa at ``points``."""
        return von_mises(self.stress_at(points))

    def grid(self, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
        """Evaluation grid of shape ``(nx, ny)`` spanning the whole domain."""
        x = np.linspace(0.0, self.bridge.span, nx)
        y = np.linspace(0.0, self.bridge.depth, ny)
        return np.meshgrid(x, y, indexing="ij")

    def grid_fields(
        self, nx: int, ny: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Displacement magnitude and von Mises stress on a ``(nx, ny)`` grid."""
        xx, yy = self.grid(nx, ny)
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        disp = self.displacement_at(points)
        sigma = self.stress_at(points)
        magnitude = np.linalg.norm(disp, axis=-1)
        return (
            magnitude.reshape(nx, ny),
            von_mises(sigma).reshape(nx, ny),
            disp.reshape(nx, ny, 2),
        )

    def midspan_deflection(self) -> float:
        """Downward deflection [m] at the middle of the bottom flange."""
        return float(
            -self.displacement_at(np.array([[0.5 * self.bridge.span, 0.0]]))[0, 1]
        )

    def max_von_mises(self) -> float:
        return float(np.max(von_mises(self.stress)))

    def sensor_measurements(
        self, n_sensors: int, seed: int = 0, margin: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sparse displacement readings, as a monitoring system would provide them.

        Returns
        -------
        points : np.ndarray
            Sensor locations, shape ``(n_sensors, 2)``.
        displacement : np.ndarray
            Measured ``(u_x, u_y)`` in metres, shape ``(n_sensors, 2)``.
        """
        rng = np.random.default_rng(seed)
        x = rng.uniform(
            margin * self.bridge.span, (1 - margin) * self.bridge.span, n_sensors
        )
        y = rng.uniform(
            margin * self.bridge.depth, (1 - margin) * self.bridge.depth, n_sensors
        )
        points = np.stack([x, y], axis=-1)
        return points, self.displacement_at(points)


def von_mises(stress: np.ndarray) -> np.ndarray:
    """Plane-stress von Mises equivalent stress."""
    s_xx, s_yy, s_xy = stress[..., 0], stress[..., 1], stress[..., 2]
    return np.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3.0 * s_xy**2)


def solve_plane_stress(
    bridge: SteelBridge, mesh: QuadMesh, *, with_self_weight: bool = True
) -> PlaneStressSolution:
    """Assemble and solve the plane-stress finite element system.

    Parameters
    ----------
    bridge : SteelBridge
        Geometry, material and loading.
    mesh : QuadMesh
        Structured quadrilateral mesh of the domain.
    with_self_weight : bool
        Include the ``rho * g`` body force.

    Returns
    -------
    PlaneStressSolution
        Nodal displacement, nodal stress and support reactions.
    """
    nodes = mesh.nodes
    elements = mesh.elements
    n_nodes = nodes.shape[0]
    n_dofs = 2 * n_nodes
    dofs = 2 * elements[:, :, None] + np.arange(2)[None, None, :]
    dofs = dofs.reshape(elements.shape[0], 8)

    xy_e = nodes[elements]
    determinant, operators = _element_matrices(xy_e, bridge.thickness)
    stiffness = bridge.constitutive_matrix

    n_elements = elements.shape[0]
    ke = np.zeros((n_elements, 8, 8))
    _, weights = _gauss_points()
    for k, b in enumerate(operators):
        ke += (
            np.einsum("eki,kl,elj->eij", b, stiffness, b)
            * (determinant[:, k] * weights[k] * bridge.thickness)[:, None, None]
        )

    rows = np.repeat(dofs, 8, axis=1).ravel()
    cols = np.tile(dofs, (1, 8)).ravel()
    matrix = sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()

    force = np.zeros(n_dofs)

    # Self weight: consistent nodal loads, downward.
    if with_self_weight:
        dx, dy = mesh.element_size
        element_area = dx * dy * np.ones(n_elements)
        weight_per_node = (
            -bridge.density * bridge.gravity * element_area * bridge.thickness / 4.0
        )
        for corner in range(4):
            np.add.at(force, 2 * elements[:, corner] + 1, weight_per_node)

    # Wheel patches: consistent nodal loads on the deck edge (top row of elements).
    top_row = elements.reshape(mesh.ny, mesh.nx, 4)[-1].reshape(-1, 4)
    top_left = top_row[:, 3]
    top_right = top_row[:, 2]
    x_left = nodes[top_left, 0]
    x_right = nodes[top_right, 0]
    element_length = x_right - x_left
    for start, stop in bridge.load_patch_bounds:
        loaded_start = np.clip(start, x_left, x_right)
        loaded_stop = np.clip(stop, x_left, x_right)
        covered = loaded_stop > loaded_start
        # Consistent nodal loads of a partially loaded bilinear edge:
        # f_left = q t \int (1 - s/l) ds, f_right = q t \int (s/l) ds.
        a = np.where(covered, loaded_start - x_left, 0.0)
        b = np.where(covered, loaded_stop - x_left, 0.0)
        left_fraction = (b - a) - (b**2 - a**2) / (2.0 * element_length)
        right_fraction = (b**2 - a**2) / (2.0 * element_length)
        scale = bridge.patch_traction * bridge.thickness
        np.add.at(force, 2 * top_left + 1, -scale * left_fraction)
        np.add.at(force, 2 * top_right + 1, -scale * right_fraction)

    # Bearings.
    fixed = np.zeros(n_dofs, dtype=bool)
    tolerance = 1e-9 * max(bridge.span, bridge.depth)
    on_left = np.isclose(nodes[:, 0], 0.0, atol=tolerance)
    on_right = np.isclose(nodes[:, 0], bridge.span, atol=tolerance)
    in_pad = nodes[:, 1] <= bridge.bearing_length + tolerance
    left_nodes = np.where(on_left & in_pad)[0]
    right_nodes = np.where(on_right & in_pad)[0]
    fixed[2 * left_nodes] = True
    fixed[2 * left_nodes + 1] = True
    fixed[2 * right_nodes + 1] = True

    free = ~fixed
    reduced = matrix[free][:, free].tocsc()
    solution = np.zeros(n_dofs)
    solution[free] = spsolve(reduced, force[free])

    reactions = matrix @ solution - force

    # Stress recovery: Gauss point stresses -> extrapolated nodal values -> average.
    displacement_e = solution[dofs]
    gauss_values = np.zeros((n_elements, 4, 3))
    for k, b in enumerate(operators):
        strain = np.einsum("eki,ei->ek", b, displacement_e)
        gauss_values[:, k, :] = strain @ stiffness.T
    extrapolation = _node_from_gauss_matrix()
    nodal_per_element = np.einsum("kn,ekc->enc", extrapolation, gauss_values)

    stress_sum = np.zeros((n_nodes, 3))
    counts = np.zeros(n_nodes)
    for corner in range(4):
        np.add.at(stress_sum, elements[:, corner], nodal_per_element[:, corner])
        np.add.at(counts, elements[:, corner], 1.0)
    stress = stress_sum / counts[:, None]

    return PlaneStressSolution(
        bridge=bridge,
        mesh=mesh,
        nodes=nodes,
        elements=elements,
        displacement=solution.reshape(n_nodes, 2),
        stress=stress,
        reactions=reactions.reshape(n_nodes, 2),
    )


def _node_from_gauss_matrix() -> np.ndarray:
    """Matrix mapping the four Gauss-point values to the four corner values."""
    gauss, _ = _gauss_points()
    xi = gauss[:, 0][:, None]
    eta = gauss[:, 1][:, None]
    node_xi = np.array([-1.0, 1.0, 1.0, -1.0])[None, :]
    node_eta = np.array([-1.0, -1.0, 1.0, 1.0])[None, :]
    # Shape functions evaluated at the Gauss points, shape (4 gauss, 4 nodes).
    shape = 0.25 * (1.0 + xi * node_xi) * (1.0 + eta * node_eta)
    return np.linalg.inv(shape)


def convergence_study(
    bridge: SteelBridge, resolutions: Sequence[tuple[int, int]]
) -> list[dict[str, float]]:
    """Midspan deflection and peak stress for a sequence of mesh resolutions."""
    theory = bridge.beam_theory()
    results = []
    for nx, ny in resolutions:
        solution = solve_plane_stress(
            bridge, QuadMesh(nx=nx, ny=ny, span=bridge.span, depth=bridge.depth)
        )
        results.append(
            {
                "nx": nx,
                "ny": ny,
                "elements": nx * ny,
                "midspan_deflection": solution.midspan_deflection(),
                "deflection_error": abs(
                    solution.midspan_deflection() / theory.midspan_deflection - 1.0
                ),
                "max_von_mises": solution.max_von_mises(),
            }
        )
    return results
