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

import numpy as np
import pytest
import torch
from sympy import Function, Number, Symbol

from physicsnemo.sym.eq.gradients import _compute_stencil3d
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

# ---------------------------------------------------------------------------
# Inline Navier-Stokes PDE (replaces ported NavierStokes class)
# ---------------------------------------------------------------------------


class NavierStokes(PDE):
    """Incompressible Navier-Stokes (steady, 3-D) for testing."""

    def __init__(self, nu=0.01, rho=1.0, dim=3, time=False):
        self.dim = dim
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        input_variables = {"x": x, "y": y, "z": z}
        if dim < 3:
            input_variables.pop("z")
        if dim < 2:
            input_variables.pop("y")

        u = Function("u")(*input_variables.values())
        v = Function("v")(*input_variables.values())
        w = Function("w")(*input_variables.values()) if dim == 3 else Number(0)
        p = Function("p")(*input_variables.values())

        nu = Number(nu)
        rho = Number(rho)

        self.equations = {}
        self.equations["continuity"] = (
            u.diff(x) + (v.diff(y) if dim >= 2 else 0) + (w.diff(z) if dim == 3 else 0)
        )
        self.equations["momentum_x"] = (
            u * u.diff(x)
            + (v * u.diff(y) if dim >= 2 else 0)
            + (w * u.diff(z) if dim == 3 else 0)
            + (1 / rho) * p.diff(x)
            - nu * u.diff(x, 2)
            - (nu * u.diff(y, 2) if dim >= 2 else 0)
            - (nu * u.diff(z, 2) if dim == 3 else 0)
        )


# ---------------------------------------------------------------------------
# Model fixture
# ---------------------------------------------------------------------------


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        u = (
            torch.sin(1 * x[:, 0:1])
            + torch.sin(8 * x[:, 1:2])
            + torch.sin(4 * x[:, 2:3])
        )
        v = (
            torch.sin(8 * x[:, 0:1])
            + torch.sin(2 * x[:, 1:2])
            + torch.sin(1 * x[:, 2:3])
        )
        w = (
            torch.sin(2 * x[:, 0:1])
            + torch.sin(2 * x[:, 1:2])
            + torch.sin(9 * x[:, 2:3])
        )
        p = (
            torch.sin(1 * x[:, 0:1])
            + torch.sin(1 * x[:, 1:2])
            + torch.sin(1 * x[:, 2:3])
        )
        return torch.cat([u, v, w, p], dim=1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def general_setup(request):
    device = request.param
    steps = 100
    x = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True).to(device)
    y = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True).to(device)
    z = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True).to(device)

    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    coords = torch.stack([xx, yy, zz], dim=0).unsqueeze(0)
    coords_unstructured = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    model = Model().to(device)

    u = (
        torch.sin(1 * coords[:, 0:1])
        + torch.sin(8 * coords[:, 1:2])
        + torch.sin(4 * coords[:, 2:3])
    )
    v = (
        torch.sin(8 * coords[:, 0:1])
        + torch.sin(2 * coords[:, 1:2])
        + torch.sin(1 * coords[:, 2:3])
    )
    w = (
        torch.sin(2 * coords[:, 0:1])
        + torch.sin(2 * coords[:, 1:2])
        + torch.sin(9 * coords[:, 2:3])
    )

    p__x = torch.cos(1 * coords[:, 0:1])
    u__x = torch.cos(1 * coords[:, 0:1])
    u__y = 8 * torch.cos(8 * coords[:, 1:2])
    u__z = 4 * torch.cos(4 * coords[:, 2:3])
    v__y = 2 * torch.cos(2 * coords[:, 1:2])
    w__z = 9 * torch.cos(9 * coords[:, 2:3])
    u__x__x = -1 * torch.sin(1 * coords[:, 0:1])
    u__y__y = -64 * torch.sin(8 * coords[:, 1:2])
    u__z__z = -16 * torch.sin(4 * coords[:, 2:3])

    true_cont = u__x + v__y + w__z
    true_mom_x = (
        u * u__x
        + v * u__y
        + w * u__z
        + p__x
        - 0.01 * u__x__x
        - 0.01 * u__y__y
        - 0.01 * u__z__z
    )

    residuals_analytical = {"continuity": true_cont, "momentum_x": true_mom_x}
    return coords, coords_unstructured, residuals_analytical, model


@pytest.fixture
def least_squares_setup(request):
    device = request.param
    steps = 100
    x = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True).to(device)
    y = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True).to(device)
    z = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True).to(device)

    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    coords = torch.stack([xx, yy, zz], dim=0).unsqueeze(0)
    coords_unstructured = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    indices = torch.arange(steps, device=device)
    i, j, k = torch.meshgrid(indices, indices, indices, indexing="ij")
    i = i.flatten()
    j = j.flatten()
    k = k.flatten()
    index = i * steps * steps + j * steps + k

    edges = []
    if steps > 1:
        edges.append(
            torch.stack([index[: -steps * steps], index[steps * steps :]], dim=1)
        )
        edges.append(torch.stack([index[:-steps], index[steps:]], dim=1))
        edges.append(torch.stack([index[:-1], index[1:]], dim=1))
    edges = torch.cat(edges).to(device)

    node_ids = torch.arange(coords_unstructured.size(0)).reshape(-1, 1).to(device)

    model = Model().to(device)

    u = (
        torch.sin(1 * coords[:, 0:1])
        + torch.sin(8 * coords[:, 1:2])
        + torch.sin(4 * coords[:, 2:3])
    )
    v = (
        torch.sin(8 * coords[:, 0:1])
        + torch.sin(2 * coords[:, 1:2])
        + torch.sin(1 * coords[:, 2:3])
    )
    w = (
        torch.sin(2 * coords[:, 0:1])
        + torch.sin(2 * coords[:, 1:2])
        + torch.sin(9 * coords[:, 2:3])
    )

    p__x = torch.cos(1 * coords[:, 0:1])
    u__x = torch.cos(1 * coords[:, 0:1])
    u__y = 8 * torch.cos(8 * coords[:, 1:2])
    u__z = 4 * torch.cos(4 * coords[:, 2:3])
    v__y = 2 * torch.cos(2 * coords[:, 1:2])
    w__z = 9 * torch.cos(9 * coords[:, 2:3])
    u__x__x = -1 * torch.sin(1 * coords[:, 0:1])
    u__y__y = -64 * torch.sin(8 * coords[:, 1:2])
    u__z__z = -16 * torch.sin(4 * coords[:, 2:3])

    true_cont = u__x + v__y + w__z
    true_mom_x = (
        u * u__x
        + v * u__y
        + w * u__z
        + p__x
        - 0.01 * u__x__x
        - 0.01 * u__y__y
        - 0.01 * u__z__z
    )

    residuals_analytical = {"continuity": true_cont, "momentum_x": true_mom_x}
    return coords, coords_unstructured, residuals_analytical, model, node_ids, edges


# ---------------------------------------------------------------------------
# Tests — same structure and thresholds as physicsnemo-sym originals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("general_setup", ["cuda"], indirect=True)
def test_residuals_autodiff(general_setup):
    coords, coords_unstructured, residuals_analytical, model = general_setup
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    phy_informer = PhysicsInformer(
        required_outputs=["continuity", "momentum_x"],
        equations=ns,
        grad_method="autodiff",
        device=coords.device,
    )
    pred_outvar = model(coords_unstructured)
    residuals = phy_informer.forward(
        {
            "coordinates": coords_unstructured,
            "u": pred_outvar[:, 0:1],
            "v": pred_outvar[:, 1:2],
            "w": pred_outvar[:, 2:3],
            "p": pred_outvar[:, 3:4],
        },
    )

    pad = 2
    for key in residuals_analytical.keys():
        error = torch.mean(
            torch.abs(
                residuals_analytical[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
                - residuals[key].reshape(100, 100, 100)[pad:-pad, pad:-pad, pad:-pad]
            )
        )
        assert error < 0.5, f"Autodiff gradient error too high for {key}: {error}"


@pytest.mark.parametrize("general_setup", ["cuda"], indirect=True)
def test_residuals_meshless_fd(general_setup):
    coords, coords_unstructured, residuals_analytical, model = general_setup
    po_posx, po_negx, po_posy, po_negy, po_posz, po_negz = _compute_stencil3d(
        coords_unstructured, model, dx=0.001
    )

    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    phy_informer = PhysicsInformer(
        required_outputs=["continuity", "momentum_x"],
        equations=ns,
        grad_method="meshless_finite_difference",
        fd_dx=0.001,
        device=coords.device,
    )
    pred_outvar = model(coords_unstructured)
    var_names = ["u", "v", "w", "p"]
    stencil_map = {
        "x::1": po_posx,
        "x::-1": po_negx,
        "y::1": po_posy,
        "y::-1": po_negy,
        "z::1": po_posz,
        "z::-1": po_negz,
    }
    inputs = {name: pred_outvar[:, i : i + 1] for i, name in enumerate(var_names)}
    for suffix, stencil in stencil_map.items():
        for i, name in enumerate(var_names):
            inputs[f"{name}>>{suffix}"] = stencil[:, i : i + 1]
    residuals = phy_informer.forward(inputs)

    pad = 2
    for key in residuals_analytical.keys():
        error = torch.mean(
            torch.abs(
                residuals_analytical[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
                - residuals[key].reshape(100, 100, 100)[pad:-pad, pad:-pad, pad:-pad]
            )
        )
        assert error < 0.5, f"Meshless FD gradient error too high for {key}: {error}"


@pytest.mark.parametrize("general_setup", ["cuda"], indirect=True)
def test_residuals_finite_difference(general_setup):
    coords, coords_unstructured, residuals_analytical, model = general_setup
    steps = 100
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    phy_informer = PhysicsInformer(
        required_outputs=["continuity", "momentum_x"],
        equations=ns,
        grad_method="finite_difference",
        fd_dx=(2 * np.pi / steps),
        device=coords.device,
    )
    pred_outvar = model(coords)
    residuals = phy_informer.forward(
        {
            "u": pred_outvar[:, 0:1],
            "v": pred_outvar[:, 1:2],
            "w": pred_outvar[:, 2:3],
            "p": pred_outvar[:, 3:4],
        },
    )

    pad = 2
    for key in residuals_analytical.keys():
        error = torch.mean(
            torch.abs(
                residuals_analytical[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
                - residuals[key].reshape(100, 100, 100)[pad:-pad, pad:-pad, pad:-pad]
            )
        )
        assert error < 0.5, (
            f"Finite Difference gradient error too high for {key}: {error}"
        )


@pytest.mark.parametrize("general_setup", ["cuda"], indirect=True)
def test_residuals_spectral(general_setup):
    coords, coords_unstructured, residuals_analytical, model = general_setup
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    phy_informer = PhysicsInformer(
        required_outputs=["continuity", "momentum_x"],
        equations=ns,
        grad_method="spectral",
        bounds=[2 * np.pi, 2 * np.pi, 2 * np.pi],
        device=coords.device,
    )
    pred_outvar = model(coords)
    residuals = phy_informer.forward(
        {
            "u": pred_outvar[:, 0:1],
            "v": pred_outvar[:, 1:2],
            "w": pred_outvar[:, 2:3],
            "p": pred_outvar[:, 3:4],
        },
    )

    pad = 2
    for key in residuals_analytical.keys():
        error = torch.mean(
            torch.abs(
                residuals_analytical[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
                - residuals[key].reshape(100, 100, 100)[pad:-pad, pad:-pad, pad:-pad]
            )
        )
        assert error < 0.5, f"Spectral gradient error too high for {key}: {error}"


@pytest.mark.parametrize("least_squares_setup", ["cuda"], indirect=True)
def test_residuals_least_squares(least_squares_setup):
    (
        coords,
        coords_unstructured,
        residuals_analytical,
        model,
        node_ids,
        edges,
    ) = least_squares_setup
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    phy_informer = PhysicsInformer(
        required_outputs=["continuity", "momentum_x"],
        equations=ns,
        grad_method="least_squares",
        device=coords.device,
    )
    pred_outvar = model(coords_unstructured)
    residuals = phy_informer.forward(
        {
            "coordinates": coords_unstructured,
            "nodes": node_ids,
            "edges": edges,
            "u": pred_outvar[:, 0:1],
            "v": pred_outvar[:, 1:2],
            "w": pred_outvar[:, 2:3],
            "p": pred_outvar[:, 3:4],
        },
    )

    pad = 2
    for key in residuals_analytical.keys():
        error = torch.mean(
            torch.abs(
                residuals_analytical[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
                - residuals[key].reshape(100, 100, 100)[pad:-pad, pad:-pad, pad:-pad]
            )
        )
        assert error < 0.5, f"Least Squares gradient error too high for {key}: {error}"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


def test_physics_informer_invalid_output():
    """Requesting a non-existent equation output raises ValueError."""
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    with pytest.raises(ValueError, match="not in equation outputs"):
        pi = PhysicsInformer(
            required_outputs=["nonexistent_equation"],
            equations=ns,
            grad_method="autodiff",
        )
        _ = pi.required_inputs


def test_physics_informer_required_inputs_autodiff():
    """Autodiff method requires 'coordinates' in the input list."""
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    pi = PhysicsInformer(
        required_outputs=["continuity"],
        equations=ns,
        grad_method="autodiff",
    )
    assert "coordinates" in pi.required_inputs


def test_physics_informer_required_inputs_least_squares():
    """Least-squares method requires connectivity-related inputs."""
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    pi = PhysicsInformer(
        required_outputs=["continuity"],
        equations=ns,
        grad_method="least_squares",
        compute_connectivity=True,
    )
    inputs = pi.required_inputs
    assert "coordinates" in inputs
    assert "nodes" in inputs
    assert "edges" in inputs


def test_pde_pprint():
    """PDE.pprint() runs without error."""
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    ns.pprint()


def test_pde_make_computations():
    """make_computations() produces one Computation per equation."""
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    comps = ns.make_computations()
    assert len(comps) == len(ns.equations)
    for comp in comps:
        assert comp.outputs[0] in ns.equations
