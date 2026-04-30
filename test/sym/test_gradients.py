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

from physicsnemo.sym.eq.gradients import (
    GradientCalculator,
    _compute_stencil3d,
    compute_connectivity_tensor,
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            torch.sin(x[:, 0:1]) * torch.sin(8 * x[:, 1:2]) * torch.sin(4 * x[:, 2:3])
        )


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

    grad_u_analytical = {
        "u__x": torch.cos(coords[:, 0:1])
        * torch.sin(8 * coords[:, 1:2])
        * torch.sin(4 * coords[:, 2:3]),
        "u__y": torch.sin(coords[:, 0:1])
        * 8
        * torch.cos(8 * coords[:, 1:2])
        * torch.sin(4 * coords[:, 2:3]),
        "u__z": torch.sin(coords[:, 0:1])
        * torch.sin(8 * coords[:, 1:2])
        * 4
        * torch.cos(4 * coords[:, 2:3]),
    }

    return coords, coords_unstructured, grad_u_analytical, model


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

    model = Model().to(device)

    grad_u_analytical = {
        "u__x": torch.cos(coords[:, 0:1])
        * torch.sin(8 * coords[:, 1:2])
        * torch.sin(4 * coords[:, 2:3]),
        "u__y": torch.sin(coords[:, 0:1])
        * 8
        * torch.cos(8 * coords[:, 1:2])
        * torch.sin(4 * coords[:, 2:3]),
        "u__z": torch.sin(coords[:, 0:1])
        * torch.sin(8 * coords[:, 1:2])
        * 4
        * torch.cos(4 * coords[:, 2:3]),
    }

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
    connectivity_tensor = compute_connectivity_tensor(node_ids, edges)

    return (
        coords,
        coords_unstructured,
        grad_u_analytical,
        model,
        node_ids,
        edges,
        connectivity_tensor,
    )


@pytest.mark.parametrize("general_setup", ["cuda"], indirect=True)
def test_gradients_autodiff(general_setup):
    coords, coords_unstructured, grad_u_analytical, model = general_setup
    grad_calc = GradientCalculator(device=coords.device)

    input_dict = {"coordinates": coords_unstructured, "u": model(coords_unstructured)}
    grad_u_autodiff = grad_calc.compute_gradients(
        input_dict, method_name="autodiff", invar="u"
    )

    pad = 2
    for key in grad_u_analytical.keys():
        error = torch.mean(
            torch.abs(
                grad_u_analytical[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
                - grad_u_autodiff[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
            )
        )
        assert error < 0.2, f"Autodiff gradient error too high for {key}: {error}"


@pytest.mark.parametrize("general_setup", ["cuda"], indirect=True)
def test_gradients_meshless_fd(general_setup):
    coords, coords_unstructured, grad_u_analytical, model = general_setup
    grad_calc = GradientCalculator(device=coords.device)

    po_posx, po_negx, po_posy, po_negy, po_posz, po_negz = _compute_stencil3d(
        coords_unstructured, model, dx=0.001
    )
    stencil_map = {
        "x::1": po_posx,
        "x::-1": po_negx,
        "y::1": po_posy,
        "y::-1": po_negy,
        "z::1": po_posz,
        "z::-1": po_negz,
    }
    input_dict = {"u": model(coords_unstructured)}
    input_dict.update({f"u>>{k}": v for k, v in stencil_map.items()})
    grads_u_meshless_fd = grad_calc.compute_gradients(
        input_dict, method_name="meshless_finite_difference", invar="u", dx=0.001
    )

    pad = 2
    for key in grad_u_analytical.keys():
        error = torch.mean(
            torch.abs(
                grad_u_analytical[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
                - grads_u_meshless_fd[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
            )
        )
        assert error < 0.2, f"Meshless FD gradient error too high for {key}: {error}"


@pytest.mark.parametrize("general_setup", ["cuda"], indirect=True)
def test_gradients_finite_difference(general_setup):
    coords, coords_unstructured, grad_u_analytical, model = general_setup
    grad_calc = GradientCalculator(device=coords.device)

    input_dict = {"u": model(coords)}
    grads_u_fd = grad_calc.compute_gradients(
        input_dict, method_name="finite_difference", invar="u", dx=2 * np.pi / 100
    )

    pad = 2
    for key in grad_u_analytical.keys():
        error = torch.mean(
            torch.abs(
                grad_u_analytical[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
                - grads_u_fd[key].reshape(100, 100, 100)[pad:-pad, pad:-pad, pad:-pad]
            )
        )
        assert error < 0.2, (
            f"Finite Difference gradient error too high for {key}: {error}"
        )


@pytest.mark.parametrize("general_setup", ["cuda"], indirect=True)
def test_gradients_spectral(general_setup):
    coords, coords_unstructured, grad_u_analytical, model = general_setup
    grad_calc = GradientCalculator(device=coords.device)

    input_dict = {"u": model(coords)}
    grads_u_spectral = grad_calc.compute_gradients(
        input_dict,
        method_name="spectral",
        invar="u",
        ell=[2 * np.pi, 2 * np.pi, 2 * np.pi],
    )

    pad = 2
    for key in grad_u_analytical.keys():
        error = torch.mean(
            torch.abs(
                grad_u_analytical[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
                - grads_u_spectral[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
            )
        )
        assert error < 0.2, f"Spectral gradient error too high for {key}: {error}"


@pytest.mark.parametrize("least_squares_setup", ["cuda"], indirect=True)
def test_gradients_least_squares(least_squares_setup):
    (
        coords,
        coords_unstructured,
        grad_u_analytical,
        model,
        node_ids,
        edges,
        connectivity_tensor,
    ) = least_squares_setup
    grad_calc = GradientCalculator(device=coords.device)

    input_dict = {
        "u": model(coords_unstructured),
        "coordinates": coords_unstructured,
        "nodes": node_ids,
        "edges": edges,
        "connectivity_tensor": connectivity_tensor,
    }
    grads_u_ls = grad_calc.compute_gradients(
        input_dict, method_name="least_squares", invar="u"
    )

    pad = 2
    for key in grad_u_analytical.keys():
        error = torch.mean(
            torch.abs(
                grad_u_analytical[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
                - grads_u_ls[key].reshape(100, 100, 100)[pad:-pad, pad:-pad, pad:-pad]
            )
        )
        assert error < 0.2, f"Least Squares gradient error too high for {key}: {error}"
