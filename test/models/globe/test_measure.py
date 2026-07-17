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

"""GLOBE consumption of the effective cell measure.

GLOBE weights its boundary integrals by the effective cell measure
``cell_areas * measure_weights`` (see
:mod:`physicsnemo.mesh.calculus.measure`), and compounds one such
measure-weighted sum per integral stage
(``n_communication_hyperlayers + 1`` in total), so an incorrect measure is
amplified to that power.  These tests pin the consumption contract:

1. Measure weights on a boundary mesh must be equivalent to scaling that
   mesh's cell areas directly (two expressions of the same effective
   measure).
2. Explicit unit weights must reproduce the no-weights output exactly.
3. Non-unit weights must actually change the output (guards against a
   refactor silently dropping the weights, e.g. reading them after
   enrichment has nested ``cell_data`` under the "physical" namespace).
"""

import torch

from physicsnemo.experimental.models.globe.model import GLOBE
from physicsnemo.mesh.calculus.measure import compose_measure_weights
from physicsnemo.mesh.primitives.procedural import lumpy_sphere

SEED = 7


def _make_model(device: torch.device) -> GLOBE:
    torch.manual_seed(SEED)
    model = GLOBE(
        n_spatial_dims=3,
        output_field_ranks={"C_p": 0, "C_f": 1},
        boundary_source_data_ranks={"vehicle": {}, "floor": {}},
        reference_length_names=["L_ref"],
        reference_area=1.0,
        n_communication_hyperlayers=2,
        hidden_layer_sizes=(16, 16),
        n_latent_scalars=2,
        n_latent_vectors=1,
        n_spherical_harmonics=2,
        theta=2.0,
        leaf_size=4,
    ).to(device)
    model.eval()
    return model


def _make_inputs(device: torch.device) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    return {
        "prediction_points": torch.randn(20, 3, generator=generator).to(device),
        "boundary_meshes": {
            "vehicle": lumpy_sphere.load(subdivisions=1, device=device),
            "floor": lumpy_sphere.load(subdivisions=0, device=device),
        },
        "reference_lengths": {
            "L_ref": torch.tensor(1.0, dtype=torch.float32, device=device)
        },
    }


def _forward(model: GLOBE, kwargs: dict) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        out = model(**kwargs)
    return {k: v.detach() for k, v in out.point_data.items()}


def test_weights_equivalent_to_scaled_areas():
    """weights w on a mesh == scaling that mesh's area cache by w."""
    device = torch.device("cpu")
    model = _make_model(device)
    gen = torch.Generator().manual_seed(SEED)

    kwargs_weighted = _make_inputs(device)
    kwargs_scaled = _make_inputs(device)
    for name in ("vehicle", "floor"):
        mesh_w = kwargs_weighted["boundary_meshes"][name]
        w = torch.rand(mesh_w.n_cells, generator=gen) + 0.5
        compose_measure_weights(mesh_w, w)

        mesh_s = kwargs_scaled["boundary_meshes"][name]
        mesh_s._cache["cell", "areas"] = mesh_s.cell_areas * w

    out_weighted = _forward(model, kwargs_weighted)
    out_scaled = _forward(model, kwargs_scaled)
    for key in out_weighted:
        torch.testing.assert_close(out_weighted[key], out_scaled[key])


def test_unit_weights_match_no_weights():
    """Explicit ones must be indistinguishable from absent weights."""
    device = torch.device("cpu")
    model = _make_model(device)

    kwargs_plain = _make_inputs(device)
    kwargs_ones = _make_inputs(device)
    for name in ("vehicle", "floor"):
        mesh = kwargs_ones["boundary_meshes"][name]
        compose_measure_weights(mesh, torch.ones(mesh.n_cells))

    out_plain = _forward(model, kwargs_plain)
    out_ones = _forward(model, kwargs_ones)
    for key in out_plain:
        torch.testing.assert_close(out_ones[key], out_plain[key], rtol=0.0, atol=0.0)


def test_weights_change_output():
    """Non-unit weights must reach the integral stages, not be dropped.

    Guards the enrichment-namespace pitfall: GLOBE nests ``cell_data``
    under "physical" before evaluating layers, so weights read after
    enrichment would silently fall back to ones.
    """
    device = torch.device("cpu")
    model = _make_model(device)

    kwargs_plain = _make_inputs(device)
    kwargs_weighted = _make_inputs(device)
    for name in ("vehicle", "floor"):
        mesh = kwargs_weighted["boundary_meshes"][name]
        compose_measure_weights(mesh, torch.full((mesh.n_cells,), 3.0))

    out_plain = _forward(model, kwargs_plain)
    out_weighted = _forward(model, kwargs_weighted)
    assert any(
        not torch.allclose(out_weighted[k], out_plain[k], rtol=1e-3, atol=1e-6)
        for k in out_plain
    )
