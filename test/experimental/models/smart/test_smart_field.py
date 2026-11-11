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

"""SMART coordinate-frame regression."""

import pytest
import torch

from physicsnemo import Module
from physicsnemo.experimental.models.smart import SMART


def _model(**overrides):
    options = dict(
        field="surface",
        out_dim=4,
        query_feature_channels=3,
        hidden_dim=24,
        num_heads=4,
        num_layers=2,
        num_latents=8,
        geometry_subsample=16,
        eval_chunk_size=3,
    )
    options.update(overrides)
    return SMART(**options)


def _inputs():
    return dict(
        geometry=torch.randn(1, 32, 3),
        queries=torch.randn(1, 7, 3),
        query_features=torch.randn(1, 7, 3),
        parameters=None,
        frame_center=torch.randn(1, 1, 3),
    )


@pytest.mark.parametrize("field", ["surface", "volume"])
@pytest.mark.parametrize("center_on_geometry", [False, True])
def test_coordinate_frame_preserves_inputs_and_checkpoint(
    field, center_on_geometry, tmp_path
):
    """Normalize geometry and queries consistently without mutating their inputs."""
    model = _model(
        field=field,
        coordinate_scale=(12.0, 4.5, 3.25),
        center_on_geometry=center_on_geometry,
    )
    reference = _model(field=field, coordinate_scale=(1.0, 1.0, 1.0))
    reference.load_state_dict(model.state_dict())
    inputs = _inputs()
    original = {k: v.clone() for k, v in inputs.items() if v is not None}
    scale = inputs["geometry"].new_tensor(model.coordinate_scale)
    normalized = {
        **inputs,
        "geometry": (inputs["geometry"] - inputs["frame_center"]) / scale,
        "queries": (inputs["queries"] - inputs["frame_center"]) / scale,
        "frame_center": torch.zeros_like(inputs["frame_center"]),
    }
    for training in (True, False):
        model.train(training)
        reference.train(training)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            actual = model(**inputs)
            torch.manual_seed(42)
            expected = reference(**normalized)
        torch.testing.assert_close(actual, expected)
        for key in original:
            torch.testing.assert_close(inputs[key], original[key], rtol=0, atol=0)
    path = tmp_path / "field.mdlus"
    model.save(path)
    restored = Module.from_checkpoint(path).eval()
    assert restored.coordinate_scale == model.coordinate_scale
    assert restored.center_on_geometry == model.center_on_geometry
    torch.testing.assert_close(restored(**inputs), model(**inputs))


@pytest.mark.parametrize("field", ["surface", "volume"])
def test_geometry_center_matches_explicit_frame(field):
    """Automatic geometry centering matches an explicit per-case bounding box."""
    model = _model(
        field=field, center_on_geometry=True, coordinate_scale=(2.4, 0.9, 0.65)
    ).eval()
    geometry = torch.randn(2, 32, 3) + torch.tensor(
        [[[10.0, 0.0, 0.0]], [[0.0, 20.0, 0.0]]]
    )
    queries = torch.randn(2, 7, 3)
    features = torch.randn(2, 7, 3)
    center = (geometry.amin(dim=1) + geometry.amax(dim=1)) * 0.5
    actual = model(geometry, queries, query_features=features)
    expected = model(geometry, queries, query_features=features, frame_center=center)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
