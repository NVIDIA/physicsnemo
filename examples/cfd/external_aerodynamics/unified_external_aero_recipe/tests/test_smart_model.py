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

"""SMART tensor adapter input and serialization contracts."""

import pytest
import torch
from smart_model import SMARTField

from physicsnemo import Module


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
    return SMARTField(**options)


def _inputs():
    return dict(
        geometry=torch.randn(1, 32, 3),
        queries=torch.randn(1, 7, 3),
        query_features=torch.randn(1, 7, 3),
        parameters=torch.randn(1, 1, 2),
        frame_center=torch.randn(1, 1, 3),
        geometry_seed=torch.tensor([[[42]]]),
    )


@pytest.mark.parametrize(
    "invalid",
    [
        {"geometry": torch.tensor(0.0)},
        {"queries": torch.randn(2, 7, 3)},
        {"query_features": torch.randn(1, 7, 4)},
        {"geometry_seed": torch.tensor(1.5)},
        {"geometry_seed": torch.tensor(True)},
        {"parameters": torch.randn(3, 1, 1)},
    ],
)
def test_invalid_inputs_fail_before_geometry_encoding(invalid, monkeypatch):
    """Reject malformed batches before the potentially expensive geometry pass."""
    model = _model()
    inputs = _inputs()
    inputs.update(invalid)

    def unexpected_encode(*args, **kwargs):
        pytest.fail("Geometry must not be encoded before validating inputs")

    monkeypatch.setattr(model, "encode_geometry", unexpected_encode)
    with pytest.raises(ValueError):
        model(**inputs)


@pytest.mark.parametrize("field", ["surface", "volume"])
def test_optional_features_and_conditions_checkpoint(field, tmp_path):
    """Disabled features/conditions survive chunking and checkpoint restoration."""
    model = _model(field=field, query_feature_channels=0, parameter_channels=0).eval()
    inputs = _inputs()
    inputs.update(query_features=None, parameters=None)
    with torch.no_grad():
        expected = model(**inputs)
        model.eval_chunk_size = 20
        torch.testing.assert_close(model(**inputs), expected)
    path = tmp_path / "adapter.mdlus"
    model.save(path)
    restored = Module.from_checkpoint(path).eval()
    assert restored.parameter_channels == 0
    assert restored.field == field
    with torch.no_grad():
        torch.testing.assert_close(restored(**inputs), expected)


@pytest.mark.parametrize("field", ["surface", "volume"])
def test_coordinate_frame_preserves_inputs_and_checkpoint(field, tmp_path):
    """Normalize both clouds once and retain the frame across checkpoint reload."""
    model = _model(field=field, coordinate_scale=(12.0, 4.5, 3.25))
    reference = _model(field=field, coordinate_scale=(1.0, 1.0, 1.0))
    reference.load_state_dict(model.state_dict())
    inputs = _inputs()
    original = {k: v.clone() for k, v in inputs.items()}
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
        torch.testing.assert_close(model(**inputs), reference(**normalized))
        for key in inputs:
            torch.testing.assert_close(inputs[key], original[key], rtol=0, atol=0)
    path = tmp_path / "coordinates.mdlus"
    model.save(path)
    restored = Module.from_checkpoint(path).eval()
    assert restored.coordinate_scale == (12.0, 4.5, 3.25)
    torch.testing.assert_close(restored(**inputs), model(**inputs))


@pytest.mark.parametrize("scale", [(0, 1, 1), (1, -1, 1), (1, float("nan"), 1), (1, 2)])
def test_invalid_coordinate_scale(scale):
    """Reject an invalid coordinate frame before constructing the model."""
    with pytest.raises(ValueError, match="coordinate_scale"):
        _model(coordinate_scale=scale)
