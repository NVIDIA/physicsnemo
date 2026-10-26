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

"""Feature output remains independent of scoring and application policy."""

import pytest
import torch

from physicsnemo.models.geotransolver import GeoTransolver


@pytest.mark.parametrize(
    "embedding,points", [(False, False), (True, False), (False, True), (True, True)]
)
def test_geometry_feature_output(device, embedding, points):
    torch.manual_seed(4)
    model = (
        GeoTransolver(
            functional_dim=3,
            out_dim=2,
            geometry_dim=3,
            global_dim=4,
            n_layers=1,
            n_hidden=16,
            n_head=4,
            slice_num=4,
            use_te=False,
        )
        .to(device)
        .eval()
    )
    assert not hasattr(model, "ood_guard")
    local = torch.randn(2, 11, 3, device=device)
    geometry = torch.randn(2, 13, 3, device=device)
    global_embedding = torch.randn(2, 2, 4, device=device)
    kwargs = dict(
        geometry=geometry,
        global_embedding=global_embedding,
        return_embedding_states=embedding,
        return_point_features=points,
    )
    original = model(local, **kwargs)
    output = model(local, **kwargs, return_geometry_features=True)
    expected_parts = original if embedding or points else (original,)
    assert len(output) == len(expected_parts) + 1
    for actual, expected in zip(output[:-1], expected_parts):
        torch.testing.assert_close(actual, expected)
    features = output[-1]
    expected = model.context_builder.geometry_tokenizer(geometry)
    torch.testing.assert_close(features, expected)
    assert not features.requires_grad
    # The combined context includes global tokens and is a different feature.
    if embedding:
        assert output[1].shape[-1] > features.shape[-1]
    # No geometry tokenizer is configured in this independent model. Removing
    # a configured modality from the first model changes its context width.
    no_geometry = (
        GeoTransolver(
            functional_dim=3,
            out_dim=2,
            global_dim=4,
            n_layers=1,
            n_hidden=16,
            n_head=4,
            slice_num=4,
            use_te=False,
        )
        .to(device)
        .eval()
    )
    _, absent = no_geometry(
        local, global_embedding=global_embedding, return_geometry_features=True
    )
    assert absent is None
