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
    tokenizations = []
    handle = model.context_builder.geometry_tokenizer.register_forward_hook(
        lambda module, inputs, output: tokenizations.append(output)
    )
    output = model.forward_with_geometry_features(
        local, geometry=geometry, global_embedding=global_embedding
    )
    handle.remove()
    assert len(tokenizations) == 1
    assert len(output) == 2
    prediction, features = output
    torch.testing.assert_close(
        prediction, original[0] if embedding or points else original
    )
    expected = model.context_builder.geometry_tokenizer(geometry)
    torch.testing.assert_close(features, expected)
    assert not features.requires_grad
    # The combined context includes global tokens and is a different feature.
    if embedding:
        assert original[1].shape[-1] > features.shape[-1]
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
    _, absent = no_geometry.forward_with_geometry_features(
        local, global_embedding=global_embedding
    )
    assert absent is None


@pytest.mark.parametrize("layout", ["single", "multiple", "structured"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_geometry_method_preserves_predictions_and_gradients(
    device, layout, checkpointing
):
    """Both entry points share stream handling, training, and checkpoint behavior."""
    torch.manual_seed(71)
    multi = layout == "multiple"
    structured = layout == "structured"
    model = (
        GeoTransolver(
            functional_dim=(3, 2) if multi else 3,
            out_dim=(2, 1) if multi else 2,
            geometry_dim=3,
            n_layers=1,
            n_hidden=16,
            n_head=4,
            slice_num=4,
            use_te=False,
            structured_shape=(3, 4) if structured else None,
            activation_checkpointing=checkpointing,
            activation_checkpointing_components=(
                "context",
                "preprocess",
                "blocks",
                "output",
            ),
        )
        .to(device)
        .train()
    )
    if multi:
        inputs = (
            torch.randn(1, 12, 3, device=device),
            torch.randn(1, 12, 2, device=device),
        )
    else:
        inputs = torch.randn((1, 3, 4, 3) if structured else (1, 12, 3), device=device)
    geometry = torch.randn((1, 3, 4, 3) if structured else (1, 15, 3), device=device)
    observed = []
    handle = model.register_forward_hook(
        lambda module, args, output: observed.append(output)
    )
    ordinary = model(inputs, geometry=geometry)
    original_parts = ordinary if multi else (ordinary,)
    sum(part.square().mean() for part in original_parts).backward()
    gradients = {
        name: None if p.grad is None else p.grad.clone()
        for name, p in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)
    prediction, tokens = model.forward_with_geometry_features(inputs, geometry=geometry)
    handle.remove()
    assert len(observed) == 1  # Direct custom methods do not run model-level hooks.
    assert not tokens.requires_grad
    torch.testing.assert_close(prediction, ordinary, rtol=0, atol=0)
    parts = prediction if multi else (prediction,)
    sum(part.square().mean() for part in parts).backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.grad, gradients[name], rtol=1e-6, atol=1e-9
        )
