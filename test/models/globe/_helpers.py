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

"""Shared helpers for the GLOBE tests in this package."""

import torch

from physicsnemo.mesh import FieldSchema
from physicsnemo.models.globe import GLOBE

### GLOBE's optional constructor arguments and their defaults.
DEFAULTS = {
    "global_schema": {},
    "n_communication_hyperlayers": 2,
    "n_latent_scalars": 12,
    "n_latent_vectors": 6,
    "smoothing_radius": 1e-8,
    "hidden_layer_sizes": [64, 64, 64],
    "n_spherical_harmonics": 4,
    "theta": 1.5,
    "leaf_size": 1,
    "network_type": "pade",
    "self_regularization_beta": None,
    "latent_compression_scale": None,
    "expand_far_targets": True,
    "use_gradient_checkpointing": True,
    "tree_build_device": None,
}


def assert_globe_attributes(model: GLOBE, kwargs: dict) -> None:
    """Assert that ``model`` has the public attributes of ``GLOBE(**kwargs)``.

    Each constructor argument is stored as given, except that the field
    declarations are parsed, ``reference_area`` is a buffer, and the device
    is a ``torch.device``; the kernels are built to match.
    """
    expected = DEFAULTS | kwargs
    bc_names = list(expected["boundary_source_schemas"])
    output_ranks = FieldSchema.parse(expected["output_schema"]).ranks

    assert model.n_spatial_dims == expected["n_spatial_dims"]
    assert model.output_schema == FieldSchema.parse(expected["output_schema"])
    assert model.boundary_source_schemas == {
        bc: FieldSchema.parse(spec)
        for bc, spec in expected["boundary_source_schemas"].items()
    }
    assert model.boundary_condition_names == bc_names
    assert model.reference_length_names == expected["reference_length_names"]
    assert isinstance(model.reference_area, torch.Tensor)
    assert "reference_area" in dict(model.named_buffers())
    assert model.reference_area.item() == expected["reference_area"]
    assert model.global_schema == FieldSchema.parse(expected["global_schema"])
    for name in (
        "n_communication_hyperlayers",
        "n_latent_scalars",
        "n_latent_vectors",
        "smoothing_radius",
        "hidden_layer_sizes",
        "n_spherical_harmonics",
        "theta",
        "leaf_size",
        "network_type",
        "self_regularization_beta",
        "latent_compression_scale",
        "expand_far_targets",
        "use_gradient_checkpointing",
    ):
        assert getattr(model, name) == expected[name], name
    if expected["tree_build_device"] is None:
        assert model.tree_build_device is None
    else:
        assert model.tree_build_device == torch.device(expected["tree_build_device"])

    assert len(model.kernel_layers) == expected["n_communication_hyperlayers"] + 1
    for layer in model.kernel_layers:
        assert list(layer.keys()) == bc_names
        for kernel in layer.values():
            assert list(kernel.kernels.keys()) == expected["reference_length_names"]
            assert kernel.leaf_size == expected["leaf_size"]
            assert kernel.network_type == expected["network_type"]
            assert (
                kernel.use_gradient_checkpointing
                == (expected["use_gradient_checkpointing"])
            )
    assert [t.bias is not None for t in model.final_field_transforms] == [
        output_ranks[name] == 0 for name in sorted(output_ranks)
    ]
