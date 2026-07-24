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

from __future__ import annotations

import torch

from physicsnemo.core.function_spec import FunctionSpec
from test.nn.functional._parity_utils import clone_case


def make_lsq_case(device: str, n_dims: int = 3):
    """Build deterministic KNN-CSR data for an LSQ mesh functional."""
    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(1900 + n_dims)
    points = torch.rand((384, n_dims), generator=generator, device=torch_device)
    dists = torch.cdist(points, points)
    knn = torch.topk(dists, k=15, largest=False, dim=1).indices[:, 1:]
    offsets = torch.arange(
        0,
        384 * 14 + 1,
        14,
        device=torch_device,
        dtype=torch.int64,
    )
    indices = knn.reshape(-1).to(torch.int64)
    return points.to(torch.float32), offsets, indices


def make_simple_cotan_case(device: str):
    """Build a three-point chain for a cotangent mesh functional."""
    torch_device = torch.device(device)
    points = torch.tensor(
        [[0.0], [1.0], [2.0]],
        dtype=torch.float32,
        device=torch_device,
    )
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64, device=torch_device)
    weights = torch.ones((2,), dtype=torch.float32, device=torch_device)
    volumes = torch.ones((3,), dtype=torch.float32, device=torch_device)
    return points, edges, weights, volumes


def check_backend_forward_parity(
    device: str,
    spec: type[FunctionSpec],
) -> None:
    """Compare all representative Warp outputs with the torch baseline."""
    for _label, args, kwargs in spec.make_inputs_forward(device=device):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = spec.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_warp = spec.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        spec.compare_forward(out_warp, out_torch)


def check_backend_backward_parity(
    device: str,
    spec: type[FunctionSpec],
    grad_arg_indices: tuple[int, ...],
) -> None:
    """Compare selected Warp argument gradients with the torch baseline."""
    for _label, args, kwargs in spec.make_inputs_backward(device=device):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = spec.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_torch.square().mean().backward()

        out_warp = spec.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        out_warp.square().mean().backward()

        for arg_index in grad_arg_indices:
            grad_torch = args_torch[arg_index].grad
            grad_warp = args_warp[arg_index].grad
            assert grad_torch is not None
            assert grad_warp is not None
            spec.compare_backward(grad_warp, grad_torch)
