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

import torch

from physicsnemo.nn.functional.interpolation.interpolation import Interpolation
from test.conftest import requires_module


@requires_module("warp")
def test_interpolation_warp_matches_torch_forward_backward(device: str):
    for label, args, kwargs in Interpolation.make_inputs(device=device):
        query_points, context_grid, grid = args

        query_torch = query_points.detach().clone().requires_grad_(True)
        query_warp = query_points.detach().clone().requires_grad_(True)
        grid_torch = context_grid.detach().clone().requires_grad_(True)
        grid_warp = context_grid.detach().clone().requires_grad_(True)

        out_torch = Interpolation.dispatch(
            query_torch,
            grid_torch,
            grid,
            implementation="torch",
            **kwargs,
        )
        out_warp = Interpolation.dispatch(
            query_warp,
            grid_warp,
            grid,
            implementation="warp",
            **kwargs,
        )
        torch.testing.assert_close(
            out_warp,
            out_torch,
            atol=5e-5,
            rtol=1e-4,
            msg=f"forward mismatch for case '{label}' on device '{device}'",
        )

        grad_out = torch.randn_like(out_torch)
        out_torch.backward(grad_out)
        out_warp.backward(grad_out)

        if query_torch.grad is None or query_warp.grad is None:
            assert query_torch.grad is None and query_warp.grad is None, (
                f"query gradient mismatch (None handling) for case '{label}' "
                f"on device '{device}'"
            )
        else:
            torch.testing.assert_close(
                query_warp.grad,
                query_torch.grad,
                atol=5e-5,
                rtol=1e-4,
                msg=f"query gradient mismatch for case '{label}' on device '{device}'",
            )

        if grid_torch.grad is None or grid_warp.grad is None:
            assert grid_torch.grad is None and grid_warp.grad is None, (
                f"context-grid gradient mismatch (None handling) for case '{label}' "
                f"on device '{device}'"
            )
        else:
            torch.testing.assert_close(
                grid_warp.grad,
                grid_torch.grad,
                atol=5e-5,
                rtol=1e-4,
                msg=f"context-grid gradient mismatch for case '{label}' on device '{device}'",
            )
