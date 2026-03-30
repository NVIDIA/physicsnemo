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

from physicsnemo.nn.functional import drop_path
from physicsnemo.nn.functional.regularization_parameterization import DropPath


# Validate the torch drop-path implementation on deterministic RNG state.
def test_drop_path_torch(device: str):
    x = torch.randn(8, 16, device=device, dtype=torch.float32)
    keep_prob = 0.75

    torch.manual_seed(1234)
    out = drop_path(
        x,
        drop_prob=1.0 - keep_prob,
        training=True,
        scale_by_keep=True,
        implementation="torch",
    )

    torch.manual_seed(1234)
    mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep_prob)
    mask.div_(keep_prob)
    expected = x * mask
    torch.testing.assert_close(out, expected)


# Validate no-op behavior for inference mode and zero drop probability.
def test_drop_path_noop_behavior(device: str):
    x = torch.randn(4, 8, device=device, dtype=torch.float32)

    out_eval = drop_path(
        x,
        drop_prob=0.5,
        training=False,
        scale_by_keep=True,
        implementation="torch",
    )
    torch.testing.assert_close(out_eval, x)

    out_zero = drop_path(
        x,
        drop_prob=0.0,
        training=True,
        scale_by_keep=True,
        implementation="torch",
    )
    torch.testing.assert_close(out_zero, x)


# Validate backward behavior for stochastic-depth masking.
def test_drop_path_torch_backward(device: str):
    x = torch.randn(8, 16, device=device, dtype=torch.float32, requires_grad=True)
    keep_prob = 0.6

    torch.manual_seed(2026)
    out = drop_path(
        x,
        drop_prob=1.0 - keep_prob,
        training=True,
        scale_by_keep=True,
        implementation="torch",
    )
    out.sum().backward()

    torch.manual_seed(2026)
    expected_grad = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(
        keep_prob
    )
    expected_grad.div_(keep_prob)
    expected_grad = expected_grad.expand_as(x)
    torch.testing.assert_close(x.grad, expected_grad)


# Validate benchmark input generation contracts for forward and backward paths.
def test_drop_path_make_inputs_forward(device: str):
    label, args, kwargs = next(iter(DropPath.make_inputs_forward(device=device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    output = DropPath.dispatch(*args, implementation="torch", **kwargs)
    assert output.ndim == 2
    assert output.dtype == torch.float32


def test_drop_path_make_inputs_backward(device: str):
    label, args, kwargs = next(iter(DropPath.make_inputs_backward(device=device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    x = args[0]
    assert x.requires_grad
    output = DropPath.dispatch(*args, implementation="torch", **kwargs)
    output.sum().backward()
    assert x.grad is not None
