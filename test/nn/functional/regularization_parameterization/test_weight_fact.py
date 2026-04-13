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

from physicsnemo.nn.functional import weight_fact
from physicsnemo.nn.functional.regularization_parameterization import WeightFact


# Validate the torch weight-factorization implementation.
def test_weight_fact_torch(device: str):
    w = torch.randn(32, 16, device=device, dtype=torch.float32)
    g, v = weight_fact(w, mean=1.0, stddev=0.1, implementation="torch")

    assert g.shape == (w.shape[0], 1)
    assert v.shape == w.shape
    assert (g > 0).all()
    torch.testing.assert_close(g * v, w, atol=1e-6, rtol=1e-6)


# Validate backward behavior for weight-factorization outputs.
def test_weight_fact_torch_backward(device: str):
    w = torch.randn(16, 8, device=device, dtype=torch.float32, requires_grad=True)
    g, v = weight_fact(w, mean=1.0, stddev=0.1, implementation="torch")

    # g * v reconstructs w, so the gradient wrt w is expected to be ones.
    (g * v).sum().backward()
    torch.testing.assert_close(w.grad, torch.ones_like(w))


# Validate benchmark input generation contracts for forward and backward paths.
def test_weight_fact_make_inputs_forward(device: str):
    label, args, kwargs = next(iter(WeightFact.make_inputs_forward(device=device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    g, v = WeightFact.dispatch(*args, implementation="torch", **kwargs)
    assert g.shape[0] == v.shape[0]
    assert g.shape[1] == 1


def test_weight_fact_make_inputs_backward(device: str):
    label, args, kwargs = next(iter(WeightFact.make_inputs_backward(device=device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    w = args[0]
    assert w.requires_grad
    g, v = WeightFact.dispatch(*args, implementation="torch", **kwargs)
    (g * v).sum().backward()
    assert w.grad is not None
