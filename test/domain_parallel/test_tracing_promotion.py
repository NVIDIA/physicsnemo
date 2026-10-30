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

r"""Which operands send a traced ShardTensor op to the DTensor fallback.

Only plain tensors (as they appear under tracing) that need a gradient are
promoted there; other tensor subclasses keep their own dispatch.
"""

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.two_tensor import TwoTensor

from physicsnemo.domain_parallel.shard_tensor import _needs_fallback_under_tracing


def test_fake_plain_parameter_needs_fallback():
    with FakeTensorMode():
        param = torch.nn.Parameter(torch.randn(4))
    assert _needs_fallback_under_tracing((param,))


def test_plain_without_grad_takes_dispatch():
    with FakeTensorMode():
        buf = torch.randn(4)
    assert not _needs_fallback_under_tracing((buf,))


def test_other_subclass_takes_dispatch():
    t = TwoTensor(torch.randn(4), torch.randn(4)).requires_grad_(True)
    assert not _needs_fallback_under_tracing((t,))


def test_no_grad_mode_takes_dispatch():
    with FakeTensorMode():
        param = torch.nn.Parameter(torch.randn(4))
    with torch.no_grad():
        assert not _needs_fallback_under_tracing((param,))
