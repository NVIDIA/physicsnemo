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

"""Tests for `is_tracing`, the thread-correct replacement for ``is_compiling``."""

import threading

import torch

from physicsnemo.utils._tracing import is_tracing


def _in_another_thread_while_compiling(fn):
    """Return ``fn()``, run in a second thread while this thread compiles."""
    result = {}

    def backend(gm, example_inputs):
        thread = threading.Thread(target=lambda: result.update(value=fn()))
        thread.start()
        thread.join()
        return gm.forward

    torch._dynamo.reset()
    torch.compile(lambda x: torch.sin(x) + 1, backend=backend)(torch.randn(4))
    return result["value"]


def test_false_in_eager() -> None:
    assert is_tracing() is False


def test_true_under_compile() -> None:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return x + is_tracing()

    torch._dynamo.reset()
    out = torch.compile(fn, backend="eager", fullgraph=True)(torch.zeros(2))
    torch.testing.assert_close(out, torch.ones(2))


def test_false_in_another_thread_while_compiling() -> None:
    """Since torch 2.13, ``is_compiling()`` is True in every thread during a
    compile; ``is_tracing()`` must stay False in a thread that runs eagerly."""
    assert _in_another_thread_while_compiling(is_tracing) is False


def test_true_during_non_strict_export() -> None:
    seen = []

    class Module(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seen.append(is_tracing())
            return x + 1

    torch.export.export(Module(), (torch.zeros(2),), strict=False)
    assert seen and all(seen)
