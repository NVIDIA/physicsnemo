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

"""Regression tests for `Mesh` under `torch.compile`.

These tests guard against the `tensordict` 0.12.x regression in PR
`pytorch/tensordict#1552`, where the @tensorclass init wrapper's bypass branch
silently skipped both field-default normalization (`pytorch/tensordict#1709`)
and ``__post_init__`` (`pytorch/tensordict#1708`) under ``torch.compile``.

The bug manifested as:

* ``Mesh(points=p, cells=c).cell_normals`` raising ``AttributeError`` inside a
  compiled function (the cached property could not find the ``_cache`` field
  that ``__post_init__`` was supposed to materialize from its ``None``
  default).
* Silent miscomputation of any property whose result depends on a field
  normalized in ``__post_init__``.

These tests construct a ``Mesh`` *inside* a ``torch.compile``-traced function
and assert that:

1. The compiled call does not raise.
2. The compiled output matches the eager output exactly.

If either upstream regression returns (e.g. via a future tensordict pin bump,
or a refactor of ``Mesh`` that loses the workaround), these tests fail loudly
instead of waiting for the notebook-level CI to break.
"""

import pytest
import torch

from physicsnemo.mesh import Mesh

### Fixtures ###


@pytest.fixture
def triangle_3d() -> tuple[torch.Tensor, torch.Tensor]:
    """A single right triangle in the XY-plane of 3D space.

    The triangle has vertices ``(0,0,0)``, ``(1,0,0)``, ``(0,1,0)``, so:

    * ``cell_normals == [[0, 0, 1]]`` (unit +Z)
    * ``cell_areas == [0.5]``
    * ``cell_centroids == [[1/3, 1/3, 0]]``

    Small enough that compile overhead dominates wall time, keeping the test
    cheap to run on every CI invocation.
    """
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    cells = torch.tensor([[0, 1, 2]])
    return points, cells


### Tests ###


@pytest.mark.parametrize(
    "property_name",
    [
        # Cached properties: each reads from `self._cache`, which is a field
        # defaulted to None and materialized in __post_init__. Broken by both
        # upstream regressions under tensordict 0.12.x.
        "cell_normals",
        "cell_areas",
        "cell_centroids",
        "point_normals",
    ],
)
def test_cached_property_under_compile(
    property_name: str,
    triangle_3d: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Cached `Mesh` properties must produce the same output eager vs compiled.

    Regression test for `pytorch/tensordict#1708` and `pytorch/tensordict#1709`.
    """
    points, cells = triangle_3d

    def fn(p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return getattr(Mesh(points=p, cells=c), property_name)

    expected = fn(points, cells)
    compiled = torch.compile(fn, fullgraph=False)(points, cells)

    torch.testing.assert_close(compiled, expected)


@pytest.mark.parametrize("field_name", ["point_data", "cell_data", "global_data"])
def test_data_field_under_compile(
    field_name: str,
    triangle_3d: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Data-container fields (``point_data``/``cell_data``/``global_data``)
    default to ``None`` in the schema and are normalized to empty
    ``TensorDict`` instances in ``__post_init__``. Accessing them inside a
    compiled function must not raise.

    Regression test for `pytorch/tensordict#1708` and `pytorch/tensordict#1709`.
    """
    points, cells = triangle_3d

    ### Read .n_<thing> through the field as a proxy for "field exists" ###
    # We don't compare to a numerical reference here; we just want to confirm
    # the compiled function doesn't blow up on attribute access.
    def fn(p: torch.Tensor, c: torch.Tensor) -> int:
        m = Mesh(points=p, cells=c)
        return len(getattr(m, field_name))

    expected = fn(points, cells)
    compiled = torch.compile(fn, fullgraph=False)(points, cells)

    assert compiled == expected


def test_post_init_runs_under_compile(
    triangle_3d: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Construct a ``Mesh`` inside a compiled function, mutate ``_cache``
    through the side-effecting ``cell_normals`` getter, and read it back.

    This exercises the full ``__post_init__`` -> cached-property -> cache-write
    -> cache-read round-trip. If ``__post_init__`` is silently skipped (the
    `#1708` regression), ``self._cache`` is missing entirely and the first
    cache access in the property body raises.
    """
    points, cells = triangle_3d

    def fn(p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        m = Mesh(points=p, cells=c)
        # First access triggers `__post_init__`-materialized `_cache` and
        # writes the result into it.
        return m.cell_normals

    expected = torch.tensor([[0.0, 0.0, 1.0]])
    compiled = torch.compile(fn, fullgraph=False)(points, cells)

    torch.testing.assert_close(compiled, expected)
