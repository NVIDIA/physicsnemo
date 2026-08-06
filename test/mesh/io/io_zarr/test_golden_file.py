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

"""Golden-file test: a store committed to the repo must stay readable.

The store under ``golden/`` was written by ``to_zarr`` at the time this
feature landed. Reading it with the current code guards against silent
format drift (attr names, tree layout, dtype handling). The expected values
are regenerated from the same seeded factories rather than pickled, so the
test is self-describing.

Regenerate (only on a deliberate, documented format change)::

    python test/mesh/io/io_zarr/test_golden_file.py
"""

from pathlib import Path

from conftest import assert_meshes_equal, make_domain_mesh

from physicsnemo.mesh import DomainMesh
from physicsnemo.mesh.io import from_zarr

GOLDEN = Path(__file__).parent / "golden" / "domain_mesh.zarr"


def test_golden_store_reads_back():
    assert GOLDEN.exists(), (
        "golden store missing; regenerate with `python test_golden_file.py`"
    )
    back = from_zarr(GOLDEN)
    assert isinstance(back, DomainMesh)
    expected = make_domain_mesh()
    assert_meshes_equal(expected.interior, back.interior)
    for name in expected.boundary_names:
        assert_meshes_equal(expected.boundaries[name], back.boundaries[name])


if __name__ == "__main__":  # golden-store regeneration
    import shutil

    from physicsnemo.mesh.io import to_zarr

    shutil.rmtree(GOLDEN, ignore_errors=True)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    to_zarr(make_domain_mesh(), GOLDEN, chunk_rows=16)
    print(f"regenerated {GOLDEN}")
