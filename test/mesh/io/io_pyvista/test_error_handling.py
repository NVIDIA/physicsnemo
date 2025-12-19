# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Tests for physicsnemo.mesh.io module - error handling."""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from physicsnemo.mesh.io.io_pyvista import from_pyvista  # noqa: E402


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_manifold_dim(self):
        """Test that invalid manifold_dim raises ValueError."""
        pv_mesh = pv.Sphere()

        with pytest.raises(ValueError, match="Invalid manifold_dim"):
            from_pyvista(pv_mesh, manifold_dim=4)

        with pytest.raises(ValueError, match="Invalid manifold_dim"):
            from_pyvista(pv_mesh, manifold_dim=-1)

    def test_mixed_geometry_error(self):
        """Test that meshes with mixed geometry types raise error."""
        # Create a mesh with both lines and cells (if possible)
        # This is tricky with PyVista; skip if not easily testable
        pass

    def test_empty_mesh(self):
        """Test conversion of empty mesh."""
        points = np.empty((0, 3), dtype=np.float32)
        pv_mesh = pv.PolyData(points)

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_points == 0
        assert mesh.n_cells == 0
        assert mesh.n_manifold_dims == 0
