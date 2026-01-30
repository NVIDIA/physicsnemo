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

r"""Test suite for EdgeRotation module.

Tests the EdgeRotation class for computing Wigner D-matrices from edge
direction vectors, which is used for rotating spherical harmonic embeddings
to edge-aligned local frames in equivariant networks.
"""

import pytest
import torch

from physicsnemo.experimental.nn.symmetry.wigner import EdgeRotation


class TestEdgeRotation:
    r"""Test suite for EdgeRotation module."""

    # =========================================================================
    # Initialization Tests
    # =========================================================================

    def test_init_buffers_registered(self) -> None:
        r"""Verify J matrices are registered as persistent buffers."""
        lmax = 3
        model = EdgeRotation(lmax=lmax)

        # Check all J matrices are registered
        for ell in range(lmax + 1):
            assert hasattr(model, f"_J_{ell}"), f"Missing buffer _J_{ell}"
            J_l = getattr(model, f"_J_{ell}")
            assert J_l.shape == (2 * ell + 1, 2 * ell + 1), f"Wrong shape for _J_{ell}"

        # Check they appear in state_dict
        state_dict = model.state_dict()
        for ell in range(lmax + 1):
            assert f"_J_{ell}" in state_dict, f"_J_{ell} not in state_dict"

    def test_init_default_mmax(self) -> None:
        r"""Verify mmax defaults to lmax when not specified."""
        lmax = 4
        model = EdgeRotation(lmax=lmax)

        assert model.lmax == lmax
        assert model.mmax == lmax, "mmax should default to lmax"

    def test_init_invalid_mmax_raises(self) -> None:
        r"""Verify ValueError when mmax > lmax."""
        with pytest.raises(ValueError, match="mmax must be <= lmax"):
            EdgeRotation(lmax=2, mmax=3)

    def test_init_dimensions_computed(self) -> None:
        r"""Verify _full_dim and _reduced_dim are correct."""
        # Test case 1: lmax=2, mmax=2 (no reduction)
        model = EdgeRotation(lmax=2, mmax=2)
        assert model._full_dim == 9, "full_dim should be (2+1)^2 = 9"
        # reduced_dim = min(5, 1) + min(5, 3) + min(5, 5) = 1 + 3 + 5 = 9
        assert model._reduced_dim == 9, "reduced_dim should be 9 when mmax=lmax"

        # Test case 2: lmax=2, mmax=1 (reduction)
        model = EdgeRotation(lmax=2, mmax=1)
        assert model._full_dim == 9, "full_dim should be (2+1)^2 = 9"
        # reduced_dim = min(3, 1) + min(3, 3) + min(3, 5) = 1 + 3 + 3 = 7
        assert model._reduced_dim == 7, "reduced_dim should be 7 when mmax=1"

        # Test case 3: lmax=3, mmax=1
        model = EdgeRotation(lmax=3, mmax=1)
        assert model._full_dim == 16, "full_dim should be (3+1)^2 = 16"
        # reduced_dim = min(3, 1) + min(3, 3) + min(3, 5) + min(3, 7) = 1 + 3 + 3 + 3 = 10
        assert model._reduced_dim == 10, "reduced_dim should be 10 when lmax=3, mmax=1"

        # Test case 4: lmax=3, mmax=0
        model = EdgeRotation(lmax=3, mmax=0)
        # reduced_dim = min(1, 1) + min(1, 3) + min(1, 5) + min(1, 7) = 1 + 1 + 1 + 1 = 4
        assert model._reduced_dim == 4, "reduced_dim should be 4 when mmax=0"

    # =========================================================================
    # Forward Shape Tests
    # =========================================================================

    def test_forward_shape_full(self) -> None:
        r"""Test output shape when mmax = lmax."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        num_nodes, max_neighbors = 4, 5
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3)

        D = model(edge_vecs)

        # full_dim = (lmax+1)^2 = 9
        # reduced_dim = 1 + 3 + 5 = 9 (same as full when mmax=lmax)
        assert D.shape == (4, 5, 9, 9), f"Expected shape (4, 5, 9, 9), got {D.shape}"

    def test_forward_shape_reduced(self) -> None:
        r"""Test output shape when mmax < lmax."""
        lmax = 3
        mmax = 1
        model = EdgeRotation(lmax=lmax, mmax=mmax)

        num_nodes, max_neighbors = 4, 5
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3)

        D = model(edge_vecs)

        # full_dim = (3+1)^2 = 16
        # reduced_dim = 1 + 3 + 3 + 3 = 10
        assert D.shape == (4, 5, 10, 16), (
            f"Expected shape (4, 5, 10, 16), got {D.shape}"
        )

    # =========================================================================
    # Mathematical Correctness Tests
    # =========================================================================

    def test_y_axis_gives_identity(self) -> None:
        r"""Edge [0,1,0] should give D ≈ I.

        In the convention used, beta = acos(y), so pointing along y-axis
        (y=1) gives beta=0. Combined with alpha=0 when x=z=0, this
        results in the identity rotation.
        """
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        # Single edge pointing along y-axis (identity direction in this convention)
        edge_vecs = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64)
        model = model.to(dtype=torch.float64)

        D = model(edge_vecs)

        # For y-axis (beta=0, alpha=0, gamma=0), the rotation should be identity
        expected = torch.eye(9, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        assert torch.allclose(D, expected, rtol=1e-5, atol=1e-5), (
            "D matrix for y-axis should be identity"
        )

    def test_orthogonality(self) -> None:
        r"""For each edge, D @ D^T ≈ I (within the full block-diagonal)."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        num_nodes, max_neighbors = 3, 4
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3, dtype=torch.float64)
        model = model.to(dtype=torch.float64)

        D = model(edge_vecs)

        # When mmax = lmax, D is square (full_dim x full_dim)
        # Check D @ D^T ≈ I for each edge
        identity = torch.eye(9, dtype=torch.float64)

        for i in range(num_nodes):
            for j in range(max_neighbors):
                D_ij = D[i, j]
                product = torch.matmul(D_ij, D_ij.T)
                assert torch.allclose(product, identity, rtol=1e-5, atol=1e-5), (
                    f"D @ D^T not identity for edge [{i}, {j}]"
                )

    def test_negative_y_axis(self) -> None:
        r"""Edge [0,-1,0] should give expected pattern.

        Negative y-axis corresponds to beta=pi (180 degree rotation about y).
        The D matrix should still be orthogonal but not identity.
        """
        lmax = 1
        model = EdgeRotation(lmax=lmax)

        # Edge pointing along negative y-axis
        edge_vecs = torch.tensor([[[0.0, -1.0, 0.0]]], dtype=torch.float64)
        model = model.to(dtype=torch.float64)

        D = model(edge_vecs)

        # Verify it's still orthogonal
        D_squeezed = D[0, 0]
        product = torch.matmul(D_squeezed, D_squeezed.T)
        identity = torch.eye(4, dtype=torch.float64)  # (1+1)^2 = 4 for lmax=1
        assert torch.allclose(product, identity, rtol=1e-5, atol=1e-5), (
            "D @ D^T not identity for negative y-axis"
        )

        # Verify it's not just the identity
        assert not torch.allclose(D_squeezed, identity, rtol=1e-3, atol=1e-3), (
            "D for negative y should not be identity"
        )

    # =========================================================================
    # Mask Tests
    # =========================================================================

    def test_mask_applied_identity(self) -> None:
        r"""Masked (False) edges get identity matrix."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        num_nodes, max_neighbors = 2, 3
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3)

        # Mask out specific edges
        mask = torch.ones(num_nodes, max_neighbors, dtype=torch.bool)
        mask[0, 0] = False
        mask[1, 2] = False

        D = model(edge_vecs, mask=mask)

        # Get expected identity in reduced form
        identity = model._get_identity_reduced(1, D.dtype, D.device)

        # Check masked edges have identity
        assert torch.allclose(D[0, 0], identity[0], rtol=1e-5, atol=1e-5), (
            "Masked edge [0,0] should have identity"
        )
        assert torch.allclose(D[1, 2], identity[0], rtol=1e-5, atol=1e-5), (
            "Masked edge [1,2] should have identity"
        )

    def test_mask_none_all_computed(self) -> None:
        r"""Without mask, all edges are computed normally."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        num_nodes, max_neighbors = 2, 3
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3, dtype=torch.float64)
        model = model.to(dtype=torch.float64)

        # Without mask
        D_no_mask = model(edge_vecs, mask=None)

        # With all-True mask
        mask = torch.ones(num_nodes, max_neighbors, dtype=torch.bool)
        D_all_true = model(edge_vecs, mask=mask)

        assert torch.allclose(D_no_mask, D_all_true, rtol=1e-10, atol=1e-10), (
            "mask=None should behave same as all-True mask"
        )

    # =========================================================================
    # State Dict Tests
    # =========================================================================

    def test_state_dict_contains_J_matrices(self) -> None:
        r"""Save state_dict and verify J buffers present."""
        lmax = 3
        model = EdgeRotation(lmax=lmax)

        state_dict = model.state_dict()

        # Check all J matrices are in state_dict
        for ell in range(lmax + 1):
            key = f"_J_{ell}"
            assert key in state_dict, f"{key} not found in state_dict"
            assert state_dict[key].shape == (2 * ell + 1, 2 * ell + 1), (
                f"Wrong shape for {key} in state_dict"
            )

    def test_state_dict_roundtrip(self) -> None:
        r"""Save/load model and verify identical output."""
        lmax = 2
        mmax = 1
        model1 = EdgeRotation(lmax=lmax, mmax=mmax)

        # Save state dict
        state_dict = model1.state_dict()

        # Create new model and load state dict
        model2 = EdgeRotation(lmax=lmax, mmax=mmax)
        model2.load_state_dict(state_dict)

        # Test with same input
        edge_vecs = torch.randn(3, 4, 3)

        D1 = model1(edge_vecs)
        D2 = model2(edge_vecs)

        assert torch.allclose(D1, D2, rtol=1e-10, atol=1e-10), (
            "Model output should be identical after state_dict roundtrip"
        )

    # =========================================================================
    # Dtype/Device Tests
    # =========================================================================

    def test_dtype_float32(self) -> None:
        r"""Float32 input → float32 output."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        edge_vecs = torch.randn(2, 3, 3, dtype=torch.float32)
        D = model(edge_vecs)

        assert D.dtype == torch.float32, f"Expected float32 output, got {D.dtype}"

    def test_dtype_float64(self) -> None:
        r"""Float64 input → float64 output."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        edge_vecs = torch.randn(2, 3, 3, dtype=torch.float64)
        D = model(edge_vecs)

        assert D.dtype == torch.float64, f"Expected float64 output, got {D.dtype}"

    @pytest.mark.cuda
    def test_cuda_device(self) -> None:
        r"""CUDA tensor placement."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        lmax = 2
        model = EdgeRotation(lmax=lmax)
        model = model.cuda()

        edge_vecs = torch.randn(2, 3, 3, device="cuda")
        D = model(edge_vecs)

        assert D.device.type == "cuda", f"Expected CUDA output, got {D.device}"

    # =========================================================================
    # Gradient Tests
    # =========================================================================

    def test_gradient_flow(self) -> None:
        r"""Verify gradients flow through edge_vecs."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        edge_vecs = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
        model = model.to(dtype=torch.float64)

        D = model(edge_vecs)

        # Compute a scalar loss
        loss = D.sum()
        loss.backward()

        assert edge_vecs.grad is not None, "Gradients should flow to edge_vecs"
        assert torch.isfinite(edge_vecs.grad).all(), "Gradients should be finite"
        assert not torch.allclose(edge_vecs.grad, torch.zeros_like(edge_vecs.grad)), (
            "Gradients should be non-zero"
        )

    # =========================================================================
    # Batch Consistency Tests
    # =========================================================================

    def test_batch_consistency(self) -> None:
        r"""Same edge vector in different batch positions → same D matrix."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        # Create edge vectors where some are duplicated
        single_edge = torch.randn(3, dtype=torch.float64)

        # Place same edge in different positions
        edge_vecs = torch.randn(3, 4, 3, dtype=torch.float64)
        edge_vecs[0, 1] = single_edge
        edge_vecs[2, 3] = single_edge

        model = model.to(dtype=torch.float64)
        D = model(edge_vecs)

        # D matrices for same edge vector should be identical
        assert torch.allclose(D[0, 1], D[2, 3], rtol=1e-10, atol=1e-10), (
            "Same edge vector should produce same D matrix"
        )
