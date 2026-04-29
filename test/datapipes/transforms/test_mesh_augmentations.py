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

"""Tests for distribution-parametrized mesh augmentations."""

import math
import warnings

import pytest
import torch
import torch.distributions as D

from physicsnemo.datapipes.transforms.mesh.augmentations import (
    RandomRotateMesh,
    RandomScaleMesh,
    RandomTranslateMesh,
    _sample_distribution,
)
from physicsnemo.mesh import DomainMesh, Mesh

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_mesh_3d() -> Mesh:
    """A minimal 3-D mesh (single triangle)."""
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    return Mesh(points=points, cells=cells)


def _simple_domain_3d() -> DomainMesh:
    """A minimal 3-D DomainMesh with interior + one boundary."""
    interior = Mesh(
        points=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        cells=torch.tensor([[0, 1, 2]], dtype=torch.int64),
    )
    wall = Mesh(
        points=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        cells=torch.tensor([[0, 1, 2]], dtype=torch.int64),
    )
    return DomainMesh(interior=interior, boundaries={"wall": wall})


def _seed(aug, seed: int):
    """Assign a seeded generator to an augmentation transform."""
    aug.set_generator(torch.Generator().manual_seed(seed))
    return aug


# ---------------------------------------------------------------------------
# _sample_distribution
# ---------------------------------------------------------------------------


class TestSampleDistribution:
    """Tests for the _sample_distribution() helper function."""

    def test_uniform_statistics(self):
        """Samples from Uniform(2, 5) should have mean ~3.5 and lie in [2, 5]."""
        gen = torch.Generator().manual_seed(0)
        dist = D.Uniform(2.0, 5.0)
        samples = torch.stack(
            [_sample_distribution(dist, (1,), gen).squeeze(0) for _ in range(5000)]
        )
        assert samples.min() >= 2.0
        assert samples.max() <= 5.0
        assert samples.mean().item() == pytest.approx(3.5, abs=0.1)

    def test_normal_statistics(self):
        """Samples from Normal(10, 0.5) should cluster near 10."""
        gen = torch.Generator().manual_seed(0)
        dist = D.Normal(10.0, 0.5)
        samples = torch.stack(
            [_sample_distribution(dist, (1,), gen).squeeze(0) for _ in range(5000)]
        )
        assert samples.mean().item() == pytest.approx(10.0, abs=0.05)
        assert samples.std().item() == pytest.approx(0.5, abs=0.05)

    def test_cauchy_median(self):
        """Cauchy(3, 1) should have median ~3 (mean is undefined)."""
        gen = torch.Generator().manual_seed(0)
        dist = D.Cauchy(3.0, 1.0)
        samples = torch.stack(
            [_sample_distribution(dist, (1,), gen).squeeze(0) for _ in range(5000)]
        )
        assert samples.median().item() == pytest.approx(3.0, abs=0.1)

    def test_laplace_statistics(self):
        """Laplace(0, 0.5) should have mean ~0 and scale ~0.5."""
        gen = torch.Generator().manual_seed(0)
        dist = D.Laplace(0.0, 0.5)
        samples = torch.stack(
            [_sample_distribution(dist, (1,), gen).squeeze(0) for _ in range(5000)]
        )
        assert samples.mean().item() == pytest.approx(0.0, abs=0.05)
        # Mean absolute deviation of Laplace(0, b) is b
        assert samples.abs().mean().item() == pytest.approx(0.5, abs=0.05)

    def test_reproducibility_with_generator(self):
        """Same seed should produce identical samples."""
        dist = D.Normal(0.0, 1.0)

        gen1 = torch.Generator().manual_seed(123)
        s1 = _sample_distribution(dist, (10,), gen1)

        gen2 = torch.Generator().manual_seed(123)
        s2 = _sample_distribution(dist, (10,), gen2)

        assert torch.allclose(s1, s2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different samples."""
        dist = D.Normal(0.0, 1.0)

        gen1 = torch.Generator().manual_seed(0)
        s1 = _sample_distribution(dist, (10,), gen1)

        gen2 = torch.Generator().manual_seed(999)
        s2 = _sample_distribution(dist, (10,), gen2)

        assert not torch.allclose(s1, s2)

    def test_multidimensional_shape(self):
        """Should return the requested shape."""
        gen = torch.Generator().manual_seed(0)
        dist = D.Normal(0.0, 1.0)
        s = _sample_distribution(dist, (3, 4), gen)
        assert s.shape == (3, 4)

    def test_fallback_warning_for_poisson(self):
        """Poisson has no icdf; should fall back with a warning."""
        gen = torch.Generator().manual_seed(0)
        dist = D.Poisson(3.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s = _sample_distribution(dist, (100,), gen)
            assert len(w) == 1
            assert "icdf" in str(w[0].message).lower()
        assert s.shape == (100,)

    def test_fallback_device(self):
        """When generator is None, samples should land on fallback_device."""
        dist = D.Normal(0.0, 1.0)
        s = _sample_distribution(dist, (5,), generator=None, fallback_device="cpu")
        assert s.device == torch.device("cpu")
        assert s.shape == (5,)

    def test_batched_distribution(self):
        """A batched distribution should produce per-element samples."""
        gen = torch.Generator().manual_seed(0)
        low = torch.tensor([-1.0, -2.0, -3.0])
        high = torch.tensor([1.0, 2.0, 3.0])
        dist = D.Uniform(low, high)
        samples = torch.stack(
            [_sample_distribution(dist, (3,), gen) for _ in range(2000)]
        )
        # Each column should stay in its respective range
        assert samples[:, 0].min() >= -1.0
        assert samples[:, 0].max() <= 1.0
        assert samples[:, 1].min() >= -2.0
        assert samples[:, 1].max() <= 2.0
        assert samples[:, 2].min() >= -3.0
        assert samples[:, 2].max() <= 3.0


# ---------------------------------------------------------------------------
# MeshTransform.to() distribution handling
# ---------------------------------------------------------------------------


class TestDistributionDeviceTransfer:
    """Tests for MeshTransform.to() moving distribution parameters."""

    def test_to_cpu_preserves_distribution(self):
        """to('cpu') should produce a working distribution on CPU."""
        aug = _seed(RandomScaleMesh(distribution=D.Normal(1.0, 0.05)), 0)
        aug.to("cpu")
        factor = aug._sample_factor()
        assert factor.device == torch.device("cpu")
        assert factor.item() == pytest.approx(1.0, abs=0.3)

    def test_to_preserves_distribution_type(self):
        """to() should keep the same distribution class."""
        aug = RandomScaleMesh(distribution=D.Laplace(0.0, 1.0))
        aug.to("cpu")
        assert isinstance(aug._distribution, D.Laplace)

    def test_to_moves_batched_distribution(self):
        """to() should move batched distribution params."""
        dist = D.Uniform(
            torch.tensor([-1.0, -2.0, -3.0]),
            torch.tensor([1.0, 2.0, 3.0]),
        )
        aug = RandomTranslateMesh(distribution=dist)
        aug.to("cpu")
        assert aug._distribution.low.device == torch.device("cpu")
        assert aug._distribution.high.device == torch.device("cpu")

    def test_to_moves_generator(self):
        """to() should recreate the generator on the target device with the same seed."""
        aug = _seed(RandomScaleMesh(distribution=D.Normal(1.0, 0.05)), 42)
        original_seed = aug._generator.initial_seed()

        aug.to("cpu")

        assert aug._generator.device == torch.device("cpu")
        assert aug._generator.initial_seed() == original_seed


# ---------------------------------------------------------------------------
# RandomScaleMesh
# ---------------------------------------------------------------------------


class TestRandomScaleMesh:
    """Tests for RandomScaleMesh with distribution-based sampling."""

    def test_default_distribution(self):
        """Default distribution should be Uniform(0.9, 1.1)."""
        aug = RandomScaleMesh()
        assert isinstance(aug._distribution, D.Uniform)

    def test_normal_distribution_clusters_near_center(self):
        """Normal(1.0, 0.05) should produce scale factors near 1.0."""
        aug = _seed(RandomScaleMesh(distribution=D.Normal(1.0, 0.05)), 42)
        mesh = _simple_mesh_3d()
        factors = []
        for _ in range(500):
            scaled = aug(mesh)
            # point (1,0,0) scaled by factor -> (factor, 0, 0)
            factors.append(scaled.points[1, 0].item())
        factors_t = torch.tensor(factors)
        assert factors_t.mean().item() == pytest.approx(1.0, abs=0.02)
        assert factors_t.std().item() == pytest.approx(0.05, abs=0.02)

    def test_lognormal_always_positive(self):
        """LogNormal should always produce positive scale factors."""
        aug = _seed(RandomScaleMesh(distribution=D.LogNormal(0.0, 0.1)), 0)
        mesh = _simple_mesh_3d()
        for _ in range(100):
            scaled = aug(mesh)
            factor = scaled.points[1, 0].item()
            assert factor > 0.0

    def test_reproducibility(self):
        """Same seed should produce identical results."""
        mesh = _simple_mesh_3d()

        aug1 = _seed(RandomScaleMesh(distribution=D.Normal(1.0, 0.5)), 7)
        r1 = aug1(mesh)

        aug2 = _seed(RandomScaleMesh(distribution=D.Normal(1.0, 0.5)), 7)
        r2 = aug2(mesh)

        assert torch.allclose(r1.points, r2.points)

    def test_apply_to_domain_consistent(self):
        """apply_to_domain should use the same factor for interior and boundary."""
        domain = _simple_domain_3d()
        aug = _seed(RandomScaleMesh(distribution=D.Uniform(0.5, 2.0)), 0)
        scaled = aug.apply_to_domain(domain)
        # Interior and wall started with the same points, so after scaling
        # by the same factor they should still match.
        assert torch.allclose(scaled.interior.points, scaled.boundaries["wall"].points)

    def test_sequence_reproducibility(self):
        """Same seed should produce identical results over multiple calls."""
        mesh = _simple_mesh_3d()

        aug1 = _seed(RandomScaleMesh(distribution=D.Normal(1.0, 0.5)), 7)
        seq1 = [aug1(mesh).points.clone() for _ in range(10)]

        aug2 = _seed(RandomScaleMesh(distribution=D.Normal(1.0, 0.5)), 7)
        seq2 = [aug2(mesh).points.clone() for _ in range(10)]

        for s1, s2 in zip(seq1, seq2):
            assert torch.allclose(s1, s2)

    def test_apply_to_domain_reproducibility(self):
        """Same seed should produce identical domain results."""
        domain = _simple_domain_3d()

        aug1 = _seed(RandomScaleMesh(distribution=D.Uniform(0.5, 2.0)), 42)
        d1 = aug1.apply_to_domain(domain)

        aug2 = _seed(RandomScaleMesh(distribution=D.Uniform(0.5, 2.0)), 42)
        d2 = aug2.apply_to_domain(domain)

        assert torch.allclose(d1.interior.points, d2.interior.points)
        assert torch.allclose(
            d1.boundaries["wall"].points, d2.boundaries["wall"].points
        )

    def test_extra_repr(self):
        """extra_repr should mention the distribution."""
        aug = RandomScaleMesh(distribution=D.Normal(1.0, 0.05))
        assert "Normal" in aug.extra_repr()


# ---------------------------------------------------------------------------
# RandomTranslateMesh
# ---------------------------------------------------------------------------


class TestRandomTranslateMesh:
    """Tests for RandomTranslateMesh with distribution-based sampling."""

    def test_default_distribution(self):
        """Default distribution should be Uniform(-0.1, 0.1)."""
        aug = RandomTranslateMesh()
        assert isinstance(aug._distribution, D.Uniform)

    def test_laplace_distribution(self):
        """Laplace(0, 0.02) should produce offsets concentrated near zero."""
        aug = _seed(RandomTranslateMesh(distribution=D.Laplace(0.0, 0.02)), 0)
        mesh = _simple_mesh_3d()
        offsets = []
        for _ in range(500):
            translated = aug(mesh)
            # The first point starts at (0,0,0), so its position = offset
            offsets.append(translated.points[0].clone())
        offsets_t = torch.stack(offsets)
        assert offsets_t.mean(dim=0).abs().max().item() < 0.01
        assert offsets_t.abs().mean().item() == pytest.approx(0.02, abs=0.005)

    def test_batched_per_axis_distribution(self):
        """Batched Uniform should sample different ranges per axis."""
        dist = D.Uniform(
            torch.tensor([-1.0, -2.0, -3.0]),
            torch.tensor([1.0, 2.0, 3.0]),
        )
        aug = _seed(RandomTranslateMesh(distribution=dist), 0)
        mesh = _simple_mesh_3d()
        offsets = []
        for _ in range(1000):
            translated = aug(mesh)
            offsets.append(translated.points[0].clone())
        offsets_t = torch.stack(offsets)
        # Axis 0: range [-1, 1]
        assert offsets_t[:, 0].min() >= -1.0
        assert offsets_t[:, 0].max() <= 1.0
        # Axis 1: range [-2, 2]
        assert offsets_t[:, 1].min() >= -2.0
        assert offsets_t[:, 1].max() <= 2.0
        # Axis 2: range [-3, 3]
        assert offsets_t[:, 2].min() >= -3.0
        assert offsets_t[:, 2].max() <= 3.0

    def test_reproducibility(self):
        """Same seed should produce identical translations."""
        mesh = _simple_mesh_3d()

        aug1 = _seed(RandomTranslateMesh(distribution=D.Normal(0.0, 1.0)), 99)
        r1 = aug1(mesh)

        aug2 = _seed(RandomTranslateMesh(distribution=D.Normal(0.0, 1.0)), 99)
        r2 = aug2(mesh)

        assert torch.allclose(r1.points, r2.points)

    def test_sequence_reproducibility(self):
        """Same seed should produce identical results over multiple calls."""
        mesh = _simple_mesh_3d()

        aug1 = _seed(RandomTranslateMesh(distribution=D.Normal(0.0, 1.0)), 99)
        seq1 = [aug1(mesh).points.clone() for _ in range(10)]

        aug2 = _seed(RandomTranslateMesh(distribution=D.Normal(0.0, 1.0)), 99)
        seq2 = [aug2(mesh).points.clone() for _ in range(10)]

        for s1, s2 in zip(seq1, seq2):
            assert torch.allclose(s1, s2)

    def test_apply_to_domain_consistent(self):
        """apply_to_domain should use the same offset for all meshes."""
        domain = _simple_domain_3d()
        aug = _seed(RandomTranslateMesh(distribution=D.Uniform(-1.0, 1.0)), 0)
        translated = aug.apply_to_domain(domain)
        assert torch.allclose(
            translated.interior.points, translated.boundaries["wall"].points
        )

    def test_apply_to_domain_reproducibility(self):
        """Same seed should produce identical domain results."""
        domain = _simple_domain_3d()

        aug1 = _seed(RandomTranslateMesh(distribution=D.Laplace(0.0, 0.5)), 77)
        d1 = aug1.apply_to_domain(domain)

        aug2 = _seed(RandomTranslateMesh(distribution=D.Laplace(0.0, 0.5)), 77)
        d2 = aug2.apply_to_domain(domain)

        assert torch.allclose(d1.interior.points, d2.interior.points)
        assert torch.allclose(
            d1.boundaries["wall"].points, d2.boundaries["wall"].points
        )


# ---------------------------------------------------------------------------
# RandomRotateMesh
# ---------------------------------------------------------------------------


class TestRandomRotateMesh:
    """Tests for RandomRotateMesh with distribution-based sampling."""

    def test_default_distribution(self):
        """Default distribution should be Uniform(-pi, pi)."""
        aug = RandomRotateMesh()
        assert isinstance(aug._distribution, D.Uniform)

    def test_small_angle_gaussian(self):
        """Normal(0, 0.1) should produce small rotations near identity."""
        aug = _seed(
            RandomRotateMesh(distribution=D.Normal(0.0, 0.1), mode="axis_aligned"),
            0,
        )
        mesh = _simple_mesh_3d()
        original_points = mesh.points.clone()
        displacements = []
        for _ in range(200):
            rotated = aug(mesh)
            disp = (rotated.points - original_points).norm(dim=-1).max()
            displacements.append(disp.item())
        # Small angles -> small displacements (max point is at distance 1 from origin)
        avg_disp = sum(displacements) / len(displacements)
        assert avg_disp < 0.2

    def test_axis_restriction(self):
        """Rotation about z-axis only should not change z coordinates."""
        aug = _seed(
            RandomRotateMesh(
                axes=["z"],
                distribution=D.Uniform(-math.pi, math.pi),
                mode="axis_aligned",
            ),
            0,
        )
        mesh = _simple_mesh_3d()
        for _ in range(20):
            rotated = aug(mesh)
            assert torch.allclose(rotated.points[:, 2], mesh.points[:, 2], atol=1e-6)

    def test_uniform_mode_ignores_distribution(self):
        """mode='uniform' should work regardless of distribution parameter."""
        aug = _seed(
            RandomRotateMesh(mode="uniform", distribution=D.Normal(0.0, 0.01)),
            0,
        )
        mesh = _simple_mesh_3d()
        rotated = aug(mesh)
        # Should produce a valid rotation (points should have same norms)
        orig_norms = mesh.points.norm(dim=-1)
        rot_norms = rotated.points.norm(dim=-1)
        assert torch.allclose(orig_norms, rot_norms, atol=1e-5)

    def test_uniform_mode_orthogonal_matrix(self):
        """mode='uniform' should produce orthogonal rotation matrices (det=+1)."""
        aug = _seed(RandomRotateMesh(mode="uniform"), 42)
        # Sample several rotation matrices and check orthogonality
        for _ in range(20):
            R = aug._sample_uniform_rotation()
            assert R.shape == (3, 3)
            assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
            assert torch.det(R).item() == pytest.approx(1.0, abs=1e-5)

    def test_reproducibility(self):
        """Same seed should produce identical rotations."""
        mesh = _simple_mesh_3d()

        aug1 = _seed(
            RandomRotateMesh(distribution=D.Normal(0.0, 1.0), mode="axis_aligned"),
            55,
        )
        r1 = aug1(mesh)

        aug2 = _seed(
            RandomRotateMesh(distribution=D.Normal(0.0, 1.0), mode="axis_aligned"),
            55,
        )
        r2 = aug2(mesh)

        assert torch.allclose(r1.points, r2.points)

    def test_apply_to_domain_consistent(self):
        """apply_to_domain should use the same rotation for all meshes."""
        domain = _simple_domain_3d()
        aug = _seed(
            RandomRotateMesh(
                distribution=D.Uniform(-math.pi, math.pi), mode="axis_aligned"
            ),
            0,
        )
        rotated = aug.apply_to_domain(domain)
        assert torch.allclose(
            rotated.interior.points,
            rotated.boundaries["wall"].points,
            atol=1e-6,
        )

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            RandomRotateMesh(mode="bogus")

    def test_uniform_mode_3d_only(self):
        """mode='uniform' should reject non-3D meshes."""
        aug = RandomRotateMesh(mode="uniform")
        mesh_2d = Mesh(
            points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            cells=torch.tensor([[0, 1, 2]]),
        )
        with pytest.raises(ValueError, match="3-D meshes"):
            aug(mesh_2d)

    def test_extra_repr_axis_aligned(self):
        """extra_repr should mention axes and distribution."""
        aug = RandomRotateMesh(
            axes=["z"], distribution=D.Normal(0.0, 0.1), mode="axis_aligned"
        )
        r = aug.extra_repr()
        assert "z" in r
        assert "Normal" in r

    def test_sequence_reproducibility(self):
        """Same seed should produce identical results over multiple calls."""
        mesh = _simple_mesh_3d()

        aug1 = _seed(
            RandomRotateMesh(distribution=D.Normal(0.0, 1.0), mode="axis_aligned"),
            55,
        )
        seq1 = [aug1(mesh).points.clone() for _ in range(10)]

        aug2 = _seed(
            RandomRotateMesh(distribution=D.Normal(0.0, 1.0), mode="axis_aligned"),
            55,
        )
        seq2 = [aug2(mesh).points.clone() for _ in range(10)]

        for s1, s2 in zip(seq1, seq2):
            assert torch.allclose(s1, s2)

    def test_uniform_mode_reproducibility(self):
        """Same seed should produce identical uniform SO(3) rotations."""
        mesh = _simple_mesh_3d()

        aug1 = _seed(RandomRotateMesh(mode="uniform"), 12)
        r1 = aug1(mesh)

        aug2 = _seed(RandomRotateMesh(mode="uniform"), 12)
        r2 = aug2(mesh)

        assert torch.allclose(r1.points, r2.points)

    def test_uniform_mode_sequence_reproducibility(self):
        """Same seed should produce identical uniform rotation sequences."""
        mesh = _simple_mesh_3d()

        aug1 = _seed(RandomRotateMesh(mode="uniform"), 12)
        seq1 = [aug1(mesh).points.clone() for _ in range(10)]

        aug2 = _seed(RandomRotateMesh(mode="uniform"), 12)
        seq2 = [aug2(mesh).points.clone() for _ in range(10)]

        for s1, s2 in zip(seq1, seq2):
            assert torch.allclose(s1, s2)

    def test_apply_to_domain_reproducibility(self):
        """Same seed should produce identical domain rotations."""
        domain = _simple_domain_3d()

        aug1 = _seed(
            RandomRotateMesh(distribution=D.Uniform(-math.pi, math.pi)),
            0,
        )
        d1 = aug1.apply_to_domain(domain)

        aug2 = _seed(
            RandomRotateMesh(distribution=D.Uniform(-math.pi, math.pi)),
            0,
        )
        d2 = aug2.apply_to_domain(domain)

        assert torch.allclose(d1.interior.points, d2.interior.points, atol=1e-6)
        assert torch.allclose(
            d1.boundaries["wall"].points,
            d2.boundaries["wall"].points,
            atol=1e-6,
        )

    def test_extra_repr_uniform(self):
        """extra_repr for uniform mode should mention 'uniform'."""
        aug = RandomRotateMesh(mode="uniform")
        assert "uniform" in aug.extra_repr()


# ---------------------------------------------------------------------------
# Composed pipeline reproducibility
# ---------------------------------------------------------------------------


class TestPipelineReproducibility:
    """Tests that a composed augmentation pipeline is reproducible end-to-end."""

    def test_scale_translate_rotate_pipeline(self):
        """Chaining scale -> translate -> rotate with matched seeds is reproducible."""
        mesh = _simple_mesh_3d()

        def _build_pipeline(seed):
            return [
                _seed(RandomScaleMesh(distribution=D.Normal(1.0, 0.1)), seed),
                _seed(RandomTranslateMesh(distribution=D.Laplace(0.0, 0.05)), seed + 1),
                _seed(RandomRotateMesh(distribution=D.Normal(0.0, 0.3)), seed + 2),
            ]

        pipeline1 = _build_pipeline(42)
        pipeline2 = _build_pipeline(42)

        # Apply each pipeline for several iterations
        for _ in range(5):
            m1 = mesh
            for aug in pipeline1:
                m1 = aug(m1)
            m2 = mesh
            for aug in pipeline2:
                m2 = aug(m2)
            assert torch.allclose(m1.points, m2.points)

    def test_pipeline_different_seeds_differ(self):
        """Different seeds should produce different pipeline results."""
        mesh = _simple_mesh_3d()

        def _build_pipeline(seed):
            return [
                _seed(RandomScaleMesh(distribution=D.Normal(1.0, 0.1)), seed),
                _seed(RandomTranslateMesh(distribution=D.Laplace(0.0, 0.05)), seed + 1),
                _seed(RandomRotateMesh(distribution=D.Normal(0.0, 0.3)), seed + 2),
            ]

        pipeline1 = _build_pipeline(0)
        pipeline2 = _build_pipeline(999)

        m1 = mesh
        for aug in pipeline1:
            m1 = aug(m1)
        m2 = mesh
        for aug in pipeline2:
            m2 = aug(m2)
        assert not torch.allclose(m1.points, m2.points)

    def test_domain_pipeline_reproducibility(self):
        """Composed pipeline should be reproducible when applied to DomainMesh."""
        domain = _simple_domain_3d()

        def _build_pipeline(seed):
            return [
                _seed(RandomScaleMesh(distribution=D.Uniform(0.8, 1.2)), seed),
                _seed(RandomTranslateMesh(distribution=D.Normal(0.0, 0.1)), seed + 1),
                _seed(
                    RandomRotateMesh(distribution=D.Uniform(-math.pi, math.pi)),
                    seed + 2,
                ),
            ]

        pipeline1 = _build_pipeline(7)
        pipeline2 = _build_pipeline(7)

        d1 = domain
        for aug in pipeline1:
            d1 = aug.apply_to_domain(d1)
        d2 = domain
        for aug in pipeline2:
            d2 = aug.apply_to_domain(d2)

        assert torch.allclose(d1.interior.points, d2.interior.points, atol=1e-6)
        assert torch.allclose(
            d1.boundaries["wall"].points,
            d2.boundaries["wall"].points,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# DataLoader-driven pipeline reproducibility
# ---------------------------------------------------------------------------


class TestDataLoaderDrivenReproducibility:
    """Tests that the DataLoader seed drives the full pipeline reproducibly."""

    def test_mesh_dataset_set_generator_distributes(self, tmp_path):
        """set_generator should give independent generators to reader + transforms."""
        from physicsnemo.datapipes.mesh_dataset import MeshDataset
        from physicsnemo.datapipes.readers.mesh import MeshReader

        mesh = _simple_mesh_3d()
        mesh.save(tmp_path / "a.pt")
        mesh.save(tmp_path / "b.pt")

        reader = MeshReader(tmp_path, pattern="*.pt")
        transforms = [
            RandomScaleMesh(distribution=D.Uniform(0.5, 2.0)),
            RandomTranslateMesh(distribution=D.Uniform(-0.5, 0.5)),
        ]
        ds = MeshDataset(reader, transforms=transforms)

        master = torch.Generator().manual_seed(42)
        ds.set_generator(master)

        # Reader and both transforms should have received generators
        assert reader._subsample_generator is not None
        assert transforms[0]._generator is not None
        assert transforms[1]._generator is not None

        # Generators should have different seeds (independent forks)
        seeds = {
            reader._subsample_generator.initial_seed(),
            transforms[0]._generator.initial_seed(),
            transforms[1]._generator.initial_seed(),
        }
        assert len(seeds) == 3

    def test_dataloader_seed_produces_identical_sequences(self, tmp_path):
        """Two MeshDatasets seeded identically produce identical transform results."""
        from physicsnemo.datapipes.mesh_dataset import MeshDataset
        from physicsnemo.datapipes.readers.mesh import MeshReader

        mesh = _simple_mesh_3d()
        for i in range(4):
            mesh.save(tmp_path / f"s{i}.pt")

        def _build(seed):
            reader = MeshReader(tmp_path, pattern="*.pt")
            transforms = [RandomScaleMesh(distribution=D.Uniform(0.5, 2.0))]
            ds = MeshDataset(reader, transforms=transforms)
            gen = torch.Generator().manual_seed(seed)
            ds.set_generator(gen)
            return ds

        ds1 = _build(123)
        ds2 = _build(123)

        for i in range(len(ds1)):
            m1, _ = ds1[i]
            m2, _ = ds2[i]
            assert torch.allclose(m1.points, m2.points)

    def test_different_seeds_produce_different_results(self, tmp_path):
        """Two MeshDatasets with different seeds produce different results."""
        from physicsnemo.datapipes.mesh_dataset import MeshDataset
        from physicsnemo.datapipes.readers.mesh import MeshReader

        mesh = _simple_mesh_3d()
        mesh.save(tmp_path / "s0.pt")

        def _build(seed):
            reader = MeshReader(tmp_path, pattern="*.pt")
            transforms = [RandomScaleMesh(distribution=D.Uniform(0.5, 2.0))]
            ds = MeshDataset(reader, transforms=transforms)
            gen = torch.Generator().manual_seed(seed)
            ds.set_generator(gen)
            return ds

        ds1 = _build(0)
        ds2 = _build(999)

        m1, _ = ds1[0]
        m2, _ = ds2[0]
        assert not torch.allclose(m1.points, m2.points)

    def test_set_epoch_changes_randomness(self, tmp_path):
        """set_epoch reseeds transforms so different epochs differ."""
        from physicsnemo.datapipes.mesh_dataset import MeshDataset
        from physicsnemo.datapipes.readers.mesh import MeshReader

        mesh = _simple_mesh_3d()
        mesh.save(tmp_path / "s0.pt")

        reader = MeshReader(tmp_path, pattern="*.pt")
        transforms = [RandomScaleMesh(distribution=D.Uniform(0.5, 2.0))]
        ds = MeshDataset(reader, transforms=transforms)
        gen = torch.Generator().manual_seed(42)
        ds.set_generator(gen)

        ds.set_epoch(0)
        m0, _ = ds[0]

        ds.set_epoch(1)
        m1, _ = ds[0]

        # Different epochs should produce different scale factors
        assert not torch.allclose(m0.points, m1.points)

    def test_set_epoch_is_deterministic(self, tmp_path):
        """Resetting to the same epoch reproduces the same result."""
        from physicsnemo.datapipes.mesh_dataset import MeshDataset
        from physicsnemo.datapipes.readers.mesh import MeshReader

        mesh = _simple_mesh_3d()
        mesh.save(tmp_path / "s0.pt")

        def _run_epoch(seed, epoch):
            reader = MeshReader(tmp_path, pattern="*.pt")
            transforms = [RandomScaleMesh(distribution=D.Uniform(0.5, 2.0))]
            ds = MeshDataset(reader, transforms=transforms)
            gen = torch.Generator().manual_seed(seed)
            ds.set_generator(gen)
            ds.set_epoch(epoch)
            m, _ = ds[0]
            return m.points.clone()

        pts_a = _run_epoch(42, 5)
        pts_b = _run_epoch(42, 5)
        assert torch.allclose(pts_a, pts_b)
