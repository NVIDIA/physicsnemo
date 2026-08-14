# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

r"""
test/models/test_ensemble_wrapper.py

Unit tests for ``physicsnemo.experimental.models.ensemble_wrapper``.

Following rules MOD-008a, MOD-008b, MOD-008c:
- MOD-008a: constructor / attribute tests
- MOD-008b: non-regression test with reference data
- MOD-008c: checkpoint loading test

Run with::

    pytest test/models/test_ensemble_wrapper.py -v
"""

import tempfile
from pathlib import Path

import pytest
import torch

from physicsnemo.models.mlp import FullyConnected
from physicsnemo.experimental.models.ensemble_wrapper import (
    EnsembleWrapper,
    EnsemblePrediction,
    EnsembleWrapperMeta,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IN_FEATURES  = 4
OUT_FEATURES = 2
BATCH_SIZE   = 16
N_MEMBERS    = 3


def make_members(n: int, seed_offset: int = 0) -> list:
    """Build N small FullyConnected members with deterministic seeds."""
    members = []
    for i in range(n):
        torch.manual_seed(i + seed_offset)
        m = FullyConnected(
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            num_layers=2,
            layer_size=16,
        )
        m.eval()
        members.append(m)
    return members


def make_input() -> torch.Tensor:
    torch.manual_seed(99)
    return torch.randn(BATCH_SIZE, IN_FEATURES)


# ---------------------------------------------------------------------------
# MOD-008a: Constructor and attribute tests
# ---------------------------------------------------------------------------


class TestEnsembleWrapperConstructor:
    r"""Following MOD-008a: constructor and attribute tests."""

    def test_construction_succeeds(self):
        """EnsembleWrapper should construct from a non-empty member list."""
        members = make_members(N_MEMBERS)
        ensemble = EnsembleWrapper(members)
        assert isinstance(ensemble, EnsembleWrapper)

    def test_n_members_property(self):
        """``n_members`` should reflect the number of members passed."""
        for n in [1, 3, 5]:
            members = make_members(n)
            assert EnsembleWrapper(members).n_members == n

    def test_empty_members_raises(self):
        """Passing an empty list should raise ``ValueError``."""
        with pytest.raises(ValueError, match="at least one"):
            EnsembleWrapper([])

    def test_metadata_type(self):
        """The wrapper's meta should be an ``EnsembleWrapperMeta`` instance."""
        ensemble = EnsembleWrapper(make_members(2))
        assert isinstance(ensemble.meta, EnsembleWrapperMeta)

    def test_members_registered_as_module_list(self):
        """Members must be stored in a ``torch.nn.ModuleList`` so that
        ``.parameters()`` and ``.to(device)`` cover all members."""
        import torch.nn as nn
        ensemble = EnsembleWrapper(make_members(N_MEMBERS))
        assert isinstance(ensemble.members, nn.ModuleList)
        assert len(ensemble.members) == N_MEMBERS

    def test_to_device_moves_all_members(self):
        """Calling ``.to(device)`` should move all member parameters."""
        ensemble = EnsembleWrapper(make_members(2))
        ensemble.to("cpu")
        for member in ensemble.members:
            for param in member.parameters():
                assert param.device.type == "cpu"


# ---------------------------------------------------------------------------
# MOD-008b: Non-regression tests with reference data
# ---------------------------------------------------------------------------


class TestEnsembleWrapperForward:
    r"""Following MOD-008b: non-regression tests with reference data."""

    def test_forward_output_shape(self):
        """``forward`` should return a tensor of shape ``(B, out_features)``."""
        ensemble = EnsembleWrapper(make_members(N_MEMBERS))
        x = make_input()
        with torch.no_grad():
            out = ensemble(x)
        assert out.shape == (BATCH_SIZE, OUT_FEATURES)

    def test_forward_returns_mean(self):
        """``forward`` must equal ``predict_with_uncertainty().mean``."""
        ensemble = EnsembleWrapper(make_members(N_MEMBERS))
        x = make_input()
        with torch.no_grad():
            fwd  = ensemble(x)
            uq   = ensemble.predict_with_uncertainty(x)
        torch.testing.assert_close(fwd, uq.mean)

    def test_predict_with_uncertainty_shapes(self):
        """``predict_with_uncertainty`` should return correct tensor shapes."""
        ensemble = EnsembleWrapper(make_members(N_MEMBERS))
        x = make_input()
        with torch.no_grad():
            result = ensemble.predict_with_uncertainty(x)

        assert isinstance(result, EnsemblePrediction)
        assert result.mean.shape        == (BATCH_SIZE, OUT_FEATURES)
        assert result.std.shape         == (BATCH_SIZE, OUT_FEATURES)
        assert result.predictions.shape == (N_MEMBERS, BATCH_SIZE, OUT_FEATURES)

    def test_std_non_negative(self):
        """Standard deviation values must be non-negative everywhere."""
        ensemble = EnsembleWrapper(make_members(N_MEMBERS))
        x = make_input()
        with torch.no_grad():
            result = ensemble.predict_with_uncertainty(x)
        assert (result.std >= 0).all()

    def test_std_zero_for_identical_members(self):
        """When all members are identical the std should be (near) zero."""
        torch.manual_seed(0)
        single = FullyConnected(in_features=IN_FEATURES, out_features=OUT_FEATURES,
                                num_layers=2, layer_size=16)
        # Three copies of the exact same weights
        members = [single, single, single]
        ensemble = EnsembleWrapper(members)
        x = make_input()
        with torch.no_grad():
            result = ensemble.predict_with_uncertainty(x)
        assert result.std.abs().max().item() < 1e-5

    def test_std_nonzero_for_different_members(self):
        """Different weight initialisations should yield non-zero std."""
        ensemble = EnsembleWrapper(make_members(N_MEMBERS, seed_offset=100))
        x = make_input()
        with torch.no_grad():
            result = ensemble.predict_with_uncertainty(x)
        assert result.std.mean().item() > 0

    def test_mean_equals_average_of_predictions(self):
        """``mean`` must be the arithmetic mean of ``predictions``."""
        ensemble = EnsembleWrapper(make_members(N_MEMBERS))
        x = make_input()
        with torch.no_grad():
            result = ensemble.predict_with_uncertainty(x)
        expected_mean = result.predictions.mean(dim=0)
        torch.testing.assert_close(result.mean, expected_mean)

    def test_single_member_std_is_zero(self):
        """With one member the std must be zero (no variance to estimate)."""
        ensemble = EnsembleWrapper(make_members(1))
        x = make_input()
        with torch.no_grad():
            result = ensemble.predict_with_uncertainty(x)
        assert result.std.abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# MOD-008c: Checkpoint loading test
# ---------------------------------------------------------------------------


class TestEnsembleWrapperCheckpoints:
    r"""Following MOD-008c: checkpoint loading tests."""

    def test_from_checkpoints_loads_correctly(self, tmp_path):
        """``from_checkpoints`` should reproduce an ensemble identical to the
        original when weights are saved and reloaded."""
        members = make_members(N_MEMBERS)
        original = EnsembleWrapper(members)

        # Save each member's state_dict
        paths = []
        for i, member in enumerate(members):
            p = tmp_path / f"member_{i}.pt"
            torch.save(member.state_dict(), p)
            paths.append(p)

        # Reload via from_checkpoints
        loaded = EnsembleWrapper.from_checkpoints(
            model_cls=FullyConnected,
            checkpoint_paths=paths,
            map_location="cpu",
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            num_layers=2,
            layer_size=16,
        )

        # Predictions must be identical
        x = make_input()
        with torch.no_grad():
            r_orig   = original.predict_with_uncertainty(x)
            r_loaded = loaded.predict_with_uncertainty(x)

        torch.testing.assert_close(r_orig.mean, r_loaded.mean)
        torch.testing.assert_close(r_orig.std,  r_loaded.std)

    def test_from_checkpoints_missing_file_raises(self, tmp_path):
        """A ``FileNotFoundError`` must be raised for a missing checkpoint."""
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            EnsembleWrapper.from_checkpoints(
                model_cls=FullyConnected,
                checkpoint_paths=[tmp_path / "does_not_exist.pt"],
                in_features=IN_FEATURES,
                out_features=OUT_FEATURES,
                num_layers=2,
                layer_size=16,
            )

    def test_from_checkpoints_n_members(self, tmp_path):
        """Ensemble size must match the number of checkpoint paths."""
        members = make_members(4)
        paths = []
        for i, m in enumerate(members):
            p = tmp_path / f"m_{i}.pt"
            torch.save(m.state_dict(), p)
            paths.append(p)

        loaded = EnsembleWrapper.from_checkpoints(
            model_cls=FullyConnected,
            checkpoint_paths=paths,
            map_location="cpu",
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            num_layers=2,
            layer_size=16,
        )
        assert loaded.n_members == 4
