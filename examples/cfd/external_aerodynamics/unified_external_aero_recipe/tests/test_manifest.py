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

"""Unit tests for manifest-mode helpers and consistency validation.

The synthetic-config E2E suite covers manifest mode indirectly via the
training-config plumbing, but the helpers below have enough internal
branching (JSON dict vs flat list vs plain text, distributed sharding
with / without ``drop_last``, deterministic shuffling, etc.) to warrant
focused unit coverage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import datasets as datasets_module
import pytest
from datasets import (
    ManifestSampler,
    _build_manifest_val_dataset,
    build_dataloaders,
    build_dataset,
    load_manifest,
    resolve_manifest_indices,
    resolve_manifest_spec,
    validate_dataset_consistency,
)
from omegaconf import DictConfig, OmegaConf

### ---------------------------------------------------------------------------
### load_manifest
### ---------------------------------------------------------------------------


class TestLoadManifest:
    """Tests for `datasets.load_manifest`."""

    def test_json_dict_with_split(self, tmp_path: Path):
        """JSON dict + ``split=name`` returns the named list, sorted."""
        path = tmp_path / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "train": ["run_3", "run_1", "run_2"],
                    "val": ["run_5"],
                }
            )
        )
        result = load_manifest(path, split="train")
        ### Sorted output regardless of input order.
        assert result == ["run_1", "run_2", "run_3"]

    def test_json_dict_without_split_raises(self, tmp_path: Path):
        """JSON dict + ``split=None`` is a config bug -- must raise."""
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"train": ["run_1"]}))
        with pytest.raises(ValueError, match="JSON dict"):
            load_manifest(path)

    def test_json_dict_unknown_split_raises(self, tmp_path: Path):
        """A typo in the split name surfaces an error listing the available keys."""
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"train": ["a"], "val": ["b"]}))
        with pytest.raises(KeyError, match="not found in manifest"):
            load_manifest(path, split="nonexistent")

    def test_json_flat_list(self, tmp_path: Path):
        """A bare JSON list is read as the entry list directly."""
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(["run_2", "run_1"]))
        result = load_manifest(path)
        assert result == ["run_1", "run_2"]

    def test_text_one_per_line(self, tmp_path: Path):
        """One-per-line text manifest, with comments and blank lines stripped."""
        path = tmp_path / "manifest.txt"
        path.write_text(
            "# this is a comment\nrun_2\n\nrun_1\n  # indented comment\nrun_3\n"
        )
        result = load_manifest(path)
        assert result == ["run_1", "run_2", "run_3"]

    def test_invalid_json_top_level_raises(self, tmp_path: Path):
        """JSON that's neither a dict nor a list is a structural error."""
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps("just a string"))
        with pytest.raises(ValueError, match="must be a list or dict"):
            load_manifest(path)


### ---------------------------------------------------------------------------
### resolve_manifest_indices
### ---------------------------------------------------------------------------


def _fake_reader(root: Path, paths: list[Path]) -> SimpleNamespace:
    """Minimal stand-in for ``MeshReader`` / ``DomainMeshReader``.

    ``resolve_manifest_indices`` only touches ``_root`` and ``_paths``,
    so a SimpleNamespace with those attributes is enough to exercise the
    matching logic without instantiating a real reader.
    """
    return SimpleNamespace(_root=root, _paths=paths)


class TestResolveManifestIndices:
    """Tests for `datasets.resolve_manifest_indices`."""

    def test_matches_intermediate_parent(self, tmp_path: Path):
        """A run directory in the middle of the path matches the manifest entry."""
        reader = _fake_reader(
            tmp_path,
            [
                tmp_path / "run_1" / "boundaries" / "surface",
                tmp_path / "run_2" / "boundaries" / "surface",
                tmp_path / "run_3" / "boundaries" / "surface",
            ],
        )
        result = resolve_manifest_indices(reader, ["run_1", "run_3"])
        assert result == [0, 2]

    def test_matches_immediate_parent(self, tmp_path: Path):
        """The leaf-file's parent dir name is also considered."""
        reader = _fake_reader(
            tmp_path,
            [
                tmp_path / "run_1" / "domain_1.pmsh",
                tmp_path / "run_2" / "domain_1.pmsh",
            ],
        )
        result = resolve_manifest_indices(reader, ["run_2"])
        assert result == [1]

    def test_no_matches_raises(self, tmp_path: Path):
        """An empty match set is always a config bug -- raise loudly."""
        reader = _fake_reader(
            tmp_path,
            [tmp_path / "run_1" / "domain_1.pmsh"],
        )
        with pytest.raises(ValueError, match="No reader paths matched"):
            resolve_manifest_indices(reader, ["never_matches"])

    def test_paths_outside_root_skipped(self, tmp_path: Path):
        """Reader paths not under the root are silently skipped (not crashed on)."""
        outside = tmp_path.parent / "outside" / "run_X" / "domain.pmsh"
        reader = _fake_reader(
            tmp_path,
            [
                outside,
                tmp_path / "run_1" / "domain.pmsh",
            ],
        )
        result = resolve_manifest_indices(reader, ["run_1"])
        ### Only the in-tree match is returned.
        assert result == [1]


### ---------------------------------------------------------------------------
### ManifestSampler
### ---------------------------------------------------------------------------


class TestManifestSampler:
    """Tests for `datasets.ManifestSampler`."""

    def test_no_shuffle_yields_original_order(self):
        """`shuffle=False` returns the indices unchanged on every epoch."""
        sampler = ManifestSampler([5, 2, 9], shuffle=False)
        assert list(sampler) == [5, 2, 9]
        sampler.set_epoch(7)
        assert list(sampler) == [5, 2, 9]

    def test_shuffle_is_deterministic_per_epoch(self):
        """Same ``seed + epoch`` yields the same permutation every time."""
        a = ManifestSampler([0, 1, 2, 3, 4, 5, 6, 7], shuffle=True, seed=42)
        b = ManifestSampler([0, 1, 2, 3, 4, 5, 6, 7], shuffle=True, seed=42)
        a.set_epoch(3)
        b.set_epoch(3)
        assert list(a) == list(b)

    def test_shuffle_changes_with_epoch(self):
        """Different epochs must (almost surely) produce different orders."""
        sampler = ManifestSampler([0, 1, 2, 3, 4, 5, 6, 7], shuffle=True, seed=0)
        sampler.set_epoch(0)
        order_0 = list(sampler)
        sampler.set_epoch(1)
        order_1 = list(sampler)
        assert order_0 != order_1
        ### Both orders must still be permutations of the input set.
        assert sorted(order_0) == sorted(order_1) == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_distributed_drop_last_truncates_evenly(self):
        """`drop_last=True` truncates so each rank gets the same count."""
        ### 7 indices across 3 ranks: drop_last takes the leading 6 (= 3*2).
        rank0 = ManifestSampler(
            list(range(7)), shuffle=False, rank=0, world_size=3, drop_last=True
        )
        rank1 = ManifestSampler(
            list(range(7)), shuffle=False, rank=1, world_size=3, drop_last=True
        )
        rank2 = ManifestSampler(
            list(range(7)), shuffle=False, rank=2, world_size=3, drop_last=True
        )
        out0, out1, out2 = list(rank0), list(rank1), list(rank2)
        assert len(out0) == len(out1) == len(out2) == 2
        ### Strided sharding: rank R sees positions [R, R+W, R+2W, ...]
        ### of the truncated 6-index list [0, 1, 2, 3, 4, 5].
        assert out0 == [0, 3]
        assert out1 == [1, 4]
        assert out2 == [2, 5]
        ### __len__ matches the actual count.
        assert len(rank0) == len(out0)

    def test_distributed_no_drop_last_pads(self):
        """`drop_last=False` pads with leading indices to make even shards."""
        ### 7 indices across 3 ranks: pad to 9, every rank yields 3.
        all_out = []
        for r in range(3):
            sampler = ManifestSampler(
                list(range(7)), shuffle=False, rank=r, world_size=3, drop_last=False
            )
            shard = list(sampler)
            assert len(shard) == 3
            assert len(sampler) == 3
            all_out.extend(shard)
        ### Padded list is [0..6, 0, 1]; concatenation of strided shards
        ### returns the same multiset.
        assert sorted(all_out) == sorted([0, 1, 2, 3, 4, 5, 6, 0, 1])


### ---------------------------------------------------------------------------
### validate_dataset_consistency
### ---------------------------------------------------------------------------


class TestValidateDatasetConsistency:
    """Tests for :func:`datasets.validate_dataset_consistency`."""

    @staticmethod
    def _first():
        return (
            {"pressure": "scalar", "wss": "vector"},
            ["l1", "l2", "mae"],
        )

    def test_matching_targets_metrics_is_silent(self, caplog):
        """All-equal blocks: no raise, no warning."""
        first_targets, first_metrics = self._first()
        with caplog.at_level(logging.WARNING):
            validate_dataset_consistency(
                ds_key="ds_b",
                ds_targets=dict(first_targets),
                ds_metrics=list(first_metrics),
                first_targets=first_targets,
                first_metrics=first_metrics,
            )
        assert caplog.records == []

    def test_targets_mismatch_raises(self):
        """Targets mismatch is the loss-correctness contract -- must raise."""
        first_targets, first_metrics = self._first()
        with pytest.raises(ValueError, match="does not match the first dataset"):
            validate_dataset_consistency(
                ds_key="ds_b",
                ds_targets={"pressure": "scalar"},  # missing wss
                ds_metrics=list(first_metrics),
                first_targets=first_targets,
                first_metrics=first_metrics,
            )

    def test_metrics_mismatch_warns(self, caplog):
        """Metrics mismatch is a soft drift -- warns, doesn't raise."""
        first_targets, first_metrics = self._first()
        with caplog.at_level(logging.WARNING, logger="training.datasets"):
            validate_dataset_consistency(
                ds_key="ds_b",
                ds_targets=dict(first_targets),
                ds_metrics=["l2"],  # softer
                first_targets=first_targets,
                first_metrics=first_metrics,
            )
        assert any("metrics=" in r.message for r in caplog.records)


### ---------------------------------------------------------------------------
### resolve_manifest_spec
### ---------------------------------------------------------------------------


class TestResolveManifestSpec:
    """Tests for :func:`datasets.resolve_manifest_spec`."""

    def test_directory_mode_returns_none(self):
        """Neither manifest style configured -> directory mode -> None."""
        ds_yaml = OmegaConf.create({"train_datadir": "/data/foo"})
        ds_block = OmegaConf.create({})
        assert resolve_manifest_spec(ds_yaml, ds_block) is None

    def test_directory_mode_with_sibling_manifest_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """A manifest next to the data that is about to be ignored is loud.

        Clearing the top-level split selectors is how a manifest/directory
        mix is configured, so this misconfiguration is easy to reach: it
        would silently sweep the manifest's held-out val/test runs into
        training.
        """
        (tmp_path / "manifest.json").write_text(
            json.dumps({"train": ["run_1"], "val": ["run_0"]})
        )
        ds_yaml = OmegaConf.create({"train_datadir": str(tmp_path)})
        with caplog.at_level(logging.WARNING, logger="training.datasets"):
            assert resolve_manifest_spec(ds_yaml, OmegaConf.create({})) is None
        assert any("manifest's splits are ignored" in r.message for r in caplog.records)

    def test_directory_mode_without_sibling_manifest_is_silent(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """Genuine directory mode must not nag."""
        ds_yaml = OmegaConf.create({"train_datadir": str(tmp_path)})
        with caplog.at_level(logging.WARNING, logger="training.datasets"):
            assert resolve_manifest_spec(ds_yaml, OmegaConf.create({})) is None
        assert not caplog.records

    def test_style_a_separate_files(self, tmp_path: Path):
        """``train_manifest`` / ``val_manifest`` (style A)."""
        train_path = tmp_path / "train.txt"
        val_path = tmp_path / "val.txt"
        train_path.write_text("run_1\n")
        val_path.write_text("run_2\n")
        ds_yaml = OmegaConf.create({})
        ds_block = OmegaConf.create(
            {
                "train_manifest": str(train_path),
                "val_manifest": str(val_path),
            }
        )
        spec = resolve_manifest_spec(ds_yaml, ds_block)
        assert spec is not None
        assert spec["train_manifest"] == str(train_path)
        assert spec["val_manifest"] == str(val_path)
        ### Style B fields are unset.
        assert spec["manifest"] is None
        assert spec["train_split"] is None

    def test_style_b_explicit_manifest_path(self, tmp_path: Path):
        """``manifest`` + ``train_split`` (style B, explicit path)."""
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"train": ["run_1"], "val": ["run_2"]}))
        ds_yaml = OmegaConf.create({})
        ds_block = OmegaConf.create(
            {
                "manifest": str(manifest),
                "train_split": "train",
                "val_split": "val",
            }
        )
        spec = resolve_manifest_spec(ds_yaml, ds_block)
        assert spec is not None
        assert spec["manifest"] == str(manifest)
        assert spec["train_split"] == "train"
        assert spec["val_split"] == "val"

    def test_style_b_auto_derives_manifest_from_datadir(self, tmp_path: Path):
        """When ``manifest`` is omitted, look for ``train_datadir/manifest.json``."""
        derived = tmp_path / "manifest.json"
        derived.write_text(json.dumps({"train": ["run_1"]}))
        ds_yaml = OmegaConf.create({"train_datadir": str(tmp_path)})
        ds_block = OmegaConf.create({"train_split": "train"})
        spec = resolve_manifest_spec(ds_yaml, ds_block)
        assert spec is not None
        assert spec["manifest"] == str(derived)
        assert spec["train_split"] == "train"

    def test_style_b_split_alone_without_derivable_manifest_raises(
        self, tmp_path: Path
    ):
        """Split key with no manifest path AND no sibling file -> raise.

        The user clearly intended manifest mode (they set ``train_split``);
        silently falling back to directory mode used to make the val loader
        iterate the train data when the dataset YAML had no ``val_datadir``,
        so we now fail loud at config-resolution time instead.
        """
        ds_yaml = OmegaConf.create({"train_datadir": str(tmp_path)})
        ds_block = OmegaConf.create({"train_split": "train"})
        with pytest.raises(ValueError, match="Manifest mode was requested"):
            resolve_manifest_spec(ds_yaml, ds_block)

    def test_val_split_alone_without_manifest_raises(self, tmp_path: Path):
        """A bare ``val_split`` is also a clear manifest-mode signal."""
        ds_yaml = OmegaConf.create({"train_datadir": str(tmp_path)})
        ds_block = OmegaConf.create({"val_split": "val"})
        with pytest.raises(ValueError, match="Manifest mode was requested"):
            resolve_manifest_spec(ds_yaml, ds_block)

    def test_directory_mode_unaffected_by_loud_failure(self, tmp_path: Path):
        """A fully-unset block remains valid directory mode (returns None)."""
        ds_yaml = OmegaConf.create({"train_datadir": str(tmp_path)})
        ds_block = OmegaConf.create({})
        assert resolve_manifest_spec(ds_yaml, ds_block) is None


### ---------------------------------------------------------------------------
### _build_manifest_val_dataset
### ---------------------------------------------------------------------------


class TestManifestValDataset:
    """Tests for :func:`datasets._build_manifest_val_dataset`.

    Manifest mode shares one reader across the train / val splits, so
    validation must not inherit the train augmentations. This mirrors
    directory mode, which always builds its val dataset with
    ``augment=False`` -- the asymmetry these tests lock down.
    """

    @staticmethod
    def _augmented_ds_yaml(datadir: Path) -> DictConfig:
        """Minimal manifest-style volume dataset YAML carrying augmentations.

        Trimmed to what the dataset builder inspects: the reader globs
        paths lazily (no file is opened at construction), so the directory
        only needs placeholder files, and the transform chain just needs a
        ``CenterMesh`` anchor plus the augmentations that get inserted
        after it.
        """
        return OmegaConf.create(
            {
                "train_datadir": str(datadir),
                "pipeline": {
                    "reader": {
                        "_target_": "${dp:DomainMeshReader}",
                        "path": "${train_datadir}",
                        "pattern": "run_*/domain_*.pdmsh",
                    },
                    "augmentations": [
                        {"_target_": "${dp:RandomRotateMesh}", "axes": ["z"]},
                        {"_target_": "${dp:RandomTranslateMesh}"},
                    ],
                    "transforms": [
                        {"_target_": "${dp:CenterMesh}"},
                    ],
                },
                "targets": {"pressure": "scalar"},
            }
        )

    @staticmethod
    def _make_datadir(tmp_path: Path, n_samples: int = 2) -> Path:
        """Create placeholder runs the reader can glob (it never opens them)."""
        for i in range(n_samples):
            run = tmp_path / f"run_{i}"
            run.mkdir()
            (run / f"domain_{i}.pdmsh").write_bytes(b"")
        return tmp_path

    def test_augment_off_returns_none(self, tmp_path: Path):
        """``augment=False`` -> val shares the train dataset (None sentinel)."""
        ds_yaml = self._augmented_ds_yaml(self._make_datadir(tmp_path))
        assert (
            _build_manifest_val_dataset(
                ds_yaml,
                augment=False,
                device=None,
                num_workers=1,
                pin_memory=False,
            )
            is None
        )

    def test_augment_on_returns_unaugmented_dataset(self, tmp_path: Path):
        """``augment=True`` -> a separate dataset whose chain has no augmentations."""
        ds_yaml = self._augmented_ds_yaml(self._make_datadir(tmp_path))

        ### Guard against a vacuous assertion: the train dataset must
        ### actually carry a stochastic augmentation for the val check to
        ### mean anything.
        train_ds = build_dataset(
            ds_yaml, augment=True, device=None, num_workers=1, pin_memory=False
        )
        assert any(getattr(t, "stochastic", False) for t in train_ds.transforms)

        val_ds = _build_manifest_val_dataset(
            ds_yaml, augment=True, device=None, num_workers=1, pin_memory=False
        )
        assert val_ds is not None
        ### A distinct object (own reader), not the train dataset.
        assert val_ds is not train_ds
        ### No stochastic (augmentation) transforms survive on the val chain.
        assert not any(getattr(t, "stochastic", False) for t in val_ds.transforms)
        ### ...but the deterministic CenterMesh transform is still present.
        assert any(type(t).__name__ == "CenterMesh" for t in val_ds.transforms)


class TestMultiDatasetManifestOffsets:
    """Local split indices map correctly into concatenated datasets."""

    _augmented_ds_yaml = staticmethod(TestManifestValDataset._augmented_ds_yaml)
    _make_datadir = staticmethod(TestManifestValDataset._make_datadir)

    @staticmethod
    def _patch_configs(
        monkeypatch: pytest.MonkeyPatch,
        configs: dict,
        *,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        monkeypatch.setattr(
            datasets_module,
            "load_dataset_config",
            lambda path: configs[path.stem],
        )
        monkeypatch.setattr(
            datasets_module,
            "DistributedManager",
            lambda: SimpleNamespace(world_size=world_size, rank=rank),
        )

    @staticmethod
    def _loader_cfg(
        primary: str,
        extras: list[str],
        *,
        train_split: str | None,
        val_split: str | None,
        augment: bool = False,
    ) -> DictConfig:
        return OmegaConf.create(
            {
                "dataset": primary,
                "extra_datasets": extras,
                "train_split": train_split,
                "val_split": val_split,
                "augment": augment,
                "sampling_resolution": None,
                "input_type": "mesh",
                "forward_kwargs": {"domain": ""},
                "training": {"batch_size": 1, "seed": 0},
                "dataloader": {
                    "num_workers": 1,
                    "pin_memory": False,
                    "prefetch_factor": 0,
                },
            }
        )

    def test_multiple_manifest_datasets_use_global_offsets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Every manifest contributes offset train and val selections."""
        first_root = tmp_path / "first"
        second_root = tmp_path / "second"
        first_root.mkdir()
        second_root.mkdir()
        self._make_datadir(first_root, n_samples=4)
        self._make_datadir(second_root, n_samples=3)
        (first_root / "manifest.json").write_text(
            json.dumps({"train": ["run_1", "run_3"], "val": ["run_0"]})
        )
        (second_root / "manifest.json").write_text(
            json.dumps({"train": ["run_0", "run_2"], "val": ["run_1"]})
        )
        configs = {
            "drivaer_ml_surface": self._augmented_ds_yaml(first_root),
            "shift_suv_estate_surface": self._augmented_ds_yaml(second_root),
        }
        self._patch_configs(monkeypatch, configs)

        train_loader, val_loader, _, _ = build_dataloaders(
            self._loader_cfg(
                "drivaer_ml_surface",
                ["shift_suv_estate_surface"],
                train_split="train",
                val_split="val",
                augment=True,
            )
        )

        ### The second dataset begins at global index 4 in both combined
        ### datasets: local train [0, 2] -> [4, 6], local val [1] -> [5].
        assert sorted(train_loader.sampler) == [1, 3, 4, 6]
        assert list(val_loader.sampler) == [0, 5]
        assert len(train_loader.dataset) == len(val_loader.dataset) == 7
        assert all(
            any(getattr(transform, "stochastic", False) for transform in ds.transforms)
            for ds in train_loader.dataset._datasets
        )
        assert all(
            not any(
                getattr(transform, "stochastic", False) for transform in ds.transforms
            )
            for ds in val_loader.dataset._datasets
        )

    def test_offsets_resolve_to_the_intended_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The offset indices address the runs the manifests actually name.

        Asserting index *numbers* only checks the arithmetic against
        itself. This walks ``MultiDataset``'s own global -> (dataset,
        local) mapping back to reader paths, which is the property that
        actually matters: mis-offset indices silently train on another
        dataset's samples.
        """
        roots = []
        for name, n_samples, manifest in (
            ("first", 4, {"train": ["run_2", "run_3"], "val": ["run_0", "run_1"]}),
            ("second", 3, {"train": ["run_0"], "val": ["run_1"]}),
            ("third", 2, {"train": ["run_1"], "val": ["run_0"]}),
        ):
            root = tmp_path / name
            root.mkdir()
            self._make_datadir(root, n_samples=n_samples)
            (root / "manifest.json").write_text(json.dumps(manifest))
            roots.append(root)
        names = [
            "drivaer_ml_surface",
            "shift_suv_estate_surface",
            "shift_suv_fastback_surface",
        ]
        self._patch_configs(
            monkeypatch,
            {name: self._augmented_ds_yaml(root) for name, root in zip(names, roots)},
        )

        train_loader, val_loader, _, _ = build_dataloaders(
            self._loader_cfg(names[0], names[1:], train_split="train", val_split="val")
        )

        def resolved(dataset, global_index: int) -> str:
            ds_idx, local = dataset._index_to_dataset_and_local(global_index)
            path = dataset._datasets[ds_idx].reader._paths[local]
            return str(path.relative_to(tmp_path))

        train_files = {resolved(train_loader.dataset, i) for i in train_loader.sampler}
        val_files = {resolved(val_loader.dataset, i) for i in val_loader.sampler}
        assert train_files == {
            "first/run_2/domain_2.pdmsh",
            "first/run_3/domain_3.pdmsh",
            "second/run_0/domain_0.pdmsh",
            "third/run_1/domain_1.pdmsh",
        }
        assert val_files == {
            "first/run_0/domain_0.pdmsh",
            "first/run_1/domain_1.pdmsh",
            "second/run_1/domain_1.pdmsh",
            "third/run_0/domain_0.pdmsh",
        }
        ### The pre-offset bug trained on the first dataset's held-out runs.
        assert train_files.isdisjoint(val_files)

    def test_distributed_ranks_shard_the_combined_indices(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Sharding partitions the concatenated selection, not one dataset's."""
        first_root = tmp_path / "first"
        second_root = tmp_path / "second"
        first_root.mkdir()
        second_root.mkdir()
        self._make_datadir(first_root, n_samples=4)
        self._make_datadir(second_root, n_samples=3)
        (first_root / "manifest.json").write_text(
            json.dumps({"train": ["run_1", "run_3"], "val": ["run_0"]})
        )
        (second_root / "manifest.json").write_text(
            json.dumps({"train": ["run_0", "run_2"], "val": ["run_1"]})
        )
        configs = {
            "drivaer_ml_surface": self._augmented_ds_yaml(first_root),
            "shift_suv_estate_surface": self._augmented_ds_yaml(second_root),
        }

        per_rank_train, per_rank_val = [], []
        for rank in range(2):
            with monkeypatch.context() as ctx:
                self._patch_configs(ctx, configs, world_size=2, rank=rank)
                train_loader, val_loader, _, _ = build_dataloaders(
                    self._loader_cfg(
                        "drivaer_ml_surface",
                        ["shift_suv_estate_surface"],
                        train_split="train",
                        val_split="val",
                    )
                )
                per_rank_train.append(list(train_loader.sampler))
                per_rank_val.append(list(val_loader.sampler))

        ### 4 train indices over 2 ranks divides evenly, so drop_last keeps
        ### all of them; each rank sees half, and no sample is duplicated.
        assert [len(shard) for shard in per_rank_train] == [2, 2]
        assert sorted(per_rank_train[0] + per_rank_train[1]) == [1, 3, 4, 6]
        assert set(per_rank_train[0]).isdisjoint(per_rank_train[1])
        ### Validation spans both datasets rather than one rank's slice.
        assert sorted(per_rank_val[0] + per_rank_val[1]) == [0, 5]

    @pytest.mark.parametrize("manifest_first", [True, False])
    def test_mixed_manifest_and_directory_offsets_are_split_specific(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        manifest_first: bool,
    ):
        """Directory ranges and manifest selections share each split's index space."""
        manifest_root = tmp_path / "manifest"
        directory_root = tmp_path / "directory"
        directory_val_root = tmp_path / "directory_val"
        for root, n_samples in (
            (manifest_root, 3),
            (directory_root, 4),
            (directory_val_root, 2),
        ):
            root.mkdir()
            self._make_datadir(root, n_samples=n_samples)
        train_manifest = tmp_path / "train.txt"
        val_manifest = tmp_path / "val.txt"
        train_manifest.write_text("run_2\n")
        val_manifest.write_text("run_0\n")
        manifest_name = "drivaer_ml_surface"
        directory_name = "shift_suv_estate_surface"
        configs = {
            manifest_name: OmegaConf.merge(
                self._augmented_ds_yaml(manifest_root),
                {
                    "train_manifest": str(train_manifest),
                    "val_manifest": str(val_manifest),
                },
            ),
            directory_name: OmegaConf.merge(
                self._augmented_ds_yaml(directory_root),
                {"val_datadir": str(directory_val_root)},
            ),
        }
        self._patch_configs(monkeypatch, configs)
        ordered_names = (
            [manifest_name, directory_name]
            if manifest_first
            else [directory_name, manifest_name]
        )

        train_loader, val_loader, _, _ = build_dataloaders(
            self._loader_cfg(
                ordered_names[0],
                ordered_names[1:],
                train_split=None,
                val_split=None,
            )
        )

        if manifest_first:
            ### Train offsets: manifest len=3; val offsets: manifest len=3.
            expected_train = [2, 3, 4, 5, 6]
            expected_val = [0, 3, 4]
        else:
            ### Train offsets: directory len=4; val offsets: directory val len=2.
            expected_train = [0, 1, 2, 3, 6]
            expected_val = [0, 1, 2]
        assert sorted(train_loader.sampler) == expected_train
        assert list(val_loader.sampler) == expected_val
        assert len(train_loader.dataset) == 7
        assert len(val_loader.dataset) == 5

    def test_missing_val_manifest_falls_back_before_offsetting(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        """A per-dataset train fallback cannot capture another dataset's indices."""
        first_root = tmp_path / "first"
        second_root = tmp_path / "second"
        first_root.mkdir()
        second_root.mkdir()
        self._make_datadir(first_root, n_samples=3)
        self._make_datadir(second_root, n_samples=2)
        first_train = tmp_path / "first_train.txt"
        second_train = tmp_path / "second_train.txt"
        second_val = tmp_path / "second_val.txt"
        first_train.write_text("run_1\n")
        second_train.write_text("run_0\n")
        second_val.write_text("run_1\n")
        configs = {
            "drivaer_ml_surface": OmegaConf.merge(
                self._augmented_ds_yaml(first_root),
                {"train_manifest": str(first_train)},
            ),
            "shift_suv_estate_surface": OmegaConf.merge(
                self._augmented_ds_yaml(second_root),
                {
                    "train_manifest": str(second_train),
                    "val_manifest": str(second_val),
                },
            ),
        }
        self._patch_configs(monkeypatch, configs)

        with caplog.at_level(logging.WARNING, logger="training.datasets"):
            train_loader, val_loader, _, _ = build_dataloaders(
                self._loader_cfg(
                    "drivaer_ml_surface",
                    ["shift_suv_estate_surface"],
                    train_split=None,
                    val_split=None,
                )
            )

        ### First local train/val [1]; second starts at 3 and contributes
        ### local train [0] / val [1].
        assert sorted(train_loader.sampler) == [1, 3]
        assert list(val_loader.sampler) == [1, 4]
        assert any("drivaer_ml_surface" in record.message for record in caplog.records)
