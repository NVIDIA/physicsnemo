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

"""Checkpoint save / resume / init-from-checkpoint behaviour."""

import pytest
import torch
from omegaconf import OmegaConf

from train import _latest_checkpoint, _load_initial_state, _save_checkpoint


def _cfg(**training):
    return OmegaConf.create({"training": training})


def _save(tmp_path, *, epoch=5, best_val=0.3):
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # One step so the optimizer carries state (exp_avg buffers) to round-trip.
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    path = tmp_path / "ckpt.pt"
    _save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        lr_scheduler=None,
        ema=None,
        epoch=epoch,
        best_val=best_val,
        cfg=OmegaConf.create({"tag": "unit"}),
    )
    return model, path


def test_init_from_checkpoint_loads_weights_only(tmp_path):
    """init_from_checkpoint restores weights but starts a fresh run (epoch 0)."""
    src_model, path = _save(tmp_path, epoch=7, best_val=0.1)

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch, best_val = _load_initial_state(
        _cfg(init_from_checkpoint={"path": str(path), "strict": True}),
        model=model,
        optimizer=optimizer,
        lr_scheduler=None,
        ema=None,
        device=torch.device("cpu"),
        ckpt_dir=tmp_path,
    )
    assert start_epoch == 0
    assert best_val == float("inf")
    for a, b in zip(model.parameters(), src_model.parameters(), strict=True):
        assert torch.allclose(a, b)
    # Fresh optimizer -- no state carried over.
    assert len(optimizer.state) == 0


def test_resume_restores_full_state(tmp_path):
    """resume restores weights + optimizer + epoch/best_val to continue."""
    src_model, path = _save(tmp_path, epoch=5, best_val=0.3)

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch, best_val = _load_initial_state(
        _cfg(resume={"enabled": True, "checkpoint_path": str(path)}),
        model=model,
        optimizer=optimizer,
        lr_scheduler=None,
        ema=None,
        device=torch.device("cpu"),
        ckpt_dir=tmp_path,
    )
    assert start_epoch == 5
    assert best_val == 0.3
    for a, b in zip(model.parameters(), src_model.parameters(), strict=True):
        assert torch.allclose(a, b)
    # Optimizer state was restored.
    assert len(optimizer.state) > 0


def test_no_config_is_fresh_start(tmp_path):
    """With neither block enabled the run starts fresh."""
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch, best_val = _load_initial_state(
        _cfg(),
        model=model,
        optimizer=optimizer,
        lr_scheduler=None,
        ema=None,
        device=torch.device("cpu"),
        ckpt_dir=tmp_path,
    )
    assert start_epoch == 0
    assert best_val == float("inf")


def test_init_from_checkpoint_non_strict_allows_subset(tmp_path):
    """A partial checkpoint loads under strict=false and leaves the rest at init."""
    # Save a model with an extra parameter the target model does not have.
    big = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    optimizer = torch.optim.Adam(big.parameters(), lr=1e-3)
    path = tmp_path / "big.pt"
    _save_checkpoint(
        path=path,
        model=big,
        optimizer=optimizer,
        lr_scheduler=None,
        ema=None,
        epoch=1,
        best_val=1.0,
        cfg=OmegaConf.create({}),
    )

    small = torch.nn.Sequential(torch.nn.Linear(4, 4))  # only the first block
    start_epoch, best_val = _load_initial_state(
        _cfg(init_from_checkpoint={"path": str(path), "strict": False}),
        model=small,
        optimizer=torch.optim.Adam(small.parameters(), lr=1e-3),
        lr_scheduler=None,
        ema=None,
        device=torch.device("cpu"),
        ckpt_dir=tmp_path,
    )
    assert start_epoch == 0
    # First block loaded from the checkpoint.
    for a, b in zip(small[0].parameters(), big[0].parameters(), strict=True):
        assert torch.allclose(a, b)


def test_auto_resume_from_stable_ckpt_dir(tmp_path):
    """With no resume/init config, the latest epoch_*.pt in ckpt_dir is loaded.

    This is the stable-directory auto-resume that lets a resubmitted job
    continue without an explicit checkpoint path.
    """
    ckpt_dir = tmp_path / "checkpoints"
    src = torch.nn.Linear(4, 4)
    opt = torch.optim.Adam(src.parameters(), lr=1e-3)
    src(torch.randn(2, 4)).sum().backward()
    opt.step()
    _save_checkpoint(
        path=ckpt_dir / "epoch_0005.pt",
        model=src,
        optimizer=opt,
        lr_scheduler=None,
        ema=None,
        epoch=5,
        best_val=0.3,
        cfg=OmegaConf.create({}),
    )

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch, best_val = _load_initial_state(
        _cfg(),  # neither resume nor init configured -> auto-resume kicks in
        model=model,
        optimizer=optimizer,
        lr_scheduler=None,
        ema=None,
        device=torch.device("cpu"),
        ckpt_dir=ckpt_dir,
    )
    assert start_epoch == 5
    assert best_val == 0.3
    for a, b in zip(model.parameters(), src.parameters(), strict=True):
        assert torch.allclose(a, b)
    assert len(optimizer.state) > 0  # optimizer state restored (full resume)


def test_latest_checkpoint_picks_highest_epoch(tmp_path):
    """_latest_checkpoint returns the highest epoch, and None when absent."""
    ckpt_dir = tmp_path / "checkpoints"
    m = torch.nn.Linear(4, 4)
    o = torch.optim.Adam(m.parameters(), lr=1e-3)
    # 9999 vs 10000 is the case a lexicographic sort would get wrong
    # ("epoch_10000" < "epoch_9999"); the numeric parse must pick 10000.
    for ep in (5, 100, 50, 9999, 10000):
        _save_checkpoint(
            path=ckpt_dir / f"epoch_{ep:04d}.pt",
            model=m,
            optimizer=o,
            lr_scheduler=None,
            ema=None,
            epoch=ep,
            best_val=1.0,
            cfg=OmegaConf.create({}),
        )
    assert _latest_checkpoint(ckpt_dir).name == "epoch_10000.pt"
    assert _latest_checkpoint(tmp_path / "does_not_exist") is None


def test_resume_enabled_without_path_raises(tmp_path):
    """resume.enabled=true without checkpoint_path is a hard error, not a
    silent fall-through to auto-resume."""
    model = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="checkpoint_path"):
        _load_initial_state(
            _cfg(resume={"enabled": True}),  # enabled, but no checkpoint_path
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            lr_scheduler=None,
            ema=None,
            device=torch.device("cpu"),
            ckpt_dir=tmp_path / "empty",
        )


def test_latest_checkpoint_ignores_unparseable(tmp_path):
    """A stray non-numeric ``epoch_*.pt`` is skipped, not silently returned."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "epoch_abc.pt").write_text("")  # unparseable epoch
    # All entries unparseable -> None (fresh start), not the garbage file.
    assert _latest_checkpoint(ckpt_dir) is None
    m = torch.nn.Linear(4, 4)
    _save_checkpoint(
        path=ckpt_dir / "epoch_0007.pt",
        model=m,
        optimizer=torch.optim.Adam(m.parameters(), lr=1e-3),
        lr_scheduler=None,
        ema=None,
        epoch=7,
        best_val=1.0,
        cfg=OmegaConf.create({}),
    )
    # With a valid checkpoint present, the unparseable one is ignored.
    assert _latest_checkpoint(ckpt_dir).name == "epoch_0007.pt"
