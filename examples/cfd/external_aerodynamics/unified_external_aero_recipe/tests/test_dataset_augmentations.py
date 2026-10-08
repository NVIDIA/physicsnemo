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

"""The shipped dataset YAMLs' augmentations, as ``augment: true`` builds them."""

from pathlib import Path

import pytest
import torch
from datasets import build_dataset, load_dataset_config
from omegaconf import OmegaConf

from physicsnemo.datapipes.transforms.mesh import RandomTranslateMesh
from physicsnemo.mesh import Mesh

_DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
_AUGMENTED_DATASETS = sorted(
    path
    for path in _DATASETS_DIR.glob("*.yaml")
    if OmegaConf.select(OmegaConf.load(path), "pipeline.augmentations")
)


@pytest.mark.parametrize("yaml_path", _AUGMENTED_DATASETS, ids=lambda p: p.stem)
def test_augmentations_build_with_distribution_validation(
    yaml_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Augmentations build with ``torch.distributions`` validation on.

    The first ``torch.compile`` call in a process, including PhysicsNeMo's
    import-time decorators, turns validation off process-wide
    (pytorch/pytorch#157926), which would hide distribution parameters that
    ``Uniform`` rejects. Translation stays horizontal: ``x`` and ``y`` in
    ``[-1, 1]``, ``z`` exactly zero.
    """
    monkeypatch.setattr(torch.distributions.Distribution, "_validate_args", True)
    cfg = load_dataset_config(yaml_path)
    ### The reader only globs at construction, so one placeholder file that
    ### matches its pattern is enough.
    placeholder = tmp_path / cfg.pipeline.reader.pattern.replace("*", "x")
    placeholder.parent.mkdir(parents=True)
    placeholder.touch()
    cfg = OmegaConf.merge(
        cfg, {"train_datadir": str(tmp_path), "sampling_resolution": 100}
    )

    dataset = build_dataset(cfg, augment=True, device=None)

    (translate,) = [t for t in dataset.transforms if isinstance(t, RandomTranslateMesh)]
    translate.set_generator(torch.Generator().manual_seed(0))
    origin = Mesh(points=torch.zeros(1, 3))
    offsets = torch.cat([translate(origin).points for _ in range(100)])
    assert (offsets[:, 2] == 0).all()
    assert offsets[:, :2].abs().max() <= 1
