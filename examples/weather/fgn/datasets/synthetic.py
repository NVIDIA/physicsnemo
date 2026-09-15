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

"""Synthetic FGN dataset — random tensors for fast CPU smoke tests."""

from __future__ import annotations

import numpy as np
import torch

from .dataset import FGNDataset


class SyntheticFGNDataset(FGNDataset):
    """Returns random Gaussian batches; no data files needed.

    Parameters come from the dataset config section and are expected in
    ``cfg`` as plain attributes (Hydra DictConfig or dataclass).
    """

    def __init__(self, cfg, train: bool = True):
        self._state_channels = list(cfg.state_channels)
        self._background_channels = list(getattr(cfg, "background_channels", []))
        self._invariant_channels = list(getattr(cfg, "invariant_channels", []))
        self._H = int(cfg.image_height)
        self._W = int(cfg.image_width)
        self._history_frames = int(cfg.history_frames)
        self._future_frames = int(cfg.future_frames)
        self._length = int(cfg.num_samples if train else max(1, cfg.num_samples // 4))

    def state_channels(self) -> list[str]:
        return self._state_channels

    def background_channels(self) -> list[str]:
        return self._background_channels

    def image_shape(self) -> tuple[int, int]:
        return self._H, self._W

    def get_invariants(self) -> np.ndarray | None:
        if not self._invariant_channels:
            return None
        return np.zeros(
            (len(self._invariant_channels), self._H, self._W), dtype=np.float32
        )

    def output_only_channels(self) -> list[int]:
        return []

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        C = len(self._state_channels)
        Cb = len(self._background_channels)
        H, W = self._H, self._W
        return {
            "history": torch.randn(self._history_frames, C, H, W),
            "target": torch.randn(self._future_frames, C, H, W),
            "background": torch.randn(Cb, H, W) if Cb else torch.zeros(0, H, W),
        }
