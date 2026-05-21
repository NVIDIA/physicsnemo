# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FGNDataset ABC and DataLoader worker initialiser.

Mirrors ``examples/weather/stormcast/datasets/dataset.py`` — same ABC pattern,
same ``worker_init`` seeding convention.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class FGNDataset(torch.utils.data.Dataset, ABC):
    """Abstract base class for all FGN training datasets.

    Subclasses must implement the five abstract methods below plus the standard
    ``torch.utils.data.Dataset`` protocol (``__len__`` and ``__getitem__``).
    ``__getitem__`` should return a dict with keys ``"history"``, ``"target"``,
    and ``"background"`` — already z-score normalized.
    """

    @abstractmethod
    def state_channels(self) -> list[str]:
        """Ordered list of state variable names (e.g. ``["t2m", "z500", ...]``)."""

    @abstractmethod
    def background_channels(self) -> list[str]:
        """Ordered list of background / conditioning variable names."""

    @abstractmethod
    def image_shape(self) -> tuple[int, int]:
        """Spatial grid size as ``(H, W)``."""

    def get_invariants(self) -> np.ndarray | None:
        """Static invariant channels as ``(C_inv, H, W)`` float32, or None."""
        return None

    def output_only_channels(self) -> list[int]:
        """Channel indices that must not be fed back as input (e.g. tp06)."""
        return []


def worker_init(wrk_id: int) -> None:
    np.random.seed(torch.utils.data.get_worker_info().seed % (2**32 - 1))
