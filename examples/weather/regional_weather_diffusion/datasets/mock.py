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

from collections.abc import Iterable, Mapping
from typing import Any, Literal

import numpy as np
import torch

from .dataset import (
    LoaderSpec,
    StormCastDataset,
    default_torch_loader,
    resolve_group_size,
)


class _MockDataset(StormCastDataset):
    """A minimal mock dataset implementation for testing without real data.

    Args:
        num_state_channels: Number of channels in the state (target) data
        num_background_channels: Number of channels in the "real" background
            conditioning (e.g. a coarse forcing field). Only used for
            ``model_type`` values that include a background source.
        num_past_state_channels: Number of channels of past-state history to
            fold into the background tensor as extra channels. Only used for
            ``model_type`` values that include a past state.
        image_size: Tuple of (height, width) for the images
        num_samples: Number of samples in the dataset (default: 100)
        model_type: which channels ``__getitem__`` populates:

            - ``"hybrid"``: background = real background + past state
            - ``"nowcasting"``: background = past state only
            - ``"downscaling"``: background = real background only
            - ``"unconditional"``: no background key at all
    """

    def __init__(
        self,
        num_state_channels: int = 3,
        num_background_channels: int = 4,
        num_past_state_channels: int = 3,
        num_invariant_channels: int = 2,
        num_scalar_cond_channels: int = 2,
        image_size: tuple[int, int] = (32, 16),
        num_samples: int = 20,
        train: bool = True,
        model_type: Literal[
            "hybrid", "nowcasting", "downscaling", "unconditional"
        ] = "hybrid",
        use_mask: bool = False,
    ):
        self._num_state_channels = num_state_channels
        self._num_background_channels = num_background_channels
        self._num_past_state_channels = num_past_state_channels
        self._num_invariant_channels = num_invariant_channels
        self._num_scalar_cond_channels = num_scalar_cond_channels
        self._image_size = image_size
        self._num_samples = num_samples
        self._model_type = model_type
        self._use_mask = use_mask

    def __len__(self) -> int:
        return self._num_samples

    def _background_channel_count(self) -> int:
        if self._model_type == "hybrid":
            return self._num_background_channels + self._num_past_state_channels
        elif self._model_type == "nowcasting":
            return self._num_past_state_channels
        elif self._model_type == "downscaling":
            return self._num_background_channels
        else:  # "unconditional"
            return 0

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a sample with random data."""
        rng = np.random.default_rng(seed=idx)  # Use idx as seed for reproducibility

        height, width = self._image_size

        # Generate random state data (the training target).
        state_target = rng.normal(
            size=(self._num_state_channels, height, width)
        ).astype(np.float32)

        n_bg = self._background_channel_count()
        item: dict[str, Any] = {"state": state_target}
        if n_bg > 0:
            item["background"] = rng.normal(size=(n_bg, height, width)).astype(
                np.float32
            )

        # Generate scalar conditions
        if self._num_scalar_cond_channels:
            item["scalar_conditions"] = rng.normal(
                size=(self._num_scalar_cond_channels,)
            ).astype(np.float32)

        # Optional per-sample mask: right half of the domain is valid
        if self._use_mask:
            mask = np.zeros((1, height, width), dtype=np.float32)
            mask[:, :, width // 2 :] = 1.0
            item["mask"] = mask

        return item

    def background_channels(self) -> list[str]:
        """Return metadata for background channels."""
        return [f"background_{i}" for i in range(self._background_channel_count())]

    def state_channels(self) -> list[str]:
        """Return metadata for state channels."""
        return [f"state_{i}" for i in range(self._num_state_channels)]

    def scalar_condition_channels(self) -> list[str]:
        """Return metadata for state channels."""
        return [f"scalar_cond_{i}" for i in range(self._num_scalar_cond_channels)]

    def image_shape(self) -> tuple[int, int]:
        """Return the (height, width) of the data."""
        return self._image_size

    def get_invariants(self) -> np.ndarray | None:
        """Return invariants used for training."""
        if self._num_invariant_channels > 0:
            rng = np.random.default_rng(seed=42)
            return rng.normal(
                size=(
                    self._num_invariant_channels,
                    self._image_size[0],
                    self._image_size[1],
                )
            ).astype(np.float32)
        else:
            return None

    def make_loader(self, spec: LoaderSpec) -> Iterable[Mapping[str, Any]]:
        """Build the batch stream, honoring ``cfg.dataset.loader.backend``.

        ``"torch"`` is the historical map-style path (:func:`default_torch_loader`).
        ``"datapipes"`` builds a PhysicsNeMo :class:`~physicsnemo.datapipes.DataLoader`
        over the same synthetic samples, purely so this recipe's test suite can
        exercise (and demonstrate) the datapipe loading strategy end to end
        without needing real data on disk. Both backends draw from the same
        rank-sharded sampler and produce identical samples for a given index,
        so they are interchangeable from the trainer's point of view.
        """
        if spec.backend == "torch":
            return default_torch_loader(self, spec)
        if spec.backend == "datapipes":
            return self._make_datapipe_loader(spec)
        raise ValueError(
            f"unknown loader backend {spec.backend!r} (expected 'torch' or 'datapipes')"
        )

    def _make_datapipe_loader(self, spec: LoaderSpec):
        """Build a PhysicsNeMo datapipe loader over the same synthetic samples."""
        # Imported here so the torch backend has no hard dependency on the
        # datapipe stack.
        from physicsnemo.datapipes import DataLoader, Dataset, Reader

        mock = self

        class _MockReader(Reader):
            """Wrap ``_MockDataset.__getitem__`` as a datapipe Reader."""

            def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
                sample = mock[index]
                return {key: torch.as_tensor(value) for key, value in sample.items()}

            def __len__(self) -> int:
                return len(mock)

        dataset = Dataset(
            _MockReader(pin_memory=bool(spec.pin_memory)),
            transforms=[],
            device=None,  # synthetic data stays on host; no device-side transform
            # Clamped to >= 1 because the thread pool rejects a zero worker
            # count; num_workers=0 means "no worker processes" in PyTorch
            # terms, which has no equivalent here.
            num_workers=max(1, spec.num_workers),
        )
        # MockDataset carries no group axis, so items_per_batch == spec.batch_size;
        # resolve_group_size is still called for parity with default_torch_loader,
        # so a future group-axis dataset can copy this pattern unchanged.
        _, items_per_batch = resolve_group_size(self, spec)
        loader = DataLoader(
            dataset,
            batch_size=items_per_batch,
            sampler=spec.sampler,
            drop_last=spec.drop_last,
            prefetch_factor=spec.prefetch_factor,
            use_streams=spec.use_streams,
            seed=spec.seed,
        )
        # The trainer's batch-handling utilities (`nested_to`, `unpack_batch`)
        # expect plain dict/Tensor batches; convert the datapipe's TensorDict
        # output so both backends present an identical batch type.
        for batch in loader:
            yield dict(batch.items())


class MockDataset(_MockDataset):
    def __init__(self, params, train):
        super().__init__(train=train, **params)
