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

"""Data-source contract for StormCast/StormScope training.

The trainer needs exactly two things from a data source: **static metadata**
(channel names, image shape, invariants) and a **stream of batches**. It does
not need the source to be indexable. That split is expressed as two tiers:

* :class:`StormCastDataSource` -- the contract the trainer programs against:
  metadata, a sample count, and :meth:`~StormCastDataSource.make_loader`.
* :class:`StormCastDataset` -- a map-style (``__getitem__``) source whose
  ``make_loader`` defaults to the classic PyTorch :class:`~torch.utils.data
  .DataLoader`. Every existing dataset subclasses this and needs no changes.

A source opts into a different loading strategy (e.g. a PhysicsNeMo datapipe)
purely by overriding ``make_loader``; the trainer's call site is unconditional.

Batch contract, for every ``make_loader`` implementation:

* ``background``: ``(B, C_background, H, W)`` -- all conditioning that varies
  per sample lives here. This includes any low-resolution/coarse forcing
  *and* any past state a forecasting model needs as input: the dataset is
  responsible for concatenating those onto the channel axis before returning
  the batch. There is no separate "past state" tensor.
* ``state``: ``(B, C_state, H, W)`` -- the training target only.
* ``mask`` (optional): broadcastable to ``state``; ``1`` = valid pixel
* ``lead_time_label`` (optional): required when ``lead_time_steps > 0``
* ``scalar_conditions`` (optional): ``(B, C_scalar)``

Tensors must already be normalized (the training loop does not normalize,
for performance reasons). Batches may be on host or on ``spec.device``.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class LoaderSpec:
    """Everything the trainer knows about how a loader must behave.

    The trainer decides *what* it needs; the data source decides *how* to
    satisfy it. Fields that only apply to some backends (``prefetch_factor``,
    ``use_streams``) are advisory -- a backend that cannot honor one ignores it.

    Attributes
    ----------
    batch_size : int
        Number of *training samples* per batch on this rank, i.e. the local
        batch size. When the source declares a group axis (see
        :meth:`StormCastDataSource.sample_group_size`) the loader draws
        ``batch_size // group_size`` items and folds the group axis back in, so
        the tensor the trainer sees always has leading dim ``batch_size``.
    sampler : Iterable[int]
        Index stream to draw from. Already rank-sharded and, for training,
        infinite -- see :meth:`utils.parallel.ParallelHelper.shard_sampler`.
    num_workers : int
        Worker count. Processes for the PyTorch backend, threads for a datapipe.
    device : torch.device or None
        The training device. A backend may deliver batches already resident
        there; :meth:`utils.parallel.ParallelHelper.sharded_data_iter` is
        idempotent either way.
    seed : int or None
        Master seed for loader-side randomness (sampling, stochastic
        transforms). ``None`` means non-reproducible ambient randomness.
    drop_last : bool
        Drop a trailing partial batch.
    pin_memory : bool
        Stage host tensors in pinned memory for async host-to-device copies.
    prefetch_factor : int
        Batches kept in flight ahead of the consumer.
    use_streams : bool
        Datapipe backends only: overlap host-to-device copies and device-side
        transforms with compute on a side CUDA stream.
    backend : str
        Which loading strategy to build (``"torch"`` or ``"datapipes"``).
        Sources implementing only one may ignore this.
    options : Mapping[str, Any]
        Backend-specific extras passed through from ``cfg.dataset.loader``.
    """

    batch_size: int
    sampler: Iterable[int]
    num_workers: int = 0
    device: torch.device | None = None
    seed: int | None = None
    drop_last: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 2
    use_streams: bool = True
    backend: str = "torch"
    options: Mapping[str, Any] = field(default_factory=dict)


def resolve_group_size(
    source: "StormCastDataSource", spec: LoaderSpec
) -> tuple[int | None, int]:
    """Resolve a source's group axis against the requested batch size.

    Parameters
    ----------
    source : StormCastDataSource
        Source whose :meth:`~StormCastDataSource.sample_group_size` is consulted.
    spec : LoaderSpec
        The requested loader behavior.

    Returns
    -------
    tuple[int | None, int]
        ``(group_size, items_per_batch)``. ``group_size`` is ``None`` when
        samples carry no group axis, in which case ``items_per_batch`` is just
        ``spec.batch_size``.

    Raises
    ------
    ValueError
        If the group size is not positive, or does not divide ``batch_size``
        (which would make the flattened batch the wrong size).
    """
    # getattr so a plain map-style torch Dataset (which predates this contract)
    # can still be passed to default_torch_loader.
    group_size = getattr(source, "sample_group_size", None)
    group = group_size() if callable(group_size) else None
    if group is None:
        return None, spec.batch_size
    group = int(group)
    if group < 1:
        raise ValueError(f"sample_group_size() must be >= 1, got {group}")
    if spec.batch_size % group != 0:
        raise ValueError(
            f"local batch size {spec.batch_size} is not divisible by the "
            f"dataset's sample group size {group}. Each drawn item expands to "
            f"{group} training samples, so the batch size must be a multiple "
            "of it (lower crops_per_sample, or raise batch_size_per_gpu)."
        )
    return group, spec.batch_size // group


def _flatten_group(value: Any) -> Any:
    """Fold a collated group axis into the batch axis.

    ``(items, group, ...)`` becomes ``(items * group, ...)``. Non-tensor values
    pass through untouched. A tensor with fewer than two dimensions raises
    rather than being silently mis-sized -- see :func:`flatten_group_collate`.
    """
    if isinstance(value, torch.Tensor):
        return value.flatten(0, 1)
    return value


def flatten_group_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate samples that carry a leading group axis.

    Each sample is ``(group, ...)``-shaped; the default collate stacks them to
    ``(items, group, ...)`` and this folds the two leading axes together, so the
    trainer sees an ordinary ``(batch, ...)`` tensor and never learns that the
    source grouped anything.

    Every field of the sample must carry the group axis: a field without one
    would come out of the default collate as ``(items,)`` and be the wrong
    length after flattening, so :func:`_flatten_group` raises on it instead.
    """
    batch = torch.utils.data.default_collate(list(samples))
    return {key: _flatten_group(value) for key, value in batch.items()}


def default_torch_loader(
    source: "StormCastDataset", spec: LoaderSpec
) -> torch.utils.data.DataLoader:
    """Build the classic PyTorch DataLoader for a map-style source.

    This is the historical loading path, unchanged apart from honoring the
    source's group axis (:func:`resolve_group_size`).

    Parameters
    ----------
    source : StormCastDataset
        Map-style dataset to draw from.
    spec : LoaderSpec
        Requested loader behavior.

    Returns
    -------
    torch.utils.data.DataLoader
        Loader yielding batch mappings on the host.
    """
    group, items_per_batch = resolve_group_size(source, spec)
    return torch.utils.data.DataLoader(
        dataset=source,
        batch_size=items_per_batch,
        sampler=spec.sampler,
        num_workers=spec.num_workers,
        worker_init_fn=worker_init,
        drop_last=spec.drop_last,
        pin_memory=spec.pin_memory and torch.cuda.is_available(),
        prefetch_factor=spec.prefetch_factor if spec.num_workers > 0 else None,
        collate_fn=flatten_group_collate if group is not None else None,
    )


class StormCastDataSource(ABC):
    """The data-source contract the trainer programs against.

    Implementations provide static metadata, a count of drawable items, and a
    batch stream. Being indexable is *not* part of this contract -- see
    :class:`StormCastDataset` for the map-style tier that adds it.

    See the module docstring for the batch contract every :meth:`make_loader`
    implementation must satisfy.
    """

    lead_time_steps: int = 0  # number of lead time embeddings

    @abstractmethod
    def __len__(self) -> int:
        """Number of items the sampler may draw from.

        This counts *drawable items*, which is not the number of training
        samples when :meth:`sample_group_size` is set: an item then expands to
        ``sample_group_size()`` samples.
        """

    @abstractmethod
    def background_channels(self) -> list[str]:
        """Metadata for the background channels. A list of channel names, one for each channel"""

    @abstractmethod
    def state_channels(self) -> list[str]:
        """Metadata for the state channels. A list of channel names, one for each channel"""

    @abstractmethod
    def image_shape(self) -> tuple[int, int]:
        """Get the (height, width) of the data."""

    @abstractmethod
    def make_loader(self, spec: LoaderSpec) -> Iterable[Mapping[str, Any]]:
        """Return an iterable of batch mappings satisfying ``spec``.

        For training the iterable is expected to be **infinite** (the sampler
        cycles); the trainer bounds iteration itself.
        """

    def scalar_condition_channels(self) -> list[str]:
        """Metadata for the scalar condition channels. A list of channel names, one for each channel"""
        return []

    def latitude(self) -> np.ndarray:
        """Return a numpy array of the latitude of the data."""
        return np.full(self.image_shape(), np.nan)

    def longitude(self) -> np.ndarray:
        """Return a numpy array of the longitude of the data."""
        return np.full(self.image_shape(), np.nan)

    def normalize_background(
        self, x: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Convert background from physical units to normalized data."""
        return x

    def denormalize_background(
        self, x: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Convert background from normalized data to physical units."""
        return x

    def normalize_state(
        self, x: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Convert state from physical units to normalized data."""
        return x

    def denormalize_state(
        self, x: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Convert state from normalized data to physical units."""
        return x

    def get_invariants(self) -> np.ndarray | None:
        """Return invariants used for training, or None if no invariants are used."""
        return None

    def index_segments(self) -> list[int] | None:
        """Lengths of contiguous, independently-shardable index groups.

        Return ``None`` (the default) for a flat index space. A source that
        concatenates several sub-sources (e.g. geographic domains) should
        return their lengths, so the sampler can hand every rank a slice of
        *each* group rather than carving the concatenation into blocks that
        each fall inside a single group.

        The lengths must sum to ``len(self)``.
        """
        return None

    def sample_group_size(self) -> int | None:
        """Number of training samples packed into each drawable item.

        Return ``None`` (the default) when one item is one sample. Return an
        integer ``N`` when every item carries a leading group axis of size
        ``N`` -- e.g. several independent crops sharing one expensive read. The
        loader then draws ``batch_size // N`` items per batch and folds the
        group axis back into the batch axis, so the trainer is unaffected.

        Sources returning an integer must emit the axis on **every** field and
        for **every** value of ``N``, including ``N == 1``, so that batch shapes
        never depend on configuration.
        """
        return None


class StormCastDataset(StormCastDataSource, torch.utils.data.Dataset, ABC):
    """An abstract class that defines the interface for map-style StormCast datasets.

    All datasets must inherit from this class and implement the methods marked as
    @abstractmethod. The other methods have default implementations and can be
    overridden by the dataset if needed, for example to provide a normalization
    scheme.

    In addition to the methods defined below, all datasets must also implement the
    following:
    - `__init__`, which should accept a `params` argument containing the dataset
        parameters and a `train` argument indicating whether the dataset is for
        training or validation
    - `__len__`, which should return the number of samples in the dataset
    - `__getitem__`, which should return a dictionary containing the following
        keys:
        - `background`: a numpy.ndarray or torch.Tensor of shape
            `(num_channels_background, height, width)`. All conditioning goes
            here, including any past state used as input to a forecasting
            model -- concatenate it onto the channel axis before returning.
        - `state`: a numpy.ndarray or torch.Tensor of shape
            `(num_channels_state, height, width)` containing the training
            target only.
        - `lead_time_label` (optional): this must be returned if lead_time_steps > 0. A single
            integer indicating which lead time embedding should be used
        - `mask` (optional): a numpy.ndarray or torch.Tensor with values in {0, 1}
            (or boolean), where 1/True marks valid pixels and 0/False marks
            invalid/excluded pixels (e.g. outside sensor coverage, LAM padding zones,
            land-sea boundaries).  The shape must broadcast with `(num_channels_state,
            height, width)`: use `(1, height, width)` for a spatial mask shared across
            all channels, `(num_channels_state, height, width)` for per-channel spatial
            masks, or `(num_channels_state, 1, 1)` to mark entire channels as invalid.
            When provided, the training loop uses the mask as a per-pixel loss weight.
            For the DiT architecture with `use_nan_mask_tokens=True`, a spatial invalid
            mask is derived (any channel invalid → token invalid) and used to replace
            invalid-region tokens with learned mask tokens.  The dataset is responsible
            for producing this mask; caching internally is encouraged when the pattern
            is static across samples.

        The outputs of __getitem__ should be already normalized (this is not done in the
        training loop for performance reasons).

        When `sample_group_size()` returns an integer, every one of those arrays
        carries an extra *leading* group axis of that size.

    An example implementation of a dataset is given in `data_loader_hrrr_era5.py`.
    """

    def make_loader(self, spec: LoaderSpec) -> Iterable[Mapping[str, Any]]:
        """Build the classic PyTorch DataLoader over this map-style dataset.

        Override to supply a different loading strategy (e.g. a PhysicsNeMo
        datapipe) while keeping the map-style surface available for tests and
        for backend-parity checks.
        """
        return default_torch_loader(self, spec)


def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed % (2**32 - 1))
