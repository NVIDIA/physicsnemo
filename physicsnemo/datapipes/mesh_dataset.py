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

"""
MeshDataset - Combines a mesh reader (MeshReader or DomainMeshReader) with mesh transforms.

Returns (Mesh, metadata) or (DomainMesh, metadata). No key-based filtering.
Supports CUDA stream-aware prefetching for overlapped IO and computation.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
from tensordict import TensorDict

from physicsnemo.datapipes._rng import fork_generator
from physicsnemo.datapipes.protocols import DatasetBase, _PrefetchResult
from physicsnemo.datapipes.readers.mesh import DomainMeshReader, MeshReader
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.mesh import DomainMesh, Mesh


@register()
class MeshDataset(DatasetBase):
    r"""
    Dataset for mesh readers and mesh-only transforms.

    Accepts :class:`MeshReader` (single-mesh) or :class:`DomainMeshReader`
    (domain mesh with interior + boundaries).

    Applies a sequence of :class:`MeshTransform` (Mesh -> Mesh).
    For single-mesh data each transform is called directly.
    For :class:`DomainMesh` data each transform is applied via
    :meth:`MeshTransform.apply_to_domain`, which handles domain-level
    ``global_data``, consistent random parameter sampling, and
    proper centering semantics.

    Supports CUDA stream-aware prefetching: when a stream is provided to
    :meth:`prefetch`, device transfer and transforms run on that stream,
    allowing overlap with training computation.

    Examples
    --------
    >>> from physicsnemo.datapipes import DataLoader, MeshDataset, MeshReader
    >>>
    >>> reader = MeshReader("data/meshes/")  # doctest: +SKIP
    >>> dataset = MeshDataset(reader, transforms=[...], device="cuda")  # doctest: +SKIP
    >>> loader = DataLoader(dataset, batch_size=1, shuffle=True)  # doctest: +SKIP

    With DistributedSampler:

    >>> from torch.utils.data.distributed import DistributedSampler
    >>> sampler = DistributedSampler(dataset)  # doctest: +SKIP
    >>> loader = DataLoader(dataset, batch_size=1, sampler=sampler)  # doctest: +SKIP
    """

    def __init__(
        self,
        reader: MeshReader | DomainMeshReader,
        *,
        transforms: Sequence[MeshTransform] | None = None,
        device: str | torch.device | None = None,
        num_workers: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        reader : MeshReader or DomainMeshReader
            Mesh reader; returns (Mesh, metadata) or (DomainMesh, metadata).
        transforms : sequence of MeshTransform, optional
            Transforms to apply in order. None means no transforms.
        device : str or torch.device, optional
            If set, move mesh data to this device after loading (before transforms).
        num_workers : int, default=1
            Number of worker threads for prefetching. Defaults to 1
            because mesh transforms construct new Mesh objects internally
            and tensordict's ``_device_recorder`` is not safe for
            concurrent TensorDict construction across threads.
        """
        super().__init__(num_workers=num_workers)
        self.reader = reader
        self.transforms = list(transforms) if transforms else []
        self._device = torch.device(device) if isinstance(device, str) else device

        if self._device is not None:
            for t in self.transforms:
                if hasattr(t, "to"):
                    t.to(self._device)

    # ------------------------------------------------------------------
    # RNG management
    # ------------------------------------------------------------------

    def set_generator(self, generator: torch.Generator) -> None:
        """Distribute forked generators to the reader and every stochastic transform.

        Forks *generator* into ``1 + len(self.transforms)`` independent
        children: the first goes to the reader, the rest map 1-to-1 to
        the transform list (deterministic transforms silently ignore
        theirs).

        Parameters
        ----------
        generator : torch.Generator
            Parent generator (typically forked from the DataLoader's
            master generator).
        """
        n_children = 1 + len(self.transforms)
        children = fork_generator(generator, n_children)

        # Child 0 → reader
        if hasattr(self.reader, "set_generator"):
            self.reader.set_generator(children[0])

        # Children 1..N → transforms (deterministic ones ignore via base no-op)
        for child, t in zip(children[1:], self.transforms):
            if hasattr(t, "set_generator"):
                if self._device is not None and self._device != child.device:
                    dev_gen = torch.Generator(device=self._device)
                    dev_gen.manual_seed(child.initial_seed())
                    t.set_generator(dev_gen)
                else:
                    t.set_generator(child)

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch to the reader and every transform.

        Reseeds all generators assigned via :meth:`set_generator` so
        each epoch produces a different but deterministic random
        sequence.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        if hasattr(self.reader, "set_epoch"):
            self.reader.set_epoch(epoch)

        for t in self.transforms:
            if hasattr(t, "set_epoch"):
                t.set_epoch(epoch)

    # ------------------------------------------------------------------
    # DatasetBase implementation
    # ------------------------------------------------------------------

    def _load(
        self, index: int
    ) -> tuple[Union[Mesh, DomainMesh, TensorDict], dict[str, Any]]:
        """Synchronous load: reader -> device transfer -> transforms."""
        data, metadata = self.reader[index]

        if self._device is not None:
            data = data.to(self._device)

        for t in self.transforms:
            if isinstance(data, DomainMesh):
                data = t.apply_to_domain(data)
            else:
                data = t(data)

        return data, metadata

    def __len__(self) -> int:
        return len(self.reader)

    # ------------------------------------------------------------------
    # Stream-aware prefetch (overrides DatasetBase defaults)
    # ------------------------------------------------------------------

    def _load_and_transform(
        self,
        index: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> _PrefetchResult:
        """Load a sample and apply transforms with optional CUDA stream.

        Parameters
        ----------
        index : int
            Sample index.
        stream : torch.cuda.Stream, optional
            Optional CUDA stream for GPU operations.

        Returns
        -------
        _PrefetchResult
            Result with data, metadata, or error.
        """
        result = _PrefetchResult(index=index)

        try:
            data, metadata = self.reader[index]

            if self._device is not None:
                if stream is not None:
                    with torch.cuda.stream(stream):
                        data = data.to(self._device, non_blocking=True)
                else:
                    data = data.to(self._device, non_blocking=True)

            for t in self.transforms:
                if stream is not None:
                    with torch.cuda.stream(stream):
                        if isinstance(data, DomainMesh):
                            data = t.apply_to_domain(data)
                        else:
                            data = t(data)
                else:
                    if isinstance(data, DomainMesh):
                        data = t.apply_to_domain(data)
                    else:
                        data = t(data)

            if stream is not None:
                result.event = torch.cuda.Event()
                result.event.record(stream)

            result.data = data
            result.metadata = metadata

        except Exception as e:
            result.error = e

        return result

    def prefetch(
        self,
        index: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Start prefetching a sample asynchronously.

        When a CUDA stream is provided, GPU operations (device transfer
        and transforms) run on that stream for overlap with computation.

        Parameters
        ----------
        index : int
            Sample index to prefetch.
        stream : torch.cuda.Stream, optional
            Optional CUDA stream for GPU operations.
        """
        if index in self._prefetch_futures:
            return

        executor = self._ensure_executor()
        future = executor.submit(self._load_and_transform, index, stream)
        self._prefetch_futures[index] = future

    def __getitem__(
        self, index: int
    ) -> tuple[Union[Mesh, DomainMesh, TensorDict], dict[str, Any]]:
        """Get a transformed sample by index.

        If the index was prefetched, returns the prefetched result
        (waiting for completion if necessary). Otherwise loads synchronously.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        tuple[Mesh | DomainMesh | TensorDict, dict[str, Any]]
            Tuple of (transformed data, metadata dict).

        Raises
        ------
        Exception
            If prefetch failed, re-raises the error.
        """
        future = self._prefetch_futures.pop(index, None)

        if future is not None:
            result = future.result()

            if isinstance(result, _PrefetchResult):
                if result.error is not None:
                    raise result.error
                if result.event is not None:
                    torch.cuda.current_stream().wait_event(result.event)
                return result.data, result.metadata

            return result

        return self._load(index)

    def close(self) -> None:
        """Close the dataset and stop prefetching.

        Waits for any in-flight prefetch tasks to complete before shutdown.
        """
        super().close()
