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
from physicsnemo.datapipes.protocols import (
    DatasetBase,
    HostPayload,
    preprocessing_stream,
    record_consumer_stream,
)
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
            Number of worker threads for the prefetch pool.  Worker threads
            run :meth:`_load_host` (disk read + pin_memory) concurrently;
            GPU operations (H2D transfer, transforms) always run on the
            main thread in :meth:`_consume`.
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
        with torch.profiler.record_function("MeshDataset._load: reader[index]"):
            data, metadata = self.reader[index]

        if self._device is not None:
            with torch.profiler.record_function("MeshDataset._load: data.to(device)"):
                data = data.to(self._device)

        for t in self.transforms:
            with torch.profiler.record_function(
                f"MeshDataset._load: transform {type(t).__name__}"
            ):
                if isinstance(data, DomainMesh):
                    data = t.apply_to_domain(data)
                else:
                    data = t(data)

        return data, metadata

    def __len__(self) -> int:
        return len(self.reader)

    # ------------------------------------------------------------------
    # Producer / consumer split (overrides DatasetBase defaults)
    # ------------------------------------------------------------------

    def _load_host(self, work_item: int) -> HostPayload:
        """Producer stage: read a mesh sample on a worker thread.

        Launches no device kernels: it only reads the raw sample. Device
        transfer and mesh transforms happen later in :meth:`_consume` on
        the consuming thread.

        Parameters
        ----------
        work_item : int
            Sample index to read from the reader.

        Returns
        -------
        HostPayload
            Payload carrying the host data and metadata, or a captured
            error.
        """
        try:
            data, metadata = self.reader[work_item]
            return HostPayload(work_item=work_item, data=data, metadata=metadata)
        except Exception as e:  # noqa: BLE001
            return HostPayload(work_item=work_item, error=e)

    def _consume(
        self,
        payload: HostPayload,
        stream: Optional[torch.cuda.Stream] = None,
        *,
        defer_sync: bool = False,
    ) -> tuple[Union[Mesh, DomainMesh, TensorDict], dict[str, Any]]:
        """Consumer stage: device transfer + transforms on the calling thread.

        Runs on whatever thread calls this (the main thread, so any device
        kernels in the transforms share the model's launching thread). When
        a CUDA ``stream`` is assigned, the host-to-device copy *and* the
        transforms run on that preprocessing stream via
        :func:`preprocessing_stream` -- so this sample's preprocessing
        overlaps the previous batch's training on the compute stream. A CUDA
        event orders the preprocessing before the compute stream (not a
        host-side synchronize), and the returned mesh's tensors are recorded
        against the compute stream (:meth:`torch.Tensor.record_stream`) so the
        caching allocator does not recycle their memory for later prep-stream
        samples while compute-stream reads are still pending.

        Parameters
        ----------
        payload : HostPayload
            Producer payload from :meth:`_load_host`.
        stream : torch.cuda.Stream, optional
            Preprocessing stream for the host-to-device transfer and
            transforms. ``None`` runs on the current stream.
        defer_sync : bool, default=False
            When False, ``compute_stream.wait_event`` is enqueued here so the
            result is immediately safe to use on the current stream. When
            True, the recorded event is appended to :attr:`_events_pending`
            and the DataLoader enqueues the wait just before the batch is
            yielded -- after the previous batch's model work -- so this
            batch's preprocessing overlaps the previous batch's compute
            instead of blocking it.

        Returns
        -------
        tuple[Mesh | DomainMesh | TensorDict, dict[str, Any]]
            Tuple of (transformed data, metadata dict).

        Raises
        ------
        Exception
            If the producer captured an error, re-raises it.
        """
        if payload.error is not None:
            raise payload.error

        data = payload.data
        metadata = payload.metadata

        def _apply_transforms(d: Any) -> Any:
            for t in self.transforms:
                if isinstance(d, DomainMesh):
                    d = t.apply_to_domain(d)
                else:
                    d = t(d)
            return d

        device_is_cuda = (
            self._device is not None and torch.device(self._device).type == "cuda"
        )
        use_stream = stream is not None and device_is_cuda
        compute_stream = torch.cuda.current_stream() if use_stream else None

        with preprocessing_stream(stream if use_stream else None):
            if self._device is not None:
                with torch.profiler.record_function(
                    "MeshDataset._consume: data.to(device)"
                ):
                    data = data.to(self._device, non_blocking=True)
            with torch.profiler.record_function(
                "MeshDataset._consume: _apply_transforms"
            ):
                data = _apply_transforms(data)

        if use_stream:
            # The compute stream will read these tensors (collate + model);
            # record it so the allocator does not recycle their blocks for
            # later prep-stream samples while those reads are still pending.
            record_consumer_stream(data, compute_stream)
            # Record an event marking the preprocessing's completion on the
            # prep stream.
            event = torch.cuda.Event()
            event.record(stream)
            if defer_sync:
                # Defer the compute-stream wait to the DataLoader so it lands
                # after the previous batch's model work (real overlap).
                self._events_pending.append(event)
            else:
                # Inline ordering for standalone callers (no DataLoader to
                # insert the wait at the right point).
                compute_stream.wait_event(event)

        return data, metadata

    def close(self) -> None:
        """Close the dataset, its reader, and stop prefetching.

        Waits for any in-flight prefetch tasks to complete before shutdown.
        """
        super().close()
        self.reader.close()
