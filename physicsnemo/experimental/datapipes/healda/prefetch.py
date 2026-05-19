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
"""Background-thread prefetch with a dedicated CUDA stream.

``prefetch_map`` wraps any iterable (typically a DataLoader) and applies a
transform function in a background thread using a separate CUDA stream.
This hides the CPU-to-GPU transfer and GPU featurization latency behind the
training forward/backward pass (better parallelsim). It works best when
`pin_memory=True` is used on the DataLoader.
"""

import dataclasses
import queue
import threading
from typing import Any, Callable, Iterable, Optional

import torch
from torch.utils.data import DataLoader


class _Done:
    pass


class _PrefetchIterator:
    """Process batches asynchronously in a background thread.

    The background thread uses a separate CUDA stream so that data movement
    and featurization do not block the main training stream.

    Args:
        dataloader: Any iterable of batches.
        transform: ``batch -> batch`` function to run on the background stream.
        queue_size: Bounded queue depth (default 2).
        cuda_stream: CUDA stream for background work (created if *None*).
    """

    def __init__(
        self,
        dataloader: Iterable,
        transform: Callable[[Any], Any],
        queue_size: int = 2,
        cuda_stream: Optional[torch.cuda.Stream] = None,
    ):
        self.dataloader = dataloader
        self.transform = transform
        self.queue_size = queue_size
        self.cuda_stream = cuda_stream or torch.cuda.Stream()

        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self.dataloader_iter = None
        self._started = False

    # -- background worker --------------------------------------------------

    def _worker(self):
        try:
            while not self.stop_event.is_set():
                try:
                    batch = next(self.dataloader_iter)
                except StopIteration:
                    self.queue.put((_Done, None))
                    break

                with torch.cuda.stream(self.cuda_stream):
                    processed_batch = self.transform(batch)

                self.cuda_stream.synchronize()
                self.queue.put((processed_batch, None))
        except Exception as e:
            self.queue.put((None, e))

    # -- lifecycle ----------------------------------------------------------

    def _start(self):
        if self._started:
            return
        self.dataloader_iter = iter(self.dataloader)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self._started = True

    def _stop(self):
        if not self._started:
            return
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self._started = False

    # -- iterator protocol --------------------------------------------------

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        self._start()
        return self

    @staticmethod
    def _record_stream(x, stream):
        """Tell the CUDA caching allocator that *stream* also owns *x*."""
        if isinstance(x, torch.Tensor):
            x.record_stream(stream)
        elif isinstance(x, list):
            for item in x:
                _PrefetchIterator._record_stream(item, stream)
        elif isinstance(x, dict):
            for item in x.values():
                _PrefetchIterator._record_stream(item, stream)
        elif dataclasses.is_dataclass(x):
            for field in dataclasses.fields(x):
                _PrefetchIterator._record_stream(getattr(x, field.name), stream)

    def __next__(self):
        if not self._started:
            raise RuntimeError("Iterator not started. Call __iter__ first.")

        batch, error = self.queue.get()

        if error is not None:
            raise error

        if batch is _Done:
            self._stop()
            raise StopIteration

        # Inform the allocator that the consumer stream also uses these tensors
        self._record_stream(batch, torch.cuda.current_stream())
        return batch

    def __del__(self):
        self._stop()


def prefetch_map(
    dataloader: DataLoader,
    transform: Callable[[Any], Any],
    queue_size: int = 2,
    cuda_stream: Optional[torch.cuda.Stream] = None,
) -> _PrefetchIterator:
    """Wrap a DataLoader with background prefetching and GPU transforms.

    Args:
        dataloader: Source of batches (typically a PyTorch DataLoader).
        transform: Function applied to each batch on a background CUDA stream.
        queue_size: Maximum number of pre-processed batches to buffer.
        cuda_stream: CUDA stream for background work (created if *None*).

    Returns:
        An iterable that yields pre-processed, GPU-resident batches.

    Example::

        loader = prefetch_map(
            dataloader,
            lambda batch: transform.device_transform(batch, device),
            queue_size=1,
        )
        for batch in loader:
            # batch is already on GPU
            loss = model(**batch)
    """
    return _PrefetchIterator(dataloader, transform, queue_size, cuda_stream)
