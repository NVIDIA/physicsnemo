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

"""Non-blocking logging of scalars that live on the GPU."""

import functools
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from types import TracebackType
from typing import Any, NamedTuple

import torch
import torch.distributed as dist


class _Record(NamedTuple):
    """One :meth:`NonBlockingLogger.log` call, waiting for its copy to land."""

    keys: list[tuple[str, ...]]  # each scalar's key path, in the order logged
    packed: list[tuple[str, ...]]  # the same keys, in their order in ``host``
    host: torch.Tensor  # the packed scalars, valid once ``landed`` completes
    landed: torch.cuda.Event | None  # None: the copy was synchronous
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def _leaves(
    values: Mapping[str, Any], prefix: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], torch.Tensor]]:
    """Yield ``(key path, tensor)`` for each leaf of a nested mapping, in order."""
    for key, value in values.items():
        if isinstance(value, Mapping):
            yield from _leaves(value, (*prefix, key))
        else:
            yield (*prefix, key), value


class NonBlockingLogger:
    r"""Log scalars from the GPU without making the host wait for the GPU.

    Reading a GPU value with ``.item()`` makes the host wait until the GPU has
    run all the work queued so far, so the host cannot queue more work in the
    meantime. :meth:`log` instead packs a step's scalars into one buffer,
    optionally reduces it across ranks, and starts one non-blocking copy of it
    to pinned host memory, which PyTorch's caching host allocator reuses from
    step to step. ``log_fn`` runs for each call once its copy has landed,
    during that or a later :meth:`log` call or at :meth:`flush`, so records
    arrive in order, usually about one step late.

    Parameters
    ----------
    log_fn : Callable[..., None]
        Called as ``log_fn(values, *args, **kwargs)`` for each :meth:`log`
        call, where ``values`` is a nested ``dict`` of Python floats with the
        logged scalars' keys.
    reduce_op : torch.distributed.ReduceOp | None, optional
        If given, the scalars are all-reduced over ``group`` with this op, in
        the same buffer and before the copy. For example, ``ReduceOp.AVG``
        logs the mean over ranks. Every rank must then make the same
        :meth:`log` calls, with the same keys and dtypes. By default ``None``,
        which logs each rank's own values. A single process never reduces.
    group : torch.distributed.ProcessGroup | None, optional
        The process group for ``reduce_op``. By default ``None``, the default
        process group.

    Notes
    -----
    Leaving a ``with`` block flushes the logger, waiting for the copies still
    in flight. If the block raised, it logs only the records whose copies have
    landed, without waiting, since the device may be in an error state.

    Example
    -------
    >>> import torch
    >>> from physicsnemo.utils.logging import NonBlockingLogger
    >>> def show(values, step):
    ...     print(step, values)
    >>> with NonBlockingLogger(show) as logger:
    ...     for step in range(3):
    ...         loss = torch.tensor(2.0**-step)  # a GPU tensor in training
    ...         logger.log({"loss": loss}, step)
    0 {'loss': 1.0}
    1 {'loss': 0.5}
    2 {'loss': 0.25}
    """

    def __init__(
        self,
        log_fn: Callable[..., None],
        *,
        reduce_op: dist.ReduceOp | None = None,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        self.log_fn = log_fn
        self.reduce_op = reduce_op
        self.group = group
        self._pending: deque[_Record] = deque()

    def __enter__(self) -> "NonBlockingLogger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.flush(wait=exc_type is None)

    def log(self, values: Mapping[str, Any], /, *args: Any, **kwargs: Any) -> None:
        """Start copying ``values`` to the host and queue ``log_fn``'s call.

        It then calls ``flush(wait=False)``, which runs ``log_fn`` for each
        queued call whose copy has landed.

        Parameters
        ----------
        values : Mapping[str, Any]
            Plain one-element tensors on one device, in a mapping that may
            nest (a ``dict`` or a :class:`~tensordict.TensorDict`). Their
            values at this call are the ones logged. They are packed in a
            dtype of at least ``float32``, or ``float64`` if any is an integer
            or a bool, so integers up to ``2**53`` come back exact (as floats).
            Call ``full_tensor()`` on a ``DTensor`` or ``ShardTensor`` first.
        *args : Any
            Positional arguments passed on to ``log_fn`` as they are, such as
            the step number.
        **kwargs : Any
            Keyword arguments passed on to ``log_fn`` as they are.
        """
        keys, scalars = [], []
        for key, value in _leaves(values):
            scalar = value.detach() if isinstance(value, torch.Tensor) else value
            if type(scalar) is not torch.Tensor or scalar.numel() != 1:
                raise ValueError(
                    "NonBlockingLogger logs plain one-element tensors, but "
                    f"{'/'.join(key)!r} is a {type(value).__name__} of shape "
                    f"{tuple(getattr(value, 'shape', ()))}."
                )
            keys.append(key)
            scalars.append(scalar.reshape(()))
        ### Pack in key order, so every rank lays out the buffer identically.
        order = sorted(range(len(keys)), key=keys.__getitem__)
        host, landed = torch.empty(0), None
        if scalars:
            all_float = all(s.is_floating_point() for s in scalars)
            dtype = functools.reduce(
                torch.promote_types,
                (s.dtype for s in scalars),
                torch.float32 if all_float else torch.float64,
            )
            buffer = torch.stack([scalars[i].to(dtype) for i in order])
            if (
                self.reduce_op is not None
                and dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size(self.group) > 1
            ):
                dist.all_reduce(buffer, op=self.reduce_op, group=self.group)
            if buffer.is_cuda:
                ### A non-blocking copy to the host lands in pinned memory.
                host, landed = buffer.to("cpu", non_blocking=True), torch.cuda.Event()
                landed.record(torch.cuda.current_stream(buffer.device))
            else:
                host = buffer.cpu()
        self._pending.append(
            _Record(keys, [keys[i] for i in order], host, landed, args, kwargs)
        )
        self.flush(wait=False)

    def flush(self, *, wait: bool = True) -> None:
        """Run ``log_fn`` for the queued calls, in order.

        Parameters
        ----------
        wait : bool, optional
            If ``True`` (default), wait for the copies still in flight. If
            ``False``, stop at the first call whose copy has not landed.
        """
        while self._pending:
            record = self._pending[0]
            if record.landed is not None and not record.landed.query():
                if not wait:
                    return
                record.landed.synchronize()
            self._pending.popleft()
            numbers = dict(zip(record.packed, record.host.tolist()))
            values: dict[str, Any] = {}
            for key in record.keys:
                node = values
                for part in key[:-1]:
                    node = node.setdefault(part, {})
                node[key[-1]] = numbers[key]
            self.log_fn(values, *record.args, **record.kwargs)
