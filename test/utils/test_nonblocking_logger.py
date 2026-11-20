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

"""Tests for :class:`physicsnemo.utils.logging.NonBlockingLogger`."""

import os
import re
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

from physicsnemo.utils.logging import NonBlockingLogger

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

### GPU clock cycles for torch.cuda._sleep: long enough (0.1 s or more) that
### copies queued behind it are still in flight when the test checks.
_BUSY_CYCLES = 500_000_000


def _recorder():
    """A ``log_fn`` that appends ``(values, args, kwargs)`` to a list."""
    calls = []
    return calls, lambda values, *args, **kwargs: calls.append((values, args, kwargs))


def _count_waits(monkeypatch):
    """Count explicit event, stream and device waits; returns the list."""
    waits = []
    for owner, name in (
        (torch.cuda.Event, "synchronize"),
        (torch.cuda.Stream, "synchronize"),
        (torch.cuda, "synchronize"),
    ):
        original = getattr(owner, name)

        def counted(*args, _original=original, _name=f"{owner}.{name}", **kwargs):
            waits.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(owner, name, counted)
    return waits


def test_logs_nested_values_in_order_as_python_numbers(device):
    """Each call reaches ``log_fn`` once, in order, with its args, and with its
    values as nested dicts of Python numbers in the order they were given
    (packing sorts the keys), bitwise equal to ``.item()``."""
    calls, log_fn = _recorder()
    expected = []
    with NonBlockingLogger(log_fn) as logger:
        for step in range(4):
            loss = torch.tensor(1.37, device=device) / (step + 1)
            metrics = TensorDict({"z_l2": loss.sqrt(), "a_l2": loss * 3})
            logger.log({"loss": loss, "metrics": metrics}, step, phase="train")
            expected.append(
                (
                    {
                        "loss": loss.item(),
                        "metrics": {
                            "z_l2": metrics["z_l2"].item(),
                            "a_l2": metrics["a_l2"].item(),
                        },
                    },
                    (step,),
                    {"phase": "train"},
                )
            )
    assert calls == expected
    assert list(calls[0][0]["metrics"]) == ["z_l2", "a_l2"]
    assert all(type(values["loss"]) is float for values, _, _ in calls)


def test_integers_and_float64_values_stay_exact(device):
    """Next to float32 and 16-bit values, integers (here above 2**24, past
    float32's exact range), bools and float64 values are logged exactly."""
    calls, log_fn = _recorder()
    fine = 1 + 2.0**-40  # not representable in float32
    with NonBlockingLogger(log_fn) as logger:
        logger.log(
            {
                "f32": torch.tensor(0.5, device=device),
                "count": torch.tensor(2**24 + 1, device=device),
                "bool": torch.tensor(True, device=device),
            }
        )
        logger.log(
            {
                "bf16": torch.tensor(1.0078125, dtype=torch.bfloat16, device=device),
                "f64": torch.tensor(fine, dtype=torch.float64, device=device),
            }
        )
    assert [values for values, _, _ in calls] == [
        {"f32": 0.5, "count": 2.0**24 + 1, "bool": 1.0},
        {"bf16": 1.0078125, "f64": fine},
    ]


def test_values_are_read_when_logged(device):
    """A value changed in place after :meth:`log` is logged as it was."""
    calls, log_fn = _recorder()
    total = torch.zeros((), device=device)
    with NonBlockingLogger(log_fn) as logger:
        for step in range(3):
            total += step + 1
            logger.log({"total": total})
    assert [values["total"] for values, _, _ in calls] == [1.0, 3.0, 6.0]


def test_empty_values_still_call_log_fn():
    calls, log_fn = _recorder()
    with NonBlockingLogger(log_fn) as logger:
        logger.log({}, 3)
    assert calls == [({}, (3,), {})]


class _Subclass(torch.Tensor):
    """Stands in for tensor subclasses such as ``DTensor``."""


@pytest.mark.parametrize(
    "value, kind",
    [
        (torch.zeros(3), "Tensor of shape (3,)"),
        (torch.zeros(2, 2), "Tensor of shape (2, 2)"),
        (torch.zeros(()).as_subclass(_Subclass), "_Subclass of shape ()"),
        (0.5, "float of shape ()"),
    ],
)
def test_rejects_values_other_than_plain_scalar_tensors(value, kind):
    """:meth:`log` raises at once, naming the value, rather than when the
    record is logged."""
    with pytest.raises(ValueError, match=rf"'metrics/u' is a {re.escape(kind)}"):
        NonBlockingLogger(print).log({"metrics": {"u": value}})


@requires_cuda
def test_copies_in_flight_wait_without_syncing_the_host(monkeypatch):
    """With the GPU busy, :meth:`log` returns before its copy lands, without
    a host-device sync; ``flush(wait=False)`` logs nothing yet, and ``flush()``
    waits and logs every call, in order."""
    calls, log_fn = _recorder()
    logger = NonBlockingLogger(log_fn)
    torch.cuda.synchronize()
    waits = _count_waits(monkeypatch)
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        torch.cuda._sleep(_BUSY_CYCLES)
        for step in range(3):
            logger.log({"loss": torch.full((), step + 0.5, device="cuda")}, step)
        logger.flush(wait=False)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert calls == [] and waits == []
    logger.flush()
    assert calls == [({"loss": step + 0.5}, (step,), {}) for step in range(3)]


@requires_cuda
def test_error_exit_logs_only_landed_records_without_waiting(monkeypatch):
    """When the ``with`` block raises, the records whose copies have landed
    are logged, in order, and the ones still in flight are dropped without a
    wait; the error propagates."""
    calls, log_fn = _recorder()
    with pytest.raises(RuntimeError, match="step failed"):
        with NonBlockingLogger(log_fn) as logger:
            logger.log({"loss": torch.ones((), device="cuda")}, 0)
            torch.cuda.synchronize()  # step 0's copy has landed
            waits = _count_waits(monkeypatch)
            torch.cuda._sleep(_BUSY_CYCLES)
            logger.log({"loss": torch.zeros((), device="cuda")}, 1)
            raise RuntimeError("step failed")
    assert calls == [({"loss": 1.0}, (0,), {})]
    assert waits == []
    torch.cuda.synchronize()


def _reduce_rank(rank, port):
    """One rank of :func:`test_reduce_op_logs_all_rank_means`."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        loss = torch.tensor(1.0 + rank)
        ### The ranks give the same keys in different orders.
        values = (
            {"loss": loss, "metrics": {"l2": loss * 10}}
            if rank == 0
            else {"metrics": {"l2": loss * 10}, "loss": loss}
        )
        calls, log_fn = _recorder()
        with NonBlockingLogger(log_fn, reduce_op=dist.ReduceOp.AVG) as logger:
            logger.log(values)
            ### Averaged in float32: the mean, 1 + 2**-8, is not a bfloat16.
            logger.log({"bf16": torch.tensor(1 + rank / 128, dtype=torch.bfloat16)})
        own_calls, own_log_fn = _recorder()
        with NonBlockingLogger(own_log_fn) as logger:
            logger.log(values)
        assert calls[0][0] == {"loss": 1.5, "metrics": {"l2": 15.0}}
        assert list(calls[0][0]) == list(values)  # each rank's own key order
        assert calls[1][0] == {"bf16": 1 + 2**-8}
        assert own_calls[0][0] == {
            "loss": 1.0 + rank,
            "metrics": {"l2": 10.0 + 10 * rank},
        }
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available()
    or not dist.is_gloo_available()
    or "spawn" not in mp.get_all_start_methods(),
    reason="needs gloo and the spawn start method",
)
def test_reduce_op_logs_all_rank_means():
    """With ``reduce_op=AVG`` on two CPU ranks, each rank logs the all-rank
    means, even when the ranks give the keys in different orders, and 16-bit
    values are averaged in float32; without it, each rank logs its own
    values."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_reduce_rank, args=(port,), nprocs=2)
