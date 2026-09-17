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

"""Tests for the lazy preload path and the iterable (generator) dataset path.

Stage 1 covers the lazy, FIFO-handle preload path (IOPump laziness,
``BATCH_BOUNDARY`` reassembly, opaque work items, the ``submit``/``consume``
primitive). Stage 2 covers iterable datasets driven main-thread-only
(finite/capped-infinite generators, ``drop_last``, self-batching
pass-through, reproducibility, and no worker pool). CUDA-guarded tests
exercise stream-bound preprocessing.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import physicsnemo.datapipes as dp
from physicsnemo.datapipes.io_pump import BATCH_BOUNDARY, IOPump
from physicsnemo.datapipes.protocols import DatasetBase, IterableDatasetBase

# ============================================================================
# Stage 1 -- IOPump (lazy, FIFO, batch boundaries)
# ============================================================================


class TestIOPump:
    """Tests for the lazy, self-driving prefetch pump."""

    def test_lazy_bounded_pull_on_infinite_source(self):
        """Pump pulls an unbounded source lazily, bounded by depth."""
        pulled: list[int] = []

        def source():
            i = 0
            while True:
                pulled.append(i)
                yield i
                i += 1

        depth = 3
        pump = IOPump(source(), lambda x: x, depth=depth)
        out = []
        for item in pump:
            out.append(item)
            if len(out) == 5:
                break
        pump.stop()

        assert out == [0, 1, 2, 3, 4]
        # The dispatcher must not have run far ahead of what was consumed:
        # at most consumed + depth + a small slack for the in-flight pull.
        assert len(pulled) <= 5 + depth + 2

    def test_batch_boundary_reassembly_irregular(self):
        """Boundaries delimit dynamically-sized batches without slot use."""
        source = [0, 1, BATCH_BOUNDARY, 2, BATCH_BOUNDARY, 3, 4, 5, BATCH_BOUNDARY]
        pump = IOPump(iter(source), lambda x: x, depth=2)

        batches: list[list[int]] = []
        current: list[int] = []
        for item in pump:
            if item is BATCH_BOUNDARY:
                batches.append(current)
                current = []
            else:
                current.append(item)
        pump.stop()

        assert batches == [[0, 1], [2], [3, 4, 5]]

    def test_fifo_order_preserved(self):
        """Handles are yielded in the order work items were pulled."""
        pump = IOPump(iter(range(20)), lambda x: x * 10, depth=4)
        out = list(pump)
        pump.stop()
        assert out == [x * 10 for x in range(20)]

    def test_dispatch_error_surfaces_not_hangs(self):
        """A dispatch exception is raised on the consumer, never a hang."""

        def boom(x):
            raise RuntimeError("dispatch failed")

        pump = IOPump(iter(range(5)), boom, depth=2)
        with pytest.raises(RuntimeError, match="dispatch failed"):
            list(pump)
        pump.stop()

    def test_source_error_surfaces_not_hangs(self):
        """A failing source is raised on the consumer, never a hang."""

        def source():
            yield 0
            raise ValueError("source failed")

        pump = IOPump(source(), lambda x: x, depth=2)
        with pytest.raises(ValueError, match="source failed"):
            list(pump)
        pump.stop()


# ============================================================================
# Stage 1 -- submit / consume FIFO primitive with opaque work items
# ============================================================================


class _DescriptorDataset(DatasetBase):
    """Map-style dataset keyed by an opaque (non-int) descriptor."""

    def __init__(self):
        super().__init__(num_workers=2)
        self._store = {"alpha": 1.0, "beta": 2.0, "gamma": 3.0}

    def _load(self, key):
        if key == "explode":
            raise KeyError("no such key")
        return TensorDict({"x": torch.tensor([self._store[key]])}), {"key": key}

    def __len__(self):
        return len(self._store)


class TestSubmitConsume:
    """Tests for the FIFO submit/consume primitive."""

    def test_opaque_descriptor_roundtrip(self):
        """submit/consume works with non-int, string work items."""
        ds = _DescriptorDataset()
        try:
            handle = ds.submit("beta")
            data, metadata = ds.consume(handle)
            assert metadata["key"] == "beta"
            assert data["x"].item() == 2.0
        finally:
            ds.close()

    def test_submit_consume_fifo_independent_of_value(self):
        """Multiple in-flight handles consume to their own results."""
        ds = _DescriptorDataset()
        try:
            handles = [ds.submit(k) for k in ("alpha", "beta", "gamma")]
            keys = [ds.consume(h)[1]["key"] for h in handles]
            assert keys == ["alpha", "beta", "gamma"]
        finally:
            ds.close()

    def test_producer_error_reraised_on_consume(self):
        """An error raised in the producer surfaces on consume."""
        ds = _DescriptorDataset()
        try:
            handle = ds.submit("explode")
            with pytest.raises(KeyError):
                ds.consume(handle)
        finally:
            ds.close()


# ============================================================================
# Stage 1 -- DataLoader laziness over the sampler
# ============================================================================


class _CountingSampler:
    """Sequential sampler that records how many indices it has yielded."""

    def __init__(self, n):
        self.n = n
        self.consumed = 0

    def __iter__(self):
        self.consumed = 0
        for i in range(self.n):
            self.consumed += 1
            yield i

    def __len__(self):
        return self.n


class TestDataLoaderLazyPreload:
    """The preload path must not materialize the whole epoch up front."""

    def test_sampler_not_fully_drained_on_early_break(self, numpy_data_dir):
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        sampler = _CountingSampler(10)
        # Keep the prefetch window small (single stream, one batch ahead) so
        # the bound is well under the epoch; with the default num_streams the
        # in-flight depth would cover this tiny 10-sample dataset entirely.
        loader = dp.DataLoader(
            dataset,
            batch_size=2,
            sampler=sampler,
            prefetch_factor=1,
            num_streams=1,
        )

        first = next(iter(loader))
        assert first["positions"].shape[0] == 2
        # Only a bounded prefix of the sampler should have been consumed,
        # never the full epoch, after pulling a single batch.
        assert sampler.consumed < 10

    def test_preload_matches_sequential_order(self, numpy_data_dir):
        reader = dp.NumpyReader(numpy_data_dir)
        dataset = dp.Dataset(reader)
        loader = dp.DataLoader(
            dataset, batch_size=3, shuffle=False, collate_metadata=True
        )
        indices = []
        for _batch, metadata_list in loader:
            indices.extend(m["index"] for m in metadata_list)
        assert indices == list(range(10))


# ============================================================================
# Stage 2 -- iterable datasets
# ============================================================================


class _RangeIterable(IterableDatasetBase):
    """Finite per-sample generator yielding (TensorDict, metadata)."""

    def __init__(self, n, dim=4):
        self.n = n
        self.dim = dim

    def __iter__(self):
        for i in range(self.n):
            data = TensorDict({"x": torch.full((self.dim,), float(i))})
            yield data, {"index": i}


class _BatchIterable(IterableDatasetBase):
    """Self-batching generator yielding ready-made batches."""

    yields_batches = True

    def __init__(self, n_batches, batch=4, dim=3):
        self.n_batches = n_batches
        self.batch = batch
        self.dim = dim

    def __iter__(self):
        for b in range(self.n_batches):
            yield TensorDict(
                {"x": torch.full((self.batch, self.dim), float(b))},
                batch_size=[self.batch],
            )


class _SeededIterable(IterableDatasetBase):
    """Per-(epoch, position) seeded generator for reproducibility tests."""

    def __init__(self, n, base_seed=0):
        self.n = n
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        for position in range(self.n):
            seed = int(
                np.random.SeedSequence(
                    [self.base_seed, self.epoch, position]
                ).generate_state(1)[0]
            )
            g = torch.Generator().manual_seed(seed)
            yield TensorDict({"x": torch.rand(3, generator=g)}), {"position": position}


class _ThreadRecordingIterable(IterableDatasetBase):
    """Records which thread the generator runs on."""

    def __init__(self, n):
        self.n = n
        self.threads = []

    def __iter__(self):
        for i in range(self.n):
            self.threads.append(threading.current_thread())
            yield TensorDict({"x": torch.zeros(2)}), {"index": i}


class TestIterableDataLoader:
    """Tests for the main-thread-only iterable (generator) path."""

    def test_per_sample_batching(self):
        loader = dp.DataLoader(_RangeIterable(10), batch_size=4)
        batches = list(loader)
        # 10 samples / 4 -> [4, 4, 2]
        assert [b["x"].shape[0] for b in batches] == [4, 4, 2]

    def test_per_sample_drop_last(self):
        loader = dp.DataLoader(_RangeIterable(10), batch_size=4, drop_last=True)
        batches = list(loader)
        # Trailing partial batch dropped -> [4, 4]
        assert [b["x"].shape[0] for b in batches] == [4, 4]

    def test_self_batching_passthrough(self):
        # The loader batch_size is intentionally different from the generator's
        # to prove it is ignored for self-batching datasets.
        loader = dp.DataLoader(_BatchIterable(3, batch=5), batch_size=2)
        batches = list(loader)
        assert len(batches) == 3
        assert all(b["x"].shape[0] == 5 for b in batches)

    def test_len_raises_for_iterable(self):
        loader = dp.DataLoader(_RangeIterable(10), batch_size=2)
        with pytest.raises(TypeError):
            len(loader)

    def test_capped_infinite_consumes_without_length(self):
        """A long generator is iterated batch-by-batch; len() is never used."""

        class _BigIterable(IterableDatasetBase):
            def __iter__(self):
                i = 0
                while i < 10_000:
                    yield TensorDict({"x": torch.zeros(2)}), {"index": i}
                    i += 1

        loader = dp.DataLoader(_BigIterable(), batch_size=4)
        seen = 0
        for _batch in loader:
            seen += 1
            if seen == 3:
                break
        assert seen == 3

    def test_shuffle_warns_for_iterable(self):
        with pytest.warns(UserWarning, match="ignored for iterable"):
            dp.DataLoader(_RangeIterable(4), batch_size=2, shuffle=True)

    def test_reproducible_across_runs_distinct_across_epochs(self):
        loader = dp.DataLoader(_SeededIterable(6), batch_size=3)

        loader.set_epoch(0)
        run_a = torch.cat([b["x"].reshape(-1) for b in loader])
        loader.set_epoch(0)
        run_b = torch.cat([b["x"].reshape(-1) for b in loader])
        loader.set_epoch(1)
        run_c = torch.cat([b["x"].reshape(-1) for b in loader])

        assert torch.equal(run_a, run_b)  # same epoch -> identical
        assert not torch.equal(run_a, run_c)  # different epoch -> distinct

    def test_runs_on_main_thread_no_worker_pool(self):
        dataset = _ThreadRecordingIterable(4)

        names_before = {t.name for t in threading.enumerate()}
        loader = dp.DataLoader(dataset, batch_size=2)
        _ = list(loader)
        names_after = {t.name for t in threading.enumerate()}

        # Generation happened on the main thread only.
        assert dataset.threads, "generator did not run"
        assert all(t is threading.main_thread() for t in dataset.threads)
        # No prefetch worker pool / pump thread was spawned for this path.
        new_threads = names_after - names_before
        assert not any(
            n.startswith("datapipe_prefetch") or n == "datapipe_pump"
            for n in new_threads
        )


# ============================================================================
# CUDA-guarded -- stream-bound consume
# ============================================================================


class TestStreamBoundConsume:
    """Preprocessing on an assigned stream (the default-stream workaround
    is gone, so transforms run on the side stream)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_submit_consume_on_side_stream(self, numpy_data_dir):
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        dataset = dp.Dataset(
            reader,
            device="cuda:0",
            transforms=dp.SubsamplePoints(
                input_keys=["positions", "features"], n_points=50
            ),
        )
        try:
            stream = torch.cuda.Stream()
            handle = dataset.submit(0, stream=stream)
            data, _metadata = dataset.consume(handle)
            torch.cuda.synchronize()
            assert data["positions"].device.type == "cuda"
            assert data["positions"].shape[0] == 50
        finally:
            dataset.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dataloader_streams_match_synchronous(self, numpy_data_dir):
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)

        ref = dp.Dataset(reader, device="cuda:0")
        ref_loader = dp.DataLoader(ref, batch_size=2, shuffle=False, prefetch_factor=0)
        expected = [b["positions"].sum().item() for b in ref_loader]

        reader2 = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        streamed = dp.Dataset(reader2, device="cuda:0")
        loader = dp.DataLoader(
            streamed,
            batch_size=2,
            shuffle=False,
            prefetch_factor=2,
            num_streams=4,
            use_streams=True,
        )
        got = [b["positions"].sum().item() for b in loader]
        torch.cuda.synchronize()
        assert got == pytest.approx(expected, rel=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_defers_wait_until_after_previous_batch(
        self, numpy_data_dir, monkeypatch
    ):
        """The compute-stream wait for batch N+1 must be enqueued *after*
        batch N is yielded, not during the lookahead consume of N+1.

        This is the overlap-correctness invariant: if the wait for the next
        batch's preprocessing landed on the compute stream before the current
        batch's model work, the model would block on the next batch's
        preprocessing (no overlap). With the deferred wait, the compute-stream
        order is ``..., model_{N-1}, wait(prep_N), model_N, ...`` so a batch's
        preprocessing overlaps the previous batch's compute.

        We spy on ``Stream.wait_event`` and record the interleaving of waits
        and yields: the wait for batch 0 must precede yielding batch 0, while
        the wait for batch 1 must come *after* batch 0 is yielded.
        """
        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        dataset = dp.Dataset(reader, device="cuda:0")
        loader = dp.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            prefetch_factor=2,
            num_streams=4,
            use_streams=True,
        )

        order: list[str] = []
        stream_cls = type(torch.cuda.current_stream())
        real_wait = stream_cls.wait_event

        def spy_wait(self, event):
            order.append("wait")
            return real_wait(self, event)

        monkeypatch.setattr(stream_cls, "wait_event", spy_wait)

        try:
            for i, _batch in enumerate(loader):
                order.append(f"yield{i}")
                if i >= 2:
                    break
        finally:
            torch.cuda.synchronize()

        wait_indices = [k for k, ev in enumerate(order) if ev == "wait"]
        assert len(wait_indices) >= 2, f"expected >=2 waits, got order={order}"
        yield0_idx = order.index("yield0")
        # Batch 0's preprocessing is gated before batch 0 is handed out.
        assert wait_indices[0] < yield0_idx, order
        # Batch 1's preprocessing wait is deferred until after batch 0 is
        # yielded (the regression guard: the old code waited inline during the
        # lookahead consume of batch 1, before batch 0 was ever yielded).
        assert wait_indices[1] > yield0_idx, order


class TestCrossStreamMemoryLifetime:
    """Tensors handed from a preprocessing stream to the compute stream must
    be recorded against the compute stream so the caching allocator does not
    recycle their blocks for later prep-stream samples while compute-stream
    reads are still pending."""

    def test_record_consumer_stream_skips_cpu_tensors(self, monkeypatch):
        from physicsnemo.datapipes.protocols import record_consumer_stream

        calls: list = []
        monkeypatch.setattr(
            torch.Tensor, "record_stream", lambda self, stream: calls.append(self)
        )

        td = TensorDict({"a": torch.ones(3), "b": {"c": torch.ones(2)}})
        # Traversal covers tensors, collections, mappings, and sequences
        # without error; CPU tensors are never recorded.
        record_consumer_stream(torch.ones(4), stream=object())
        record_consumer_stream(td, stream=object())
        record_consumer_stream({"x": torch.ones(2), "meta": "str"}, stream=object())
        record_consumer_stream([torch.ones(1), (td, None)], stream=object())
        assert calls == []

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_record_consumer_stream_records_all_cuda_leaves(self, monkeypatch):
        from physicsnemo.datapipes.protocols import record_consumer_stream

        recorded: list = []
        monkeypatch.setattr(
            torch.Tensor,
            "record_stream",
            lambda self, stream: recorded.append((self, stream)),
        )

        td = TensorDict(
            {
                "a": torch.ones(3, device="cuda"),
                "b": {"c": torch.ones(2, device="cuda")},
                "cpu": torch.ones(2),
            }
        )
        stream = torch.cuda.Stream()
        record_consumer_stream({"td": td, "t": torch.ones(1, device="cuda")}, stream)
        assert len(recorded) == 3
        assert all(s is stream for _, s in recorded)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_stream_overlap_values_correct_without_host_sync(self, temp_dir):
        """Regression test for allocator reuse across streams.

        Batch values are validated on the GPU with *no host sync inside the
        loop*, while ``torch.cuda._sleep`` stands in for a GPU-bound model
        and keeps the compute queue deep, so the host (and the prefetch
        lookahead) runs ahead of GPU execution.  Without
        ``record_consumer_stream`` in ``_consume``, a later sample's
        host-to-device copy on the same round-robin prep stream can reuse a
        freed sample's block and overwrite it before the compute stream's
        pending collate read executes, corrupting the batch.  A per-batch
        ``.item()`` (as in test_dataloader_streams_match_synchronous) would
        drain the queue and mask exactly this race.
        """
        n_samples, width = 32, 4096
        for i in range(n_samples):
            np.savez(
                temp_dir / f"sample_{i:03d}.npz",
                values=np.full((width,), float(i), dtype=np.float32),
            )
        reader = dp.NumpyReader(temp_dir, pin_memory=True)
        dataset = dp.Dataset(reader, device="cuda:0")
        loader = dp.DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            prefetch_factor=2,
            num_streams=2,
            use_streams=True,
        )

        residuals = []
        idx = 0
        try:
            for batch in loader:
                # Keep the compute stream busy so the host runs ahead and
                # freed sample blocks still have pending compute reads.
                torch.cuda._sleep(20_000_000)
                values = batch["values"]
                expected = torch.arange(
                    idx, idx + values.shape[0], device=values.device
                ).to(values.dtype)
                residuals.append((values - expected.unsqueeze(1)).abs().max())
                idx += values.shape[0]
            worst = torch.stack(residuals).max()
            torch.cuda.synchronize()
            assert idx == n_samples
            assert worst.item() == 0.0
        finally:
            dataset.close()
