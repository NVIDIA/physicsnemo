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
exercise stream-bound preprocessing and Warp-on-a-non-default-stream.
"""

from __future__ import annotations

import threading
import time

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


class _StageLockedDataset(DatasetBase):
    """Dataset that records whether worker load overlaps consume."""

    def __init__(self):
        super().__init__(num_workers=1, serialize_load_consume=True)
        self.load_entries: list[int] = []
        self.consume_started = threading.Event()
        self.release_consume = threading.Event()

    def _load(self, index):
        return TensorDict({"x": torch.tensor([float(index)])}), {"index": index}

    def _load_host(self, work_item):
        self.load_entries.append(work_item)
        return super()._load_host(work_item)

    def _consume(self, payload, stream=None):
        self.consume_started.set()
        self.release_consume.wait(timeout=5.0)
        return super()._consume(payload, stream)

    def __len__(self):
        return 2


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

    def test_stage_lock_prevents_load_consume_overlap(self):
        """Opt-in stage lock keeps worker loads out of active consume."""
        ds = _StageLockedDataset()
        try:
            first = ds.submit(0)
            first.future.result(timeout=5.0)

            consumer = threading.Thread(target=ds.consume, args=(first,))
            consumer.start()
            assert ds.consume_started.wait(timeout=5.0)

            second = ds.submit(1)
            time.sleep(0.1)
            assert ds.load_entries == [0]

            ds.release_consume.set()
            consumer.join(timeout=5.0)
            second.future.result(timeout=5.0)
            assert ds.load_entries == [0, 1]
        finally:
            ds.release_consume.set()
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
        loader = dp.DataLoader(
            dataset, batch_size=2, sampler=sampler, prefetch_factor=1
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
# CUDA-guarded -- stream-bound consume and Warp-on-non-default-stream
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


class TestWarpIterableOnStream:
    """Warp launches on a non-default stream from the main thread are safe."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_darcy_online_simulation_through_iterable_loader(self):
        from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

        class _DarcyIterable(IterableDatasetBase):
            yields_batches = True

            def __init__(self, num_batches):
                self._sim = Darcy2D(resolution=32, batch_size=2, device="cuda")
                self._num_batches = num_batches

            def __iter__(self):
                sim_iter = iter(self._sim)
                for _ in range(self._num_batches):
                    yield next(sim_iter)

        loader = dp.DataLoader(_DarcyIterable(2), use_streams=True)
        batches = list(loader)
        torch.cuda.synchronize()  # surfaces any illegal-memory-access
        assert len(batches) == 2
        for batch in batches:
            assert batch["permeability"].device.type == "cuda"
            assert batch["darcy"].device.type == "cuda"


class TestWarpFunctionalTransformOnStreams:
    """A Warp ``FunctionSpec`` transform driven through the multi-stream
    preload path. The functional binds the current torch stream as a Warp
    stream internally; the loader binds the same stream around the consume.
    Both must reuse one cached wrapper -- otherwise the inner wrapper
    unregisters the shared stream on teardown and the next launch faults
    (illegal memory access)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_functional_warp_transform_multi_stream(self, numpy_data_dir):
        from physicsnemo.datapipes.transforms.base import Transform
        from physicsnemo.nn.functional import signed_distance_field

        class _SDFTransform(Transform):
            """Evaluate an SDF (a Warp functional) against the sample points."""

            def __call__(self, data):
                points = data["positions"].reshape(-1, 3).float()
                vertices = torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    device=points.device,
                )
                faces = torch.tensor([[0, 1, 2]], device=points.device)
                sdf, _ = signed_distance_field(vertices, faces, points)
                data["sdf"] = sdf.reshape(-1, 1)
                return data

        reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
        dataset = dp.Dataset(reader, device="cuda:0", transforms=_SDFTransform())
        loader = dp.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            prefetch_factor=2,
            num_streams=4,
            use_streams=True,
        )
        # Iterate well past num_streams so every stream is reused at least
        # once; a churned registration faults on the second pass.
        batches = list(loader)
        torch.cuda.synchronize()  # surfaces any illegal-memory-access
        assert len(batches) == 10
        for batch in batches:
            assert batch["sdf"].device.type == "cuda"
