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

"""MeshDataset ``cache_host``: repeated requests for an index read the reader once."""

import torch

from physicsnemo.datapipes.mesh_dataset import MeshDataset
from physicsnemo.mesh import Mesh


class _CountingReader:
    """Minimal reader: two triangles per sample, counts how often each index is read."""

    def __init__(self, n: int = 3) -> None:
        self.n = n
        self.reads: dict[int, int] = {}

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        self.reads[index] = self.reads.get(index, 0) + 1
        pts = (
            torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
            )
            + index
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        return Mesh(points=pts, cells=cells), {"index": index}

    def close(self) -> None:  # MeshDataset.close() calls reader.close() when present
        pass


def test_cache_host_reads_each_index_once():
    reader = _CountingReader()
    ds = MeshDataset(reader, cache_host=True)
    a0, _ = ds[0]
    a1, _ = ds[0]
    b0, _ = ds[1]
    assert reader.reads == {0: 1, 1: 1}
    assert torch.equal(a0.points, a1.points) and float(b0.points.min()) == 1.0
    ds.close()


def test_cache_host_off_rereads():
    reader = _CountingReader()
    ds = MeshDataset(reader)  # default: no cache
    ds[0]
    ds[0]
    assert reader.reads == {0: 2}
    assert ds.cache_host is False
    ds.close()


def test_cache_host_views_reads_k_views_then_cycles():
    reader = _CountingReader()
    ds = MeshDataset(reader, cache_host=True, cache_host_views=3)
    got = [
        ds[0][0].points[0, 0].item() for _ in range(7)
    ]  # reader returns index-offset points; views identical here
    assert reader.reads == {0: 3}  # three views read, then cycled
    assert len(got) == 7 and all(g == 0.0 for g in got)
    ds.close()


class _SlowCountingReader(_CountingReader):
    """Counting reader whose reads overlap in time, so a race would over-read."""

    def __getitem__(self, index: int):
        import time

        time.sleep(0.02)
        return super().__getitem__(index)


def test_cache_host_concurrent_requests_read_at_most_k_views():
    from concurrent.futures import ThreadPoolExecutor

    reader = _SlowCountingReader()
    ds = MeshDataset(reader, cache_host=True, cache_host_views=2)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: ds._read_host(0), range(16)))
    assert reader.reads == {0: 2}
    assert len({id(data) for data, _ in results}) == 2
    ds.close()


def test_cache_host_concurrent_requests_serve_one_identical_sample():
    from concurrent.futures import ThreadPoolExecutor

    reader = _SlowCountingReader()
    ds = MeshDataset(reader, cache_host=True, num_workers=4)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: ds._read_host(1), range(8)))
    assert reader.reads == {1: 1}
    assert len({id(data) for data, _ in results}) == 1
    ds.close()
