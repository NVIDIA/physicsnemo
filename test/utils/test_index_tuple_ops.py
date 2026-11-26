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

"""Tests for bounded integer tuple operations."""

import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from physicsnemo.utils._index_tuple_ops import (
    _SORTING_NETWORKS,
    _sort_by_network,
    sort_index_tuples,
    sort_index_tuples_,
    unique_index_tuples,
    unique_rows,
)

_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA not available"
        ),
    ),
]


def _assert_matches_torch_unique(
    rows: torch.Tensor,
    index_bound: int,
    *,
    return_inverse: bool = True,
    return_counts: bool = True,
) -> None:
    expected = torch.unique(
        rows,
        dim=0,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )
    actual = unique_index_tuples(
        rows,
        index_bound=index_bound,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
            assert torch.equal(actual_tensor, expected_tensor)
    else:
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)


def test_unique_index_tuples_matches_torch_unique_for_edges() -> None:
    rows = torch.tensor(
        [
            [0, 1],
            [0, 2],
            [0, 1],
            [2, 3],
            [1, 3],
            [2, 3],
        ],
        dtype=torch.long,
    )
    _assert_matches_torch_unique(rows, index_bound=4)


def test_unique_index_tuples_matches_torch_unique_for_faces() -> None:
    rows = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 2],
            [2, 3, 4],
            [1, 2, 4],
        ],
        dtype=torch.long,
    )
    _assert_matches_torch_unique(rows, index_bound=5)


def test_unique_index_tuples_handles_empty_rows() -> None:
    rows = torch.empty((0, 2), dtype=torch.long)
    _assert_matches_torch_unique(rows, index_bound=1)
    _assert_matches_torch_unique(rows, index_bound=0)


def test_unique_index_tuples_falls_back_when_packing_would_overflow() -> None:
    rows = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [1, 2, 3, 4]])
    _assert_matches_torch_unique(rows, index_bound=10_000_000)


@pytest.mark.parametrize("n_columns", [1, 2, 3, 4])
@pytest.mark.parametrize("index_bound", [5, 17])
def test_unique_index_tuples_randomized_equivalence(
    n_columns: int,
    index_bound: int,
) -> None:
    generator = torch.Generator().manual_seed(1234 + n_columns + index_bound)
    rows = torch.randint(
        low=0,
        high=index_bound,
        size=(200, n_columns),
        generator=generator,
    )
    _assert_matches_torch_unique(rows, index_bound=index_bound)


@pytest.mark.parametrize(
    ("return_inverse", "return_counts"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_unique_index_tuples_return_modes(
    return_inverse: bool,
    return_counts: bool,
) -> None:
    rows = torch.tensor([[0, 2], [0, 2], [1, 3], [2, 4], [1, 3]])
    _assert_matches_torch_unique(
        rows,
        index_bound=5,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )


def test_unique_index_tuples_handles_noncontiguous_rows() -> None:
    base = torch.tensor(
        [
            [0, 99, 1],
            [0, 88, 1],
            [2, 77, 3],
            [0, 66, 1],
        ],
        dtype=torch.long,
    )
    rows = base[:, ::2]
    assert not rows.is_contiguous()
    _assert_matches_torch_unique(rows, index_bound=4)


def test_unique_index_tuples_handles_single_row_and_all_duplicates() -> None:
    _assert_matches_torch_unique(torch.tensor([[3, 1, 4]]), index_bound=5)
    _assert_matches_torch_unique(torch.tensor([[2, 2], [2, 2], [2, 2]]), index_bound=3)


def test_unique_index_tuples_rejects_floating_rows() -> None:
    with pytest.raises(TypeError, match="integer"):
        unique_index_tuples(torch.ones(3, 2), index_bound=2)


def test_unique_index_tuples_rejects_non_2d_rows() -> None:
    with pytest.raises(ValueError, match="2D"):
        unique_index_tuples(torch.ones(2, 2, 2, dtype=torch.long), index_bound=2)


def test_unique_index_tuples_rejects_invalid_index_bound() -> None:
    with pytest.raises(ValueError, match="index_bound"):
        unique_index_tuples(torch.ones(2, 2, dtype=torch.long), index_bound=0)


@pytest.mark.parametrize("n_columns", [1, 2, 3, 4])
def test_sorting_networks_sort_every_binary_input(n_columns: int) -> None:
    """By the 0-1 principle, a network that sorts every 0/1 input sorts every input."""
    rows = torch.tensor(list(itertools.product([0, 1], repeat=n_columns)))
    actual = _sort_by_network(rows, _SORTING_NETWORKS[n_columns])
    assert torch.equal(actual, torch.sort(rows, dim=-1).values)


@pytest.mark.parametrize("n_columns", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_sorting_networks_match_torch_sort(n_columns: int, dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(4321 + n_columns)
    # A small index range forces repeated entries within rows
    rows = torch.randint(
        low=0, high=4, size=(300, n_columns), generator=generator, dtype=dtype
    )
    actual = _sort_by_network(rows, _SORTING_NETWORKS[n_columns])
    assert actual.dtype == dtype
    assert torch.equal(actual, torch.sort(rows, dim=-1).values)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize("n_rows", [1_000, 300_000])
@pytest.mark.parametrize("n_columns", [2, 3, 4, 5])
def test_sort_index_tuples_matches_torch_sort(
    device: str, n_rows: int, n_columns: int
) -> None:
    """Both dispatch paths (torch.sort and the sorting network) sort correctly."""
    generator = torch.Generator().manual_seed(8765 + n_columns)
    rows = torch.randint(0, 50, size=(n_rows, n_columns), generator=generator).to(
        device
    )
    assert torch.equal(sort_index_tuples(rows), torch.sort(rows, dim=-1).values)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize("n_rows", [1_000, 300_000])
@pytest.mark.parametrize("n_columns", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_sort_index_tuples_inplace_matches_torch_sort(
    device: str, n_rows: int, n_columns: int, dtype: torch.dtype
) -> None:
    """The in-place variant sorts like torch.sort, in place for the network sizes."""
    generator = torch.Generator().manual_seed(2468 + n_columns)
    rows = torch.randint(0, 50, size=(n_rows, 3, n_columns), generator=generator)
    rows = rows.to(device=device, dtype=dtype)
    expected = torch.sort(rows, dim=-1).values

    actual = sort_index_tuples_(rows)

    assert actual.dtype == dtype
    assert torch.equal(actual, expected)
    in_place = n_columns <= 4 and (device == "cpu" or rows.numel() >= 1 << 19)
    assert (actual.data_ptr() == rows.data_ptr()) == in_place


def test_sort_index_tuples_handles_batched_noncontiguous_and_empty_rows() -> None:
    generator = torch.Generator().manual_seed(8765)
    rows = torch.randint(0, 100, size=(7, 5, 6), generator=generator)[..., ::2]
    assert not rows.is_contiguous()
    assert torch.equal(
        _sort_by_network(rows, _SORTING_NETWORKS[3]), torch.sort(rows, dim=-1).values
    )

    empty = torch.empty((0, 3), dtype=torch.long)
    assert sort_index_tuples(empty).shape == (0, 3)


def _bits(t: torch.Tensor) -> torch.Tensor:
    """``t``'s bit patterns, so that equality also tells ``-0.0`` from ``0.0``."""
    if t.is_floating_point():
        return t.view(
            {2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
        )
    return t


def _assert_unique_rows_matches_torch_unique(rows: torch.Tensor) -> None:
    """Values, inverse and counts of every return mode equal torch.unique's bitwise."""
    for return_inverse, return_counts in itertools.product((False, True), repeat=2):
        expected = torch.unique(
            rows,
            dim=0,
            sorted=True,
            return_inverse=return_inverse,
            return_counts=return_counts,
        )
        actual = unique_rows(
            rows, return_inverse=return_inverse, return_counts=return_counts
        )
        if not isinstance(expected, tuple):
            expected, actual = (expected,), (actual,)
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            assert a.dtype == e.dtype and a.device == e.device
            assert torch.equal(_bits(a), _bits(e))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("n_columns", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("n_rows", [1_000, 100_000])
def test_unique_rows_matches_torch_unique_for_int64_rows(
    device: str, n_columns: int, n_rows: int
) -> None:
    generator = torch.Generator().manual_seed(4321 + n_columns)
    rows = torch.randint(-3, 4, size=(n_rows, n_columns), generator=generator)
    _assert_unique_rows_matches_torch_unique(rows.to(device))


@pytest.mark.parametrize("device", _DEVICES)
def test_unique_rows_matches_torch_unique_for_float64_rows_with_signed_zeros(
    device: str,
) -> None:
    """-0.0 and 0.0 are one value, and each unique row is the first of its copies."""
    rows = torch.tensor(
        [[0.0, 1.5], [-0.0, 1.5], [-1.0, 0.0], [-1.0, -0.0], [0.0, 1.5], [2.0, 0.5]],
        dtype=torch.float64,
    )
    _assert_unique_rows_matches_torch_unique(rows.to(device))
    _assert_unique_rows_matches_torch_unique(rows.flip(0).to(device))
    generator = torch.Generator().manual_seed(1)
    rows = torch.randint(-2, 3, size=(5_000, 3), generator=generator) / 4.0
    _assert_unique_rows_matches_torch_unique(rows.double().to(device))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.int64, torch.float64])
def test_unique_rows_matches_torch_unique_for_empty_and_single_rows(
    device: str, dtype: torch.dtype
) -> None:
    _assert_unique_rows_matches_torch_unique(
        torch.empty(0, 3, dtype=dtype, device=device)
    )
    _assert_unique_rows_matches_torch_unique(
        torch.tensor([[3, 1, 4]], dtype=dtype, device=device)
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_unique_rows_matches_torch_unique_for_cells_and_grouped_directions(
    device: str,
) -> None:
    """Oriented cells, three int64 vertex ids and a parity, with coincident copies;
    and float64 rows of a group id and a direction, with repeats, as in ISLA's
    ``_geometry._grouped_min_norm``."""
    generator = torch.Generator().manual_seed(2)
    n = 300_000
    cells = torch.randint(0, n // 3, size=(n, 4), generator=generator)
    cells[:, 3] %= 2
    _assert_unique_rows_matches_torch_unique(cells.to(device))
    group = torch.randint(0, 2_000, size=(20_000,), generator=generator)
    directions = torch.randn(500, 3, generator=generator, dtype=torch.float64)
    pick = torch.randint(0, 500, size=(20_000,), generator=generator)
    rows = torch.cat([group[:, None].double(), directions[pick]], 1)
    _assert_unique_rows_matches_torch_unique(rows.to(device))


def test_unique_rows_calls_no_torch_unique_over_rows(monkeypatch) -> None:
    """``unique_rows`` never asks ``torch.unique`` for a row sort (its merge sort is
    what the helper avoids), in any return mode."""
    unique = torch.unique

    def unique_of_values(*args, **kwargs):
        assert len(args) < 5 and kwargs.get("dim") is None, "torch.unique over rows"
        return unique(*args, **kwargs)

    rows = torch.tensor([[2, 1], [0, 3], [2, 1], [0, 0]])
    expected = torch.unique(rows, dim=0, return_inverse=True, return_counts=True)
    monkeypatch.setattr(torch, "unique", unique_of_values)
    actual = unique_rows(rows, return_inverse=True, return_counts=True)
    assert all(torch.equal(a, e) for a, e in zip(actual, expected, strict=True))


#: The stress test's calls of 0.2M to 1.8M rows of four int64 columns, how many
#: rounds of large kernels the default stream gets per call, and how often a call's
#: result is also compared with ``torch.unique`` on CPU.
_STRESS_CALLS, _STRESS_LOAD, _STRESS_SAMPLE = 20_000, 4, 1_000

#: The stress test's wall-time bound in seconds, and the calls it must reach within
#: it (``torch.unique(dim=0)`` faults there within a few thousand, measured on GB200).
_STRESS_SECONDS, _STRESS_MIN_CALLS = 600, 2_000


def _stress(find: str) -> dict:
    """Up to ``_STRESS_CALLS`` calls (fewer if ``_STRESS_SECONDS`` run out) of
    ``unique_rows`` (``find == "unique_rows"``) or ``torch.unique(dim=0)``
    (``"torch"``) on a side stream at priority -1, while the default stream runs
    large elementwise and GEMM kernels; the host waits for the side stream alone,
    and the side stream waits for the default stream at every 50th call only. Every result is checked
    on the device (the inverse in range, ``uniq[inverse] == rows``, the unique rows
    strictly increasing) and every ``_STRESS_SAMPLE``-th against ``torch.unique``
    on CPU. Returns the number of calls and the counts of failed checks."""
    device = torch.device("cuda", torch.cuda.current_device())
    side = torch.cuda.Stream(device, priority=-1)
    generator = torch.Generator(device=device).manual_seed(1234)
    sizes = torch.randint(200_000, 1_800_000, (_STRESS_CALLS,)).tolist()
    a = torch.randn(400_000, 256, device=device)
    b = torch.rand(400_000, 256, device=device) + 0.5
    m = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    bad = torch.zeros((), dtype=torch.int64, device=device)
    wrong = calls = 0
    start = time.perf_counter()
    for k, n in enumerate(sizes):
        if time.perf_counter() - start > _STRESS_SECONDS:
            break
        for _ in range(_STRESS_LOAD):
            c = a / b
            m @ m
            a.add_(c, alpha=1e-6)
        ready = torch.cuda.current_stream(device).record_event()
        with torch.cuda.stream(side):
            if k % 50 == 0:
                side.wait_event(ready)
            rows = torch.randint(0, n // 3, (n, 4), device=device, generator=generator)
            rows[:, 3] %= 2
            if find == "unique_rows":
                uniq, inverse = unique_rows(rows, return_inverse=True)
            else:
                uniq, inverse = torch.unique(rows, dim=0, return_inverse=True)
            in_range = (inverse >= 0).all() & (inverse < uniq.shape[0]).all()
            equal = (uniq[inverse.clamp(0, uniq.shape[0] - 1)] == rows).all()
            diff = uniq[1:] - uniq[:-1]
            lead = diff.gather(1, (diff != 0).to(torch.int8).argmax(1)[:, None])
            bad += (~(in_range & equal & (lead > 0).all())).to(torch.int64)
            if k % _STRESS_SAMPLE == 0:
                expected = torch.unique(rows.cpu(), dim=0, return_inverse=True)
                wrong += not all(
                    torch.equal(x.cpu(), e)
                    for x, e in zip((uniq, inverse), expected, strict=True)
                )
            side.synchronize()
        calls += 1
    torch.cuda.synchronize()
    return {"calls": calls, "bad": int(bad), "wrong": wrong}


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unique_rows_on_a_side_stream_beside_large_kernels(
    request: pytest.FixtureRequest,
) -> None:
    """``unique_rows`` on a side stream, beside the default stream's large kernels,
    gives every result right and ends cleanly (``_stress``, in a fresh process: a
    device-side fault kills its CUDA context). ``torch.unique(dim=0)`` there, with
    torch 2.13 on CUDA 13 (CCCL 3.0.1), faults within a few thousand calls. It runs
    only when the run's ``-m`` expression selects ``slow``."""
    if "slow" not in (request.config.getoption("-m") or "").replace("not slow", ""):
        pytest.skip("slow stress test: pass -m slow to run it")
    root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
    done = subprocess.run(  # noqa: S603 - this file, with a fixed argument
        [sys.executable, __file__, "unique_rows"],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=2 * _STRESS_SECONDS,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.splitlines()[-1])
    assert result["calls"] >= _STRESS_MIN_CALLS, result
    assert result["bad"] == 0 and result["wrong"] == 0, result


if __name__ == "__main__":
    print(json.dumps(_stress(sys.argv[1])))
