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

import torch
from jaxtyping import Float, Int

from physicsnemo.core.function_spec import FunctionSpec

from ._torch_impl import farthest_point_sampling as fps_torch
from ._warp_impl import farthest_point_sampling as fps_warp


class FarthestPointSampling(FunctionSpec):
    """Greedy farthest-point sampling (FPS) of a point cloud.

    Iteratively selects the point whose minimum distance to the already
    selected set is largest, producing a well-spread subset of ``num_samples``
    points. Provides a pure-PyTorch baseline and a Warp-accelerated backend;
    auto-dispatch prefers Warp when available and falls back to torch.

    Accepts both unbatched ``(N, D)`` and batched ``(B, N, D)`` inputs; the
    batch axis is stripped from the output for unbatched inputs. ``D`` is the
    coordinate (or feature) dimension and may be any value >= 1.

    The selection is a deterministic greedy traversal from a fixed start
    (index 0) unless ``random_start=True``. For inputs without distance ties
    the two backends select the same set of indices; the Warp backend casts
    to ``float32`` internally, so for ``float64`` inputs near-tie selections
    may differ.

    The Warp backend is **CUDA-only** — it fuses the whole selection into one
    kernel using tile block-reductions, which only cooperate across lanes on
    the CUDA backend. Auto-dispatch uses Warp on CUDA tensors and the torch
    baseline on CPU; passing ``implementation="warp"`` with CPU tensors raises.
    The fused kernel uses one thread block per cloud, so throughput is highest
    for many clouds or moderate cloud sizes; a single very large cloud
    (``B = 1``, large ``N``) is occupancy-bound and sees a smaller speedup.

    Parameters
    ----------
    points : torch.Tensor
        Point cloud of shape ``(N, D)`` or ``(B, N, D)``.
    num_samples : int
        Number of points to select; ``1 <= num_samples <= N``.
    random_start : bool, optional
        Start from a random point per cloud instead of index 0. Default
        ``False``. Note: this draws randomness internally and is
        non-deterministic; it is not recommended under ``torch.compile`` (the
        op is not traced as a random operation).
    implementation : str, optional
        Explicit backend name (``"warp"`` or ``"torch"``). ``None`` auto-selects.
        Default ``None``.

    Returns
    -------
    torch.Tensor
        Selected indices of shape ``(num_samples,)`` (unbatched) or
        ``(B, num_samples)`` (batched), dtype ``int64``.

    Raises
    ------
    ValueError
        If ``points`` is not rank 2 or 3, or if ``num_samples`` is outside
        ``[1, N]``.
    """

    _BENCHMARK_CASES = (
        ("small-p1024-d3-k128", 1, 1024, 3, 128),
        ("medium-p4096-d3-k512", 1, 4096, 3, 512),
        ("large-p16384-d3-k1024", 1, 16384, 3, 1024),
        ("batched-b4-p4096-d3-k512", 4, 4096, 3, 512),
    )

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        points: Float[torch.Tensor, "*batch num_points dim"],
        num_samples: int,
        random_start: bool = False,
    ) -> Int[torch.Tensor, "*batch num_samples"]:
        """Warp-accelerated farthest-point sampling."""
        return fps_warp(points, num_samples, random_start)

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: Float[torch.Tensor, "*batch num_points dim"],
        num_samples: int,
        random_start: bool = False,
    ) -> Int[torch.Tensor, "*batch num_samples"]:
        """Pure-PyTorch farthest-point sampling."""
        return fps_torch(points, num_samples, random_start)

    @classmethod
    def dispatch(
        cls,
        points: torch.Tensor,
        num_samples: int,
        random_start: bool = False,
        implementation: str | None = None,
    ) -> torch.Tensor:
        """Select a backend: Warp on CUDA, torch baseline on CPU.

        The Warp backend is CUDA-only (tile block-reductions), so auto-select
        falls back to torch for CPU tensors. An explicit ``implementation``
        is honored as-is (and the Warp backend will raise on CPU input).
        """
        impls = cls._get_impls()
        cls._check_impl(implementation, impls)
        if implementation is not None:
            impl = impls[implementation]
            if not impl.available:
                raise ImportError(
                    f"Implementation '{implementation}' is not available "
                    f"for {cls.__name__}"
                )
            return impl.func(points, num_samples, random_start)

        warp_impl = impls.get("warp")
        if points.is_cuda and warp_impl is not None and warp_impl.available:
            return warp_impl.func(points, num_samples, random_start)
        return impls["torch"].func(points, num_samples, random_start)

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield ``(label, args, kwargs)`` tuples for forward-pass benchmarking."""
        device = torch.device(device)
        for (
            label,
            batch_size,
            num_points,
            point_dim,
            num_samples,
        ) in cls._BENCHMARK_CASES:
            if batch_size == 1:
                points = torch.rand(num_points, point_dim, device=device)
            else:
                points = torch.rand(batch_size, num_points, point_dim, device=device)
            yield (label, (points, num_samples), {})

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Order-invariant comparison of two selected-index sets.

        Sorted on purpose: the benchmark harness calls this on *unseeded*
        random float32 inputs, and the two backends accumulate squared
        distances in a different order (torch's vectorized reduction vs. the
        Warp kernel's per-axis loop). On a genuine near-tie that can flip a
        single ``argmax`` and reorder an otherwise-identical selection, which a
        direct comparison would report as a spurious mismatch. Comparing sorted
        sets stays robust to that while still catching a real divergence
        (different *sets*).

        Trade-off: this would not flag two backends that select the same set in
        a different order — but for greedy FPS a divergent pick changes the
        selected set from that step onward, so "same set, different order" does
        not occur in practice. Exact ordering and values are covered instead by
        the known-answer and greedy-optimality tests.
        """
        torch.testing.assert_close(
            output.sort(dim=-1).values, reference.sort(dim=-1).values
        )


farthest_point_sampling = FarthestPointSampling.make_function("farthest_point_sampling")
