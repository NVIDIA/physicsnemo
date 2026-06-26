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

from typing import Literal

import torch
from jaxtyping import Float, Int

from physicsnemo.core.function_spec import FunctionSpec

from ._torch_impl import segment_softmax as segment_softmax_torch
from ._warp_impl import segment_softmax as segment_softmax_warp


class SegmentSoftmax(FunctionSpec):
    """Stable softmax over CSR/ragged segments.

    ``segment_softmax`` normalizes entries along axis 0 independently for each
    segment described by CSR-style ``offsets``. For a segment ``i``, entries
    ``logits[offsets[i]:offsets[i + 1]]`` are treated as one softmax group.
    The output has the same shape and dtype as ``logits`` and preserves the
    flat ragged layout.

    For an entry ``j`` in segment ``i``, the computation is:

    .. math::

       y_j =
       \\frac{\\exp(x_j - m_i)}
            {\\sum_{k=offsets_i}^{offsets_{i+1}-1} \\exp(x_k - m_i)}

    where ``m_i = max(logits[offsets[i]:offsets[i + 1]])`` is subtracted for
    numerical stability. The reduction is performed independently for every
    trailing channel of ``logits``.

    This is the ragged-neighborhood analogue of attention softmax: each
    query, node, anchor, or supernode can have a different number of edges
    without padding to a fixed neighborhood size. Trailing dimensions are
    normalized independently, so ``logits`` may be ``(E,)``, ``(E, H)``, or
    ``(E, ...)`` for multi-head or channel-wise attention scores.

    Examples
    --------
    A flat vector with two segments:

    >>> logits = torch.tensor([2.0, 1.0, 0.0, 4.0, 2.0])
    >>> offsets = torch.tensor([0, 3, 5])
    >>> segment_softmax(logits, offsets)
    tensor([0.6652, 0.2447, 0.0900, 0.8808, 0.1192])

    The first three entries are normalized together and the last two entries
    are normalized together. Each non-empty segment sums to one.

    In sparse/local attention, ``offsets`` typically groups flattened edges by
    query or anchor. For anchor-to-point cross-attention:

    >>> weights = segment_softmax(edge_scores, anchor_offsets)
    >>> messages = weights.unsqueeze(-1) * point_values[point_indices]
    >>> anchor_updates = torch.segment_reduce(
    ...     messages,
    ...     "sum",
    ...     offsets=anchor_offsets,
    ...     axis=0,
    ... )

    This pattern computes one attention distribution per anchor without
    padding every anchor neighborhood to the same length and without requiring
    PyG or ``torch_scatter``.

    Parameters
    ----------
    logits : torch.Tensor
        Floating point scores with shape ``(num_entries, ...)``. Softmax is
        applied over ``num_entries`` within each segment for every trailing
        channel independently.
    offsets : torch.Tensor
        CSR segment pointer of shape ``(num_segments + 1,)`` with dtype
        ``int32`` or ``int64``. ``offsets[0]`` must be 0, offsets must be
        monotonically nondecreasing, and ``offsets[-1]`` must equal
        ``logits.shape[0]``. Empty segments are allowed and produce no output
        entries.
    implementation : {"warp", "torch"} or None
        Explicit backend name. ``None`` auto-selects Warp for CUDA tensors
        when available and falls back to the torch baseline otherwise.

    Returns
    -------
    torch.Tensor
        Segment-normalized weights with the same shape and dtype as
        ``logits``. Values in each non-empty segment sum to one along axis 0.

    Notes
    -----
    - The segment axis is fixed to axis 0 in this initial API. Move or reshape
      other axes before calling this functional.
    - This version accepts CSR ``offsets`` only. It does not accept
      per-entry ``segment_ids``.
    - The operation is differentiable with respect to ``logits``. ``offsets``
      are integer topology metadata and do not receive gradients.
    - Empty segments are valid. Since they own no entries, they produce no
      output values and do not affect neighboring segments.
    - The Warp backend computes in float32 internally and casts the result
      back to the input dtype.
    """

    _BENCHMARK_CASES = (
        ("small-s2048-l16-scalar", 2048, 16, ()),
        ("attention-s8192-l32-h8", 8192, 32, (8,)),
        ("wide-s4096-l48-h4-d16", 4096, 48, (4, 16)),
        ("tiny-neighborhoods-s32768-l8-h4", 32768, 8, (4,)),
    )

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        logits: Float[torch.Tensor, "num_entries ..."],
        offsets: Int[torch.Tensor, "num_segments_plus_one"],
    ) -> Float[torch.Tensor, "num_entries ..."]:
        """Warp-accelerated CSR segmented softmax."""
        return segment_softmax_warp(logits, offsets)

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        logits: Float[torch.Tensor, "num_entries ..."],
        offsets: Int[torch.Tensor, "num_segments_plus_one"],
    ) -> Float[torch.Tensor, "num_entries ..."]:
        """Pure-PyTorch segmented softmax baseline."""
        return segment_softmax_torch(logits, offsets)

    @classmethod
    def dispatch(
        cls,
        logits: torch.Tensor,
        offsets: torch.Tensor,
        implementation: Literal["warp", "torch"] | None = None,
    ) -> torch.Tensor:
        impls = cls._get_impls()
        cls._check_impl(implementation, impls)

        if implementation is not None:
            impl = impls[implementation]
            if not impl.available:
                raise ImportError(
                    f"Implementation '{implementation}' is not available "
                    f"for {cls.__name__}"
                )
            return impl.func(logits, offsets)

        warp_impl = impls.get("warp")
        if logits.is_cuda and warp_impl is not None and warp_impl.available:
            return warp_impl.func(logits, offsets)
        return impls["torch"].func(logits, offsets)

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        for label, num_segments, average_length, tail_shape in cls._BENCHMARK_CASES:
            offsets = _make_offsets(num_segments, average_length, device)
            num_entries = int(offsets[-1].item())
            logits = torch.randn(
                (num_entries, *tail_shape),
                device=device,
                dtype=torch.float32,
            )
            yield label, (logits, offsets), {}

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        for label, num_segments, average_length, tail_shape in cls._BENCHMARK_CASES:
            offsets = _make_offsets(num_segments, average_length, device)
            num_entries = int(offsets[-1].item())
            logits = torch.randn(
                (num_entries, *tail_shape),
                device=device,
                dtype=torch.float32,
                requires_grad=True,
            )
            yield label, (logits, offsets), {}

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)

    @classmethod
    def compare_backward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)


def _make_offsets(
    num_segments: int,
    average_length: int,
    device: torch.device,
) -> torch.Tensor:
    # Deterministic ragged lengths with average close to ``average_length``.
    segment_ids = torch.arange(num_segments, device=device, dtype=torch.int64)
    jitter = (segment_ids % 7) - 3
    lengths = torch.clamp(average_length + jitter, min=1)
    offsets = torch.empty(num_segments + 1, device=device, dtype=torch.int64)
    offsets[0] = 0
    torch.cumsum(lengths, dim=0, out=offsets[1:])
    return offsets


segment_softmax = SegmentSoftmax.make_function("segment_softmax")


__all__ = ["SegmentSoftmax", "segment_softmax"]
