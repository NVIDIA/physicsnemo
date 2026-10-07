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

r"""Private helpers shared by the conformal package.

Field-container handling (plain tensor or ``TensorDict``), exact conformal
rank arithmetic, and input validators that calibration and prediction share,
so both apply the same checks.

A calibrated threshold must never round below its order statistic, or the
finite-sample guarantee fails. Quantiles therefore stay in the scores'
promoted dtype, and :func:`cast_directed` rounds outward whenever a cast
loses precision.
"""

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from numbers import Real
from typing import TYPE_CHECKING, Literal, get_args

import torch
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor

if TYPE_CHECKING:
    from .scores import _NonconformityScore

TENSOR_KEY = "__tensor__"
"""Reserved field key used when the inputs are plain tensors."""


def field_items(
    x: Tensor | TensorDict,
    keys: Sequence[str] | None = None,
) -> list[tuple[str, Tensor]]:
    """Return sorted ``(key, tensor)`` pairs; a plain tensor uses :data:`TENSOR_KEY`."""
    if isinstance(x, Tensor):
        if keys is not None:
            raise TypeError(
                f"keys={list(keys)} selects TensorDict fields, but the input is a "
                "plain tensor. Pass a TensorDict or drop keys."
            )
        return [(TENSOR_KEY, x)]
    if not isinstance(x, TensorDict):
        raise TypeError(
            f"Inputs must be a torch.Tensor or TensorDict, got {type(x).__name__}."
        )
    available = sorted(x.keys())
    if not available:
        raise ValueError("TensorDict inputs must contain at least one field.")
    if TENSOR_KEY in available:
        raise ValueError(
            f"{TENSOR_KEY!r} is a reserved field name; rename this TensorDict field."
        )
    if keys is not None:
        missing = sorted(set(keys) - set(available))
        if missing:
            raise KeyError(
                f"Requested fields {missing} not present; available: {available}."
            )
        available = [k for k in available if k in set(keys)]
    items = [(k, x[k]) for k in available]
    for key, value in items:
        if not isinstance(value, Tensor):
            raise TypeError(
                f"Field {key!r} must contain a torch.Tensor, got "
                f"{type(value).__name__}."
            )
    return items


def require_container_kind(
    x: object, keys: Sequence[str] | None, owner: str, name: str
) -> None:
    """Raise ``TypeError`` if ``x`` is the other container kind than calibration used.

    ``keys`` is ``None`` when calibration used a plain tensor.
    """
    if keys is None and isinstance(x, TensorDict):
        raise TypeError(
            f"{owner} was calibrated on a plain tensor; pass {name} as a tensor, "
            "not a TensorDict."
        )
    if keys is not None and isinstance(x, Tensor):
        raise TypeError(
            f"{owner} was calibrated on TensorDict fields {list(keys)}; pass "
            f"{name} as a TensorDict."
        )


def pack_fields(
    items: Mapping[str, Tensor],
) -> Tensor | TensorDict:
    """Inverse of :func:`field_items`."""
    if set(items.keys()) == {TENSOR_KEY}:
        return items[TENSOR_KEY]
    return TensorDict(dict(items), batch_size=[])


def slice_aux(
    aux: Mapping[str, Tensor] | Mapping[str, Mapping[str, Tensor]] | None,
    field: str,
) -> Mapping[str, Tensor] | None:
    """Return one field's aux mapping; ``TensorDict`` inputs nest aux by field."""
    if aux is None:
        return None
    if not isinstance(aux, Mapping):
        raise TypeError(f"aux must be a mapping, got {type(aux).__name__}.")
    if field == TENSOR_KEY:
        return aux  # type: ignore[return-value]
    entry = aux.get(field)
    if entry is None and aux and all(isinstance(v, Tensor) for v in aux.values()):
        raise TypeError(
            "aux for TensorDict inputs must be nested by field name, got "
            f"top-level tensor entries {sorted(aux)}; pass "
            f"aux={{'{field}': {{key: tensor}}, ...}}."
        )
    if entry is not None and not isinstance(entry, Mapping):
        raise TypeError(
            f"Field '{field}': aux entry must be a mapping, got {type(entry).__name__}."
        )
    return entry  # type: ignore[return-value]


def validate_alpha(alpha: object) -> float:
    """Return ``alpha`` as a float in ``(0, 1)``; reject inexact non-floats."""
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError(f"alpha must be a real number, got {alpha!r}.")
    as_float = float(alpha)
    if not isinstance(alpha, float) and alpha != as_float:
        raise TypeError(
            f"alpha={alpha!r} is not exactly representable as a float; pass a "
            "float such as 0.1."
        )
    if not math.isfinite(as_float) or not 0.0 < as_float < 1.0:
        raise ValueError(f"alpha must be a finite value in (0, 1), got {as_float}.")
    return as_float


def validate_n_cal(n_cal: object) -> int:
    """Return ``n_cal`` as a positive ``int``; reject ``bool``."""
    if isinstance(n_cal, bool) or not isinstance(n_cal, int):
        raise TypeError(f"n_cal must be an integer, got {n_cal!r}.")
    if n_cal < 1:
        raise ValueError(f"n_cal must be >= 1, got {n_cal}.")
    return n_cal


def alpha_as_fraction(alpha: float) -> Fraction:
    """Return ``alpha`` as an exact fraction of its shortest decimal form.

    Exact arithmetic keeps ``150 * 0.82 == 123`` and still rounds
    ``alpha=0.49999999995`` up; no single float tolerance does both.
    """
    return Fraction(str(validate_alpha(alpha)))


def require_feasible_alpha(n_cal: int, alpha: float) -> Fraction:
    """Require ``alpha >= 1 / (n_cal + 1)``; return ``alpha`` as an exact fraction.

    Below that, the quantile rank exceeds ``n_cal``.
    """
    alpha_exact = alpha_as_fraction(alpha)
    n_cal = validate_n_cal(n_cal)
    if alpha_exact < Fraction(1, n_cal + 1):
        min_n = math.ceil((1 - alpha_exact) / alpha_exact)
        raise ValueError(
            f"alpha={alpha} needs at least {min_n} calibration samples (got "
            f"{n_cal}); collect more or raise alpha."
        )
    return alpha_exact


def conformal_quantile_index(n_cal: int, alpha: float) -> int:
    """Return ``k = ceil((n_cal + 1)(1 - alpha))`` exactly; ``1 <= k <= n_cal``."""
    return math.ceil((n_cal + 1) * (1 - require_feasible_alpha(n_cal, alpha)))


def kth_smallest_of_samples(
    per_sample: Sequence[Float[Tensor, "*dims"]],
    k: int,
    *,
    chunk_numel: int = 2**26,
) -> Float[Tensor, "*dims"]:
    """Return the elementwise ``k``-th smallest value across samples.

    ``kthvalue`` is exact and, unlike ``torch.quantile``, has no
    ``2**24``-element limit. The result uses the samples' promoted dtype.
    """
    n = len(per_sample)
    first = per_sample[0]
    cell_shape = first.shape
    dtype = first.dtype
    for t in per_sample[1:]:
        dtype = torch.promote_types(dtype, t.dtype)
    flats = [t.reshape(-1) for t in per_sample]
    n_cells = flats[0].numel()
    cells_per_chunk = max(1, chunk_numel // n)
    out = torch.empty(n_cells, dtype=dtype, device=first.device)
    block = torch.empty(
        (n, min(cells_per_chunk, n_cells)), dtype=dtype, device=first.device
    )
    for start in range(0, n_cells, cells_per_chunk):
        stop = min(start + cells_per_chunk, n_cells)
        view = block[:, : stop - start]
        for row, flat in zip(view, flats):
            row.copy_(flat[start:stop])
        out[start:stop] = torch.kthvalue(view, k, dim=0).values
    return out.reshape(cell_shape)


def cast_directed(t: Tensor, dtype: torch.dtype, *, up: bool) -> Tensor:
    """Cast to ``dtype``, rounding toward ``+inf`` (``up=True``) or ``-inf``.

    Outward rounding keeps a threshold from landing below its order statistic.
    """
    cast = t.to(dtype)
    if dtype == t.dtype or not (t.is_floating_point() and cast.is_floating_point()):
        return cast
    exact = t.to(torch.float64)
    got = cast.to(torch.float64)
    wrong_side = got < exact if up else got > exact
    toward = torch.inf if up else -torch.inf
    bumped = torch.nextafter(cast, torch.full_like(cast, toward))
    return torch.where(wrong_side, bumped, cast)


Tier = Literal["cellwise", "functional"]
"""Guarantee tier names."""

TIERS: tuple[str, ...] = get_args(Tier)


def _field_label(key: str) -> str:
    """Name a field in error messages; plain tensors never show :data:`TENSOR_KEY`."""
    return "Plain tensor" if key == TENSOR_KEY else f"Field '{key}'"


def check_points(points: Tensor, pending: list[Tensor] | None = None) -> Tensor:
    """Require finite floating ``points`` of shape ``(n_points, n_spatial_dims)``.

    With ``pending``, append the finiteness flag there instead of syncing.
    """
    if not isinstance(points, Tensor):
        raise TypeError(f"points must be a torch.Tensor, got {type(points).__name__}.")
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError(
            "points must have non-empty shape (n_points, n_spatial_dims), "
            f"got {tuple(points.shape)}."
        )
    if not points.is_floating_point():
        raise TypeError(f"points must have a floating dtype, got {points.dtype}.")
    finite = torch.isfinite(points).all()
    if pending is not None:
        pending.append(finite)
    elif not finite:
        raise ValueError("points contains non-finite coordinate value(s).")
    return points


def fetch_ints(values: Sequence[Tensor]) -> list[int]:
    """Copy 0-d integer or bool tensors to the host in one transfer (one sync)."""
    device = values[0].device
    stacked = torch.stack([v.to(device=device, dtype=torch.int64) for v in values])
    return stacked.tolist()


def _signed64(value: int) -> int:
    """Reinterpret an unsigned 64-bit value as a signed ``int64``."""
    return value - (1 << 64) if value >= 1 << 63 else value


_GOLDEN = _signed64(0x9E3779B97F4A7C15)
_MIX = ((30, _signed64(0xBF58476D1CE4E5B9)), (27, _signed64(0x94D049BB133111EB)))
_INT_VIEW = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _shift_right(x: Tensor, shift: int) -> Tensor:
    """Logical right shift of ``int64`` values."""
    return x.bitwise_right_shift(shift).bitwise_and_((1 << (64 - shift)) - 1)


def _mix_(x: Tensor) -> Tensor:
    """SplitMix64 finalizer in place: a bijection that scrambles every bit."""
    for shift, multiplier in _MIX:
        x.bitwise_xor_(_shift_right(x, shift)).mul_(multiplier)
    return x.bitwise_xor_(_shift_right(x, 31))


def _points_checksum(points: Tensor) -> list[Tensor]:
    """Two ``int64`` checksums of the raw coordinate bits, computed on device.

    Each element's bits are hashed together with its flat index, and the
    hashes are summed twice: once plainly and once weighted by ``2 * i + 1``.
    The hash is a bijection of the bits, so changing one value always changes
    the first sum. Integer sums wrap and do not depend on reduction order, so
    the result is the same on every device and memory layout.
    """
    bits = points.detach().view(_INT_VIEW[points.element_size()])
    bits = bits.to(torch.int64).reshape(-1)
    index = torch.arange(bits.numel(), dtype=torch.int64, device=bits.device)
    hashed = _mix_(index.mul(_GOLDEN).bitwise_xor_(bits))
    return [hashed.sum(), hashed.mul_(index.mul_(2).add_(1)).sum()]


def points_fingerprint(
    points: Tensor,
) -> str:
    """Return a checksum of point coordinates, dtype, and shape; order matters.

    The checksum runs on the points' device and syncs once. Not cached:
    in-place writes through ``.data`` or ``.numpy()`` do not bump the
    version counter.
    """
    pending: list[Tensor] = []
    check_points(points, pending)
    finite, *sums = fetch_ints([*pending, *_points_checksum(points)])
    if not finite:
        check_points(points)
    digest = "".join(f"{value & ((1 << 64) - 1):016x}" for value in sums)
    return f"{_mesh_label(points)}-{digest}"


def _mesh_label(points: Tensor) -> str:
    """The ``<dtype>-<n_points>x<n_dims>`` prefix of a mesh fingerprint."""
    n_points, n_dims = points.shape
    return f"{str(points.dtype).removeprefix('torch.')}-{n_points}x{n_dims}"


def _same_mesh(points: Tensor, expected: str) -> Tensor:
    """Whether ``points`` has fingerprint ``expected``, as a 0-d device bool."""
    label, digest = expected.rsplit("-", 1)
    sums = _points_checksum(points)
    same = torch.full((), label == _mesh_label(points), device=points.device)
    for index, value in enumerate(sums):
        stored = _signed64(int(digest[16 * index : 16 * (index + 1)], 16))
        same &= value == stored
    return same


def require_mesh(
    points: Tensor | None,
    expected: str | None,
    hint: str = "",
    pending: list[Tensor] | None = None,
) -> str:
    """Fingerprint ``points`` and require it to equal ``expected`` unless ``None``.

    With ``pending`` (and ``expected`` set), append the finiteness and match
    flags there instead of syncing; the caller must check them.
    """
    if points is None:
        raise ValueError(
            "Cellwise conformal requires points=, the calibration mesh coordinates."
        )
    if pending is not None:
        check_points(points, pending)
        pending.append(_same_mesh(points, expected))
        return expected
    fingerprint = points_fingerprint(points)
    if expected is not None and fingerprint != expected:
        raise ValueError(
            "points does not match the exact calibration mesh; cellwise "
            "conformal needs the same mesh (coordinates, dtype, and point "
            f"order) on every call.{hint}"
        )
    return fingerprint


def check_point_alignment(
    key: str, tensor: Tensor, points: Tensor, name: str = "prediction"
) -> Tensor:
    """Require one leading entry of ``tensor`` per mesh point."""
    if tensor.ndim == 0 or tensor.shape[0] != points.shape[0]:
        raise ValueError(
            f"{_field_label(key)} ({name}): shape {tuple(tensor.shape)} must have "
            f"one leading entry per point (points has {points.shape[0]})."
        )
    return tensor


def clamp_min_floor(t: Tensor, eps: float) -> Tensor:
    """Clamp ``t`` to at least ``max(eps, finfo(t.dtype).tiny)``.

    ``eps`` alone can underflow to zero in float16.
    """
    if not math.isfinite(eps):
        raise ValueError(f"eps must be finite, got {eps}.")
    floor = eps
    if t.is_floating_point():
        info = torch.finfo(t.dtype)
        if eps > info.max:
            raise ValueError(
                f"eps={eps} exceeds the largest finite {t.dtype} value "
                f"({info.max}); use a smaller eps or a wider dtype."
            )
        floor = max(eps, float(info.tiny))
    return t.clamp_min(floor)


def broadcast_difficulty(t: Tensor, ref: Tensor, key: str) -> Tensor:
    """Reshape per-point scales to ``(n_points, 1, ...)`` to broadcast on ``ref``."""
    if t.ndim == 0:
        return t
    if ref.shape[0] != t.shape[0]:
        raise ValueError(
            f"{_field_label(key)}: AuxDifficulty gave {t.shape[0]} scales, but "
            f"the field has {ref.shape[0]} points (leading dimension)."
        )
    return t.reshape(t.shape[0], *([1] * (ref.ndim - 1)))


def normalize_keys(keys: Sequence[str] | None) -> tuple[str, ...] | None:
    """Return ``keys`` as a deduplicated tuple or ``None``; reject a single string."""
    if keys is None:
        return None
    if isinstance(keys, str):
        raise TypeError(
            f"keys must be a list of field names, not the string {keys!r}; "
            f"pass [{keys!r}]."
        )
    return tuple(dict.fromkeys(keys))


def positive_finite_float(value: object, name: str) -> float:
    """Coerce a strategy scalar (``eps``, ``value``) to a positive finite float."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a positive finite value, got {value!r}."
        ) from exc
    if not (value > 0 and math.isfinite(value)):
        raise ValueError(f"{name} must be a positive finite value, got {value}.")
    return value


def require_matching_keys(
    present: Iterable[str], expected: Iterable[str], subject: str
) -> None:
    """Raise ``KeyError`` unless both key sets are equal."""
    present = set(present)
    expected = set(expected)
    if present != expected:
        differing = sorted(present.symmetric_difference(expected))
        raise KeyError(f"{subject}; differing keys: {differing}.")


def check_exact_shape(
    key: str, name: str, tensor: Tensor, reference_name: str, reference: Tensor
) -> Tensor:
    """Require ``tensor.shape == reference.shape``, with no broadcasting."""
    if tensor.shape != reference.shape:
        raise ValueError(
            f"{_field_label(key)}: {name} shape {tuple(tensor.shape)} != "
            f"{reference_name} shape {tuple(reference.shape)}. Shapes must "
            "match exactly; broadcasting is not allowed."
        )
    return tensor


def check_aux(
    key: str,
    score: "_NonconformityScore",
    prediction: Tensor,
    aux: Mapping[str, Tensor] | None,
    pending: list[Tensor] | None = None,
) -> None:
    """Require the score's aux entries to be finite and match the prediction shape.

    An infinite sigma would otherwise make the score zero. ``pending`` works
    as in :func:`check_finite`.
    """
    if aux is None:
        return
    for aux_key in [k for k in score.aux_keys if k in aux]:
        value = aux[aux_key]
        if not isinstance(value, Tensor):
            raise TypeError(
                f"{_field_label(key)}: aux '{aux_key}' must be a torch.Tensor, got "
                f"{type(value).__name__}."
            )
        check_exact_shape(key, f"aux '{aux_key}'", value, "prediction", prediction)
        check_real(key, f"aux '{aux_key}'", value, pending)


def check_floating(key: str, name: str, tensor: Tensor) -> Tensor:
    """Require a floating dtype: integer data would truncate interval endpoints."""
    if not tensor.is_floating_point():
        raise TypeError(
            f"{_field_label(key)}: {name} must use a floating-point dtype, got "
            f"{tensor.dtype}; cast it with .float()."
        )
    return tensor


def check_finite(
    key: str, name: str, tensor: Tensor, pending: list[Tensor] | None = None
) -> Tensor:
    """Reject NaN and inf values with a per-field message.

    With ``pending``, append the finiteness flag there instead of syncing; the
    caller fetches all flags at once and repeats the checks if one fails.
    """
    finite = torch.isfinite(tensor).all()
    if pending is not None:
        pending.append(finite)
    elif not finite:
        n_bad = int((~torch.isfinite(tensor)).sum())
        raise ValueError(
            f"{_field_label(key)}: {n_bad} non-finite value(s) (NaN/inf) in {name}; "
            "remove or mask them first."
        )
    return tensor


def check_real(
    key: str, name: str, tensor: Tensor, pending: list[Tensor] | None = None
) -> Tensor:
    """Require finite floating-point data; ``pending`` as in :func:`check_finite`."""
    return check_finite(key, name, check_floating(key, name, tensor), pending)


def validate_provenance(value: object) -> dict:
    """Validate and copy a predictor's strict-JSON provenance mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"provenance must be a mapping, got {type(value).__name__}.")
    try:
        snapshot = json.loads(json.dumps(value, allow_nan=False))
    except ValueError as exc:
        raise ValueError(f"provenance must contain only finite numbers: {exc}") from exc
    except TypeError:
        snapshot = None
    # The round trip silently turns tuples into lists and non-string keys into
    # strings; comparing with the input rejects both.
    if snapshot != value:
        raise TypeError(
            "provenance must contain only strict-JSON values: string keys, lists "
            "(not tuples), and str, int, float, bool, or None leaves."
        )
    if "mesh_fingerprint" in snapshot:
        raise ValueError(
            "provenance must not contain 'mesh_fingerprint'; that name is "
            "reserved for the predictor's mesh check."
        )
    return snapshot
