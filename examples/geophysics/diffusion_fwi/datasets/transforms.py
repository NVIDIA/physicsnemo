# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from abc import ABC
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import torch
import torch.nn.functional as F


class DataLoaderUfuncTransform(ABC):
    """
    Base class for dataloader transforms that apply element-wise callables to
    each element of a batch.

    The instance is itself callable. When called with an iterator (e.g.
    ``torch.utils.data.DataLoader``), it returns a *generator* that lazily
    applies the provided unary functions (``ufuncs``) to every batch yielded
    by the iterator.

    Parameters
    ----------
    ufuncs : Callable[[torch.Tensor], Any] | Dict[str, Optional[Callable]] |
        Tuple[Optional[Callable], ...] | List[Optional[Callable]]
        • When batches are dictionaries, ``ufuncs`` must be a mapping from
          keys to callables (or ``None`` to leave the element untouched).
        • When batches are tuples / lists, ``ufuncs`` must be a sequence whose
          length matches the batch length. ``None`` values are treated as the
          identity transform.

    Returns
    -------
    Iterator[Any]
        Iterator yielding batches of tensors transformed by the provided
        ufuncs.

    Examples
    --------
    >>> # Batches are dictionaries
    >>> tf = DataLoaderUfuncTransform({"x": f1, "y": f2, "z": None})
    >>> new_iter = tf(old_iter)
    >>> # Batches are tuples / lists
    >>> tf = DataLoaderUfuncTransform((f1, f2, None))
    >>> new_iter = tf(old_iter)
    """

    def __init__(
        self,
        iterator: Iterator[Any],
        ufuncs: Union[
            Callable[[torch.Tensor], Any],
            Dict[str, Optional[Callable[[torch.Tensor], Any]]],
            Tuple[Optional[Callable[[torch.Tensor], Any]], ...],
            List[Optional[Callable[[torch.Tensor], Any]]],
        ],
    ) -> None:
        """Wrap *iterator* and apply *ufuncs* lazily to its batches."""

        super().__init__()
        self._iter = iterator
        self._ufuncs = ufuncs

    def __iter__(self):
        self._iterator = iter(self._iter)
        return self

    def __next__(self):
        batch = next(self._iterator)
        return self._apply_to_batch(batch)

    def __getattr__(self, name):  # Delegate unknown attrs to base iterator
        return getattr(self._iter, name)

    def __len__(self):
        return len(self._iter)

    def _apply_single(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply transform(s) to a single tensor."""

        if callable(self._ufuncs):
            return self._ufuncs(tensor)

        if isinstance(self._ufuncs, (list, tuple)) and len(self._ufuncs) == 1:
            fn = self._ufuncs[0]
            return fn(tensor) if callable(fn) else tensor

        raise ValueError(
            "Expected a single callable or a list/tuple of callables with "
            "length 1, got a list/tuple with length "
            f"{len(self._ufuncs)}."
        )

    def _apply_to_batch(self, batch: Any) -> Any:  # noqa: D401
        """Apply the stored ufuncs to a batch (tensor(s))."""

        # Single tensor batch
        if torch.is_tensor(batch):
            return self._apply_single(batch)

        # Dict batch
        if isinstance(batch, dict):
            if not all(torch.is_tensor(v) for v in batch.values()):
                raise TypeError("All values in dict batch must be tensors.")

            if callable(self._ufuncs):
                return {k: self._ufuncs(v) for k, v in batch.items()}

            if isinstance(self._ufuncs, dict):
                out: Dict[str, Any] = {}
                for k, v in batch.items():
                    fn = self._ufuncs.get(k)
                    out[k] = fn(v) if callable(fn) else v
                return out

            raise TypeError("Dict batch requires callable or dict ufuncs.")

        # List / tuple batch
        if isinstance(batch, (list, tuple)):
            if not all(torch.is_tensor(v) for v in batch):
                raise TypeError("All elements in list/tuple batch must be tensors.")

            if callable(self._ufuncs):
                transformed = [self._ufuncs(v) for v in batch]
            else:
                if not isinstance(self._ufuncs, (list, tuple)):
                    raise TypeError(
                        "List/tuple batch requires callable or list/tuple ufuncs."
                    )
                if len(self._ufuncs) != len(batch):
                    raise ValueError(
                        "Length mismatch between ufuncs and list/tuple batch."
                    )
                transformed = [
                    (fn(v) if callable(fn) else v) for v, fn in zip(batch, self._ufuncs)
                ]

            if isinstance(batch, tuple):
                return tuple(transformed)
            return transformed

        raise TypeError(
            f"Unsupported batch type: {type(batch).__name__}. Expected "
            "tensor, dict, list or tuple."
        )

    def _validate_args(
        self, *args: Any, args_names: Optional[Sequence[str]] = None
    ) -> None:
        """Validate that all *args* share a compatible structure.

        Rules
        -----
        1. If the reference argument is a ``dict``, every other argument must
           also be a ``dict`` with the **same** keys.
        2. If the reference argument is a list/tuple, every other argument must
           also be a list/tuple of the **same** length.
        3. Otherwise the reference is treated as *scalar* and no other argument
           may be a container type (dict, list or tuple).
        """

        if not args:
            return  # nothing to validate

        if args_names is None:
            args_names = [f"arg{i}" for i in range(len(args))]
        if len(args_names) != len(args):
            raise ValueError("args and args_names must have the same length.")

        ref, ref_name = args[0], args_names[0]

        # For dict args
        if isinstance(ref, dict):
            for other, name in zip(args[1:], args_names[1:]):
                if not isinstance(other, dict):
                    raise TypeError(f"{name} must be dict when {ref_name} is dict.")
                if ref.keys() != other.keys():
                    raise ValueError(
                        f"{ref_name} and {name} dicts must share the same keys."
                    )
            return

        # For sequence args
        if isinstance(ref, (list, tuple)):
            for other, name in zip(args[1:], args_names[1:]):
                if not isinstance(other, (list, tuple)):
                    raise TypeError(
                        f"{name} must be a sequence when {ref_name} is a sequence."
                    )
                if len(ref) != len(other):
                    raise ValueError(
                        f"{ref_name} and {name} must be same-length sequences."
                    )
            return

        # Scalar reference
        for other, name in zip(args[1:], args_names[1:]):
            if isinstance(other, (dict, list, tuple)):
                raise TypeError(f"{ref_name} is scalar but {name} is container type.")


# ---------------------------------------------------------------------------
# Convenience transforms
# ---------------------------------------------------------------------------


class ToDevice(DataLoaderUfuncTransform):
    """Transform to move tensors of each batch to the given device(s).

    Parameters
    ----------
    iterator : Iterator[Any]
        Original data iterator yielding batches of tensors.
    device : (torch.device | str) | Dict[str, Optional[device]] | Sequence[Optional[device]]
        • *Scalar* : same target device for every tensor.
        • *Dict*   : key-specific target devices for *dict* batches; ``None``
          leaves the tensor on its current device.
        • *Sequence* : position-specific target devices for tuple/list batches;
          ``None`` leaves the tensor untouched.
    non_blocking : bool, optional
        Forwarded to ``Tensor.to``.
    """

    def __init__(
        self,
        iterator: Iterator[Any],
        device: Union[
            torch.device,
            str,
            Dict[str, Optional[Union[torch.device, str]]],
            Sequence[Optional[Union[torch.device, str]]],
        ],
        *,
        non_blocking: bool = False,
    ) -> None:
        self._validate_args(device, args_names=("device",))

        if isinstance(device, (torch.device, str)):

            def _move(t: torch.Tensor) -> torch.Tensor:
                return t.to(device=device, non_blocking=non_blocking)

            ufuncs = _move

        elif isinstance(device, dict):
            ufuncs = {}
            for k, dev in device.items():
                if dev is None:
                    ufuncs[k] = None
                else:
                    ufuncs[k] = lambda t, dev=dev: t.to(
                        device=dev, non_blocking=non_blocking
                    )

        elif isinstance(device, Sequence):
            ufuncs_list: List[Optional[Callable]] = []
            for dev in device:
                if dev is None:
                    ufuncs_list.append(None)
                else:
                    ufuncs_list.append(
                        lambda t, dev=dev: t.to(device=dev, non_blocking=non_blocking)
                    )
            ufuncs = ufuncs_list

        else:
            raise TypeError("Unsupported type for device parameter.")

        super().__init__(iterator, ufuncs)


class ZscoreNormalize(DataLoaderUfuncTransform):
    """Z-score normalisation transform.

    Parameters
    ----------
    iterator : Iterator[Any]
        Data iterator yielding batches.
    mean, std : scalar | dict | sequence
        ``mean`` and ``std`` follow the same rules:
            • *Scalar*: same value for every tensor.
            • *Dict*: key-specific values for *dict* batches. Missing or ``None``
            disables normalisation for that key.
            • *Tuple/List*: position-specific values for tuple/list batches; ``None``
            disables normalisation for that position.
    eps : float, optional
        Small term to avoid division by zero.
    """

    def __init__(
        self,
        iterator: Iterator[Any],
        mean: Union[float, Dict[str, Optional[float]], Sequence[Optional[float]]],
        std: Union[float, Dict[str, Optional[float]], Sequence[Optional[float]]],
        *,
        eps: float = 1e-8,
    ) -> None:
        self._validate_args(mean, std, args_names=("mean", "std"))

        # Build ufuncs
        if isinstance(mean, (int, float)):

            def _scalar_norm(t: torch.Tensor) -> torch.Tensor:  # noqa: D401
                return (t - mean) / (std + eps)

            ufuncs: Any = _scalar_norm

        elif isinstance(mean, dict):
            ufuncs = {}
            for k, m in mean.items():
                s = std[k]
                if m is None or s is None:
                    ufuncs[k] = None
                else:
                    ufuncs[k] = lambda t, m=m, s=s: (t - m) / (s + eps)

        elif isinstance(mean, Sequence):
            seq: List[Optional[Callable]] = []
            for m, s in zip(mean, std):
                if m is None or s is None:
                    seq.append(None)
                else:
                    seq.append(lambda t, m=m, s=s: (t - m) / (s + eps))
            ufuncs = seq

        else:
            raise TypeError("Unsupported type for mean/std parameters.")

        super().__init__(iterator, ufuncs)


class MinMaxNormalize(DataLoaderUfuncTransform):
    """Transform to scale tensors to `[-1, 1]` using provided min / max statistics.

    Parameters
    ----------
    iterator : Iterator[Any]
        Data iterator yielding batches.
    min_val, max_val : scalar | dict | sequence
        ``min_val`` and ``max_val`` follow the same rules:
            • *Scalar*: same value for every tensor.
            • *Dict*: key-specific values for *dict* batches. Missing or ``None``
            disables normalisation for that key.
            • *Tuple/List*: position-specific values for tuple/list batches; ``None``
            disables normalisation for that position.
    eps : float, optional
        Small term to avoid division by zero.
    """

    def __init__(
        self,
        iterator: Iterator[Any],
        min_val: Union[
            float,
            Dict[str, Optional[float]],
            Sequence[Optional[float]],
        ],
        max_val: Union[
            float,
            Dict[str, Optional[float]],
            Sequence[Optional[float]],
        ],
        *,
        eps: float = 1e-8,
    ) -> None:
        self._validate_args(min_val, max_val, args_names=("min_val", "max_val"))

        if isinstance(min_val, (int, float)):

            def _scalar_scale(t: torch.Tensor) -> torch.Tensor:
                return 2.0 * ((t - min_val) / (max_val - min_val + eps)) - 1.0

            ufuncs: Any = _scalar_scale

        elif isinstance(min_val, dict):
            ufuncs = {}
            for k, a in min_val.items():
                b = max_val[k]
                if a is None or b is None:
                    ufuncs[k] = None
                else:
                    ufuncs[k] = (
                        lambda t, a=a, b=b: 2.0 * ((t - a) / (b - a + eps)) - 1.0
                    )

        elif isinstance(min_val, Sequence):
            seq_funcs: List[Optional[Callable]] = []
            for a, b in zip(min_val, max_val):
                if a is None or b is None:
                    seq_funcs.append(None)
                else:
                    seq_funcs.append(
                        lambda t, a=a, b=b: 2.0 * ((t - a) / (b - a + eps)) - 1.0
                    )
            ufuncs = seq_funcs

        else:
            raise TypeError("Unsupported type for min_val/max_val parameters.")

        super().__init__(iterator, ufuncs)


class Interpolate(DataLoaderUfuncTransform):
    """Selective 2D interpolation on 4D tensors (N, C, H, W).

    Assumptions
    -----------
    • ``dim`` is one of: tuple-of-tuples, dict[str, tuple], sequence-of-tuples.
    • ``size`` follows the same container structure as ``dim``.
      Each inner size tuple length must match its inner dim tuple length.
    • ``mode`` follows the same container structure as ``dim`` and ``size``.
    """

    def __init__(
        self,
        iterator: Iterator[Any],
        size: Dict[str, Optional[Tuple[int, ...]]]
        | Sequence[Optional[Tuple[int, ...]]],
        dim: Dict[str, Optional[Tuple[int, ...]]] | Sequence[Optional[Tuple[int, ...]]],
        *,
        mode: Dict[str, Optional[str]] | Sequence[Optional[str]] = "bilinear",
    ) -> None:
        # Validate container alignment (all three share structure)
        self._validate_args(dim, size, args_names=("dim", "size"))

        # Factory to build interp ufunc
        def _make_interp(
            dims: Tuple[int, ...],
            sizes: Tuple[int, ...],
            mode_str: str,
        ) -> Callable[[torch.Tensor], torch.Tensor]:
            def _fn(t: torch.Tensor) -> torch.Tensor:
                if t.dim() != 4:
                    raise ValueError(
                        "InterpolateTransform expects 4D tensors (B, C, H, W)."
                    )
                if len(dims) != len(sizes):
                    raise ValueError("Each dim entry must have a matching size.")

                # Get output height and width
                _, _, h, w = t.shape
                oh, ow = h, w
                for d, s in zip(dims, sizes):
                    if d == 2 or d == -2:
                        oh = int(s)
                    elif d == 3 or d == -1:
                        ow = int(s)
                    else:
                        raise ValueError(f"Unsupported dim: {d}")

                return F.interpolate(t, size=(oh, ow), mode=mode_str)

            return _fn

        # Dict container
        if isinstance(dim, dict):
            ufuncs_dict: Dict[str, Optional[Callable]] = {}
            for k, dims_k in dim.items():
                sizes_k = size[k]
                mode_k = mode[k]
                if dims_k is None or sizes_k is None or mode_k is None:
                    ufuncs_dict[k] = None
                else:
                    ufuncs_dict[k] = _make_interp(dims_k, sizes_k, mode_k)
            ufuncs = ufuncs_dict

        # TODO: probably some way to simplify this
        # Sequence container
        elif isinstance(dim, (list, tuple)):
            if len(dim) != len(size) or len(dim) != len(mode):
                raise ValueError("dim, size and mode must be same-length sequences.")
            # Single spec: apply to every element via callable
            if len(dim) == 1:
                dims0 = dim[0]
                sizes0 = size[0]
                mode0 = mode
                if dims0 is None or sizes0 is None or mode0 is None:
                    ufuncs = lambda t: t
                else:
                    ufuncs = _make_interp(dims0, sizes0, mode0)
            else:
                ufuncs_list: List[Optional[Callable]] = []
                for dims_i, sizes_i, mode_i in zip(dim, size, mode):
                    if dims_i is None or sizes_i is None or mode_i is None:
                        ufuncs_list.append(None)
                    else:
                        ufuncs_list.append(_make_interp(dims_i, sizes_i, mode_i))
                ufuncs = ufuncs_list
        else:
            raise TypeError("Unsupported structure for dim/size/mode parameters.")

        super().__init__(iterator, ufuncs)
