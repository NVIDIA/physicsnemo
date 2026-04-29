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

"""
MultiDataset - Compose multiple Dataset instances behind a single dataset-like interface.

MultiDataset presents a single index space (concatenation of all constituent datasets)
and delegates __getitem__, prefetch, and close to the appropriate sub-dataset.
Each sub-dataset can have its own Reader and transforms. Optional output strictness
validates that all sub-datasets produce the same TensorDict keys (outputs) so default
collation works.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional, Sequence

import torch
from tensordict import TensorDict

from physicsnemo.datapipes._rng import fork_generator
from physicsnemo.datapipes.protocols import DatasetBase
from physicsnemo.datapipes.registry import register

# Metadata key added by MultiDataset to identify which sub-dataset produced the sample.
DATASET_INDEX_METADATA_KEY = "dataset_index"


def _validate_strict_outputs(datasets: Sequence[DatasetBase]) -> list[str]:
    """
    Check that all non-empty datasets produce the same TensorDict keys; return them.

    Loads one sample from each non-empty dataset and compares output keys (after
    transforms). This validates output schema, not reader field_names.

    Parameters
    ----------
    datasets : Sequence[DatasetBase]
        Datasets to validate.

    Returns
    -------
    list[str]
        Common output keys (sorted, from first non-empty dataset).

    Raises
    ------
    ValueError
        If any non-empty dataset has different output keys.
    """
    if not datasets:
        return []
    ref_keys: Optional[list[str]] = None
    ref_index: Optional[int] = None
    for i, ds in enumerate(datasets):
        if len(ds) == 0:
            continue
        data, _ = ds[0]
        keys = sorted(data.keys())
        if ref_keys is None:
            ref_keys = keys
            ref_index = i
        elif keys != ref_keys:
            raise ValueError(
                "output_strict=True requires identical output keys (TensorDict keys) "
                f"across datasets: dataset {ref_index} has {ref_keys}, dataset {i} has {keys}"
            )
    if ref_keys is not None:
        return list(ref_keys)
    first = datasets[0]
    return list(first.field_names) if hasattr(first, "field_names") else []


@register()
class MultiDataset:
    r"""
    A dataset that composes multiple :class:`DatasetBase` instances behind one index space.

    Accepts both :class:`Dataset` (TensorDict pipelines) and :class:`MeshDataset`
    (Mesh pipelines) as sub-datasets. Global indices are mapped to
    (dataset_index, local_index) by concatenation: indices 0..len0-1 come from the
    first dataset, len0..len0+len1-1 from the second, and so on. Each constituent
    can have its own Reader and transforms. Metadata is enriched with
    ``dataset_index`` so batches can identify the source.

    Parameters
    ----------
    *datasets : DatasetBase
        One or more Dataset or MeshDataset instances passed as positional
        arguments (Reader + transforms each). Order defines index mapping:
        first dataset occupies 0..len(ds0)-1, etc.
    output_strict : bool, default=True
        If True, require all datasets to produce the same TensorDict keys (output
        keys after transforms) so :class:`DefaultCollator` can stack batches. If
        False, no check is done; use a custom collator when keys or shapes differ.
        Note that `output_strict=True` will load the first instance of all datasets
        upon construction. Think of it as a debugging parameter: if you are sure
        that your datasets are working properly, and want to defer loading,
        you can safely disable this.

    Raises
    ------
    ValueError
        If no datasets are provided or if ``output_strict=True`` and output keys differ.

    Notes
    -----
    MultiDataset implements the same interface as :class:`Dataset` (``__len__``,
    ``__getitem__``, ``prefetch``,
    ``cancel_prefetch``, ``close``, ``field_names``) and can be passed to
    :class:`DataLoader` in place of a single dataset. Prefetch and close are
    delegated to the sub-dataset that owns the index. When ``output_strict=True``,
    validation checks that each dataset's *output* TensorDict (after transforms)
    has the same keys, not the reader's field_names. When ``output_strict=False``,
    :attr:`field_names` returns the first dataset's field names; with heterogeneous
    datasets, prefer a custom collator and use metadata ``dataset_index`` to
    group or pad by source.

    Shuffling and sampling
    ---------------------
    The DataLoader sees one linear index space of size :math:`\\sum_k \\text{len}(\\text{datasets}[k])`.
    With ``shuffle=True``, the default :class:`RandomSampler` shuffles these global
    indices, so each batch is a random mix of samples from all sub-datasets. There
    is no per-dataset balancing: if one dataset is much larger, its samples will
    appear more often. For balanced or stratified sampling, use a custom
    :class:`torch.utils.data.Sampler` (e.g. weighted or one sample per dataset per
    batch) and pass it to the DataLoader.

    Metadata
    --------
    Every sample returned by :meth:`__getitem__` has its metadata dict extended
    with the key :const:`DATASET_INDEX_METADATA_KEY` (``"dataset_index"``), the
    integer index of the sub-dataset that produced the sample (0 for the first
    dataset, 1 for the second, etc.). Sub-dataset–specific metadata (e.g. file
    path, index within that dataset) is unchanged. When using the DataLoader with
    ``collate_metadata=True``, each batch yields a list of metadata dicts aligned
    with the batch dimension; each dict includes ``dataset_index`` so you can
    filter, weight, or aggregate by source in the training loop.

    Examples
    --------
    >>> from physicsnemo.datapipes import Dataset, MultiDataset, HDF5Reader, Normalize
    >>> ds_a = Dataset(HDF5Reader("a.h5", fields=["x", "y"]), transforms=None)  # doctest: +SKIP
    >>> ds_b = Dataset(HDF5Reader("b.h5", fields=["x", "y"]), transforms=None)   # doctest: +SKIP
    >>> multi = MultiDataset(ds_a, ds_b, output_strict=True)                     # doctest: +SKIP
    >>> len(multi) == len(ds_a) + len(ds_b)                                      # doctest: +SKIP
    True
    >>> data, meta = multi[0]   # from ds_a                                      # doctest: +SKIP
    >>> meta["dataset_index"]   # 0                                              # doctest: +SKIP
    """

    def __init__(
        self,
        *datasets: DatasetBase,
        output_strict: bool = True,
    ) -> None:
        if len(datasets) < 1:
            raise ValueError(
                f"MultiDataset requires at least one dataset, got {len(datasets)}"
            )
        for i, ds in enumerate(datasets):
            if not isinstance(ds, DatasetBase):
                raise TypeError(
                    f"datasets[{i}] must be a Dataset or MeshDataset instance, "
                    f"got {type(ds).__name__}"
                )

        self._datasets = list(datasets)
        self._output_strict = output_strict

        # Cumulative lengths: cumul[k] = sum(len(datasets[j]) for j in range(k))
        # So index i is in dataset k when cumul[k] <= i < cumul[k+1], local = i - cumul[k]
        cumul = [0]
        for ds in self._datasets:
            cumul.append(cumul[-1] + len(ds))
        self._cumul = cumul

        if output_strict:
            self._field_names = _validate_strict_outputs(self._datasets)
        else:
            first = self._datasets[0]
            if hasattr(first, "field_names"):
                self._field_names = list(first.field_names)
            else:
                self._field_names = []

    def _index_to_dataset_and_local(self, index: int) -> tuple[int, int]:
        """
        Map global index to (dataset_index, local_index).

        Parameters
        ----------
        index : int
            Global index in [0, len(self)).

        Returns
        -------
        tuple[int, int]
            (dataset_index, local_index).

        Raises
        ------
        IndexError
            If index is out of range.
        """
        n = len(self)
        if index < 0:
            index = n + index
        if index < 0 or index >= n:
            raise IndexError(
                f"Index {index} out of range for MultiDataset with {n} samples"
            )
        # Find k such that cumul[k] <= index < cumul[k+1]
        for k in range(len(self._cumul) - 1):
            if self._cumul[k] <= index < self._cumul[k + 1]:
                return k, index - self._cumul[k]
        # Fallback (should not be reached)
        return len(self._datasets) - 1, index - self._cumul[-2]

    def _index_to_dataset_and_local_optional(
        self, index: int
    ) -> Optional[tuple[int, int]]:
        """
        Map global index to (dataset_index, local_index), or None if out of range.

        Used by cancel_prefetch to match Dataset behavior (no-op for invalid index).
        """
        n = len(self)
        if index < 0:
            index = n + index
        if index < 0 or index >= n:
            return None
        for k in range(len(self._cumul) - 1):
            if self._cumul[k] <= index < self._cumul[k + 1]:
                return k, index - self._cumul[k]
        return len(self._datasets) - 1, index - self._cumul[-2]

    def __len__(self) -> int:
        """Return the total number of samples (sum of all sub-dataset lengths)."""
        return self._cumul[-1]

    # ------------------------------------------------------------------
    # RNG management
    # ------------------------------------------------------------------

    def set_generator(self, generator: torch.Generator) -> None:
        """Fork *generator* and distribute one child per sub-dataset.

        Parameters
        ----------
        generator : torch.Generator
            Parent generator (typically forked from the DataLoader's
            master generator).
        """
        children = fork_generator(generator, len(self._datasets))
        for child, ds in zip(children, self._datasets):
            if hasattr(ds, "set_generator"):
                ds.set_generator(child)

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch to every sub-dataset.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        for ds in self._datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

    def __getitem__(self, index: int) -> tuple[TensorDict, dict[str, Any]]:
        """
        Return the transformed sample and metadata for the given global index.

        Metadata is enriched with ``dataset_index`` (key :const:`DATASET_INDEX_METADATA_KEY`).

        Parameters
        ----------
        index : int
            Global sample index. Supports negative indexing.

        Returns
        -------
        tuple[TensorDict, dict[str, Any]]
            (TensorDict, metadata dict) from the owning sub-dataset.
        """
        ds_id, local_i = self._index_to_dataset_and_local(index)
        data, metadata = self._datasets[ds_id][local_i]
        metadata = dict(metadata)
        metadata[DATASET_INDEX_METADATA_KEY] = ds_id
        return data, metadata

    def prefetch(
        self,
        index: int,
        stream: Optional[Any] = None,
    ) -> None:
        """
        Start prefetching the sample at the given global index.

        Delegates to the sub-dataset that owns that index.

        Parameters
        ----------
        index : int
            Global sample index to prefetch.
        stream : object, optional
            Optional CUDA stream for the sub-dataset prefetch.
        """
        ds_id, local_i = self._index_to_dataset_and_local(index)
        self._datasets[ds_id].prefetch(local_i, stream=stream)

    def cancel_prefetch(self, index: Optional[int] = None) -> None:
        """
        Cancel prefetch for the given index or all sub-datasets.

        When index is provided, only cancels if it is in range; out-of-range
        indices are ignored to match :class:`Dataset` behavior.

        Parameters
        ----------
        index : int, optional
            Global index to cancel, or None to cancel all.
        """
        if index is None:
            for ds in self._datasets:
                ds.cancel_prefetch(None)
        else:
            mapped = self._index_to_dataset_and_local_optional(index)
            if mapped is not None:
                ds_id, local_i = mapped
                self._datasets[ds_id].cancel_prefetch(local_i)

    def __iter__(self) -> Iterator[tuple[TensorDict, dict[str, Any]]]:
        """Iterate over all samples in global index order."""
        for i in range(len(self)):
            yield self[i]

    @property
    def field_names(self) -> list[str]:
        """
        Field names in samples.

        With ``output_strict=True``, returns the common output keys (TensorDict
        keys after transforms). With ``output_strict=False``, returns the first
        dataset's field names.
        """
        return list(self._field_names)

    def close(self) -> None:
        """Close all sub-datasets and release resources."""
        for ds in self._datasets:
            ds.close()

    def __enter__(self) -> "MultiDataset":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        parts = [f"  ({i}): {ds}" for i, ds in enumerate(self._datasets)]
        return (
            f"MultiDataset(\n  output_strict={self._output_strict},\n  datasets=[\n"
            + ",\n".join(parts)
            + "\n  ]\n)"
        )
