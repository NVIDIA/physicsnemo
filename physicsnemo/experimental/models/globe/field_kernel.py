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

import itertools
import logging
from functools import cache, cached_property
from math import comb, prod
from typing import TYPE_CHECKING, Final, Literal, Sequence

import psutil
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from tensordict import PackLayout, TensorDict
from tensordict.utils import NestedKey
from torch.profiler import record_function
from torch.utils.checkpoint import checkpoint

from physicsnemo.core.module import Module
from physicsnemo.mesh import FieldSchema, FieldSchemaLike
from physicsnemo.nn import Mlp, Pade
from physicsnemo.nn.functional.equivariant_ops import (
    legendre_polynomials,
    smooth_log,
    vector_project,
)

logger = logging.getLogger("globe.field_kernel")

if TYPE_CHECKING:
    from physicsnemo.mesh.spatial.cluster_tree import (
        ClusterTree,
        DualInteractionPlan,
    )


### Static, per-device memory-budget helpers used by chunk-size sizing.

# Fraction of total device memory we are willing to spend on a single
# kernel evaluation's chunked autograd state (peak intermediate footprint
# for one (Phase A | B | C | D) call).  Conservative because parameters,
# DDP buckets, and activations from other kernels coexist on-device.
#
# Expressed as an integer percent (rather than e.g. ``0.25``) so that
# ``_device_chunk_budget_bytes`` stays in pure-integer arithmetic.  Under
# ``torch.compile`` with the default ``specialize_float=False``, a
# module-level Python float is treated as an unbacked symbolic float
# ("zf" symbol), which then infects every ``chunk_size`` derivation
# downstream and crashes Dynamo at the ``checkpoint`` boundary in
# ``_gather_and_evaluate`` with "Source of '...' is None when lifting it
# to input of top-level".
_CHUNK_MEMORY_BUDGET_PERCENT: Final[int] = 25


def _ceil_div(a: int, b: int) -> int:
    """Integer ceiling division: equivalent to ``math.ceil(a / b)`` for
    non-negative ``a`` and positive ``b``, but with no Python ``float``
    intermediate.

    Used here because, under ``torch.compile`` with the default
    ``specialize_float=False``, a Python ``float`` in the trace is treated
    as an unbacked symbolic free variable that propagates into the chunk
    size and crashes Dynamo at the ``checkpoint`` boundary in
    :meth:`BarnesHutKernel._gather_and_evaluate`.  ``-(-a // b)`` is the
    standard Python idiom: floor division rounds toward negative infinity,
    so negating both ends produces a ceiling on the original quotient.
    """
    return -(-a // b)


@torch.compiler.disable
@cache
def _device_total_memory_bytes(device: torch.device) -> int:
    """Cached lookup of total physical memory available on ``device``.

    On CUDA, returns ``torch.cuda.get_device_properties.total_memory``.
    On CPU, returns total system RAM via :func:`psutil.virtual_memory`.
    The CPU path is debug-only - production runs on CUDA.

    ``@torch.compiler.disable`` is layered above ``@cache`` so Dynamo
    bails on the outer call and never traces the device query here.
    Without this, Dynamo treats ``torch.cuda.get_device_properties``
    as an opaque Python function and tries to evaluate it as a
    constant during graph capture, which crashes on CPU-only hosts
    (``RuntimeError: Found no NVIDIA driver``) and produces unbacked
    symbolic ints on CUDA hosts that poison downstream chunk-size
    arithmetic.
    """
    if device.type == "cuda":
        return int(torch.cuda.get_device_properties(device).total_memory)
    if device.type == "cpu":
        return int(psutil.virtual_memory().total)
    raise ValueError(f"Unsupported {device.type=!r}")


@torch._dynamo.assume_constant_result
def _device_chunk_budget_bytes(device: torch.device) -> int:
    """Static memory budget for a single chunked kernel evaluation."""
    return _device_total_memory_bytes(device) * _CHUNK_MEMORY_BUDGET_PERCENT // 100


def _sort_key(key: NestedKey) -> str:
    r"""Sort key that fixes where each named feature goes in a packed tensor.

    The kernels have always ordered their features by ``str`` of the full
    nested key, so this compares ``repr``\ s of key tuples below a common
    prefix.  The network's input and output columns follow this order, so
    changing it would scramble the columns of every trained checkpoint.
    (:meth:`TensorDict.pack <tensordict.TensorDictBase.pack>` sorts nested
    keys as dotted strings, which disagrees for some keys, so the kernels
    pack with layouts in this order; see :func:`_input_layout`.)

    Parameters
    ----------
    key : NestedKey
        A feature's key: a string, or a tuple of strings for a nested key.

    Returns
    -------
    str
        The ``str`` of the key as a tuple below a common prefix.
    """
    return str(("_", *((key,) if isinstance(key, str) else key)))


def _input_layout(
    schema: FieldSchema, n_spatial_dims: int, *, sort_dotted_names: bool
) -> PackLayout:
    r"""Layout in which the kernels pack one group of input features.

    The scalar fields come first, then the vector fields, each group in
    :func:`_sort_key` order.  The network's input columns follow this order,
    so trained checkpoints depend on it.  Leaves are keyed by the nested keys
    of :meth:`FieldSchema.key <physicsnemo.mesh.FieldSchema.key>`.  Vector
    leaves keep their components together, so the :math:`N_v D` vector
    columns reshape to :math:`(*, N_v, D)`.

    Parameters
    ----------
    schema : FieldSchema
        The scalar and vector fields of the group.
    n_spatial_dims : int
        Number of components of each vector field.
    sort_dotted_names : bool
        Whether to order the fields by their dotted names instead of their
        nested keys.  This keeps the column order of earlier versions, which
        saw source features with flattened keys and global features with
        nested keys.  The two orders differ only for unusual names.

    Returns
    -------
    PackLayout
        Layout for :meth:`TensorDict.pack <tensordict.TensorDictBase.pack>`,
        whose packed tensor has shape :math:`(*, N_s + N_v D)` for
        :math:`N_s` scalar and :math:`N_v` vector fields.
    """
    ranks = schema.ranks

    def order(name: str) -> str:
        return _sort_key(name if sort_dotted_names else FieldSchema.key(name))

    names = sorted((n for n in ranks if ranks[n] == 0), key=order) + sorted(
        (n for n in ranks if ranks[n] == 1), key=order
    )
    return PackLayout(
        keys=tuple(FieldSchema.key(n) for n in names),
        shapes=tuple(() if ranks[n] == 0 else (n_spatial_dims,) for n in names),
    )


def _pack_inputs(
    td: TensorDict,
    layout: PackLayout,
    device: torch.device,
    label: Literal["source_data", "global_data"],
) -> Float[torch.Tensor, "*batch n_features"]:
    r"""Pack ``td`` in ``layout``, with an empty result on ``device`` if ``layout`` has no fields.

    Parameters
    ----------
    td : TensorDict
        Features whose leaves are exactly the fields of ``layout``.
    layout : PackLayout
        Layout from :func:`_input_layout`.
    device : torch.device
        Device of the empty result: a tensordict packs its own fields only.
    label : {"source_data", "global_data"}
        Name of ``td`` in error messages.

    Returns
    -------
    Float[torch.Tensor, "*batch n_features"]
        The packed features, of shape :math:`(*, F)`.

    Raises
    ------
    KeyError
        If the leaves of ``td`` differ from the fields of ``layout``.
    ValueError
        If a leaf has another shape than its field.
    """
    ### ``pack`` raises for wrong fields too, but without naming ``td``.
    if not torch.compiler.is_compiling():

        def path(key: NestedKey) -> tuple[str, ...]:
            return (key,) if isinstance(key, str) else tuple(key)

        actual = {path(k) for k in td.keys(include_nested=True, leaves_only=True)}
        expected = {path(k) for k in layout.keys}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            message = (
                f"{label} must have exactly the declared fields; missing: "
                f"{[k[0] if len(k) == 1 else k for k in missing]}, extra: "
                f"{[k[0] if len(k) == 1 else k for k in extra]}."
            )
            flat_keys = [
                k[0] for k in extra if len(k) == 1 and tuple(k[0].split(".")) in missing
            ]
            if flat_keys:
                message += (
                    f" Flat keys such as {flat_keys[0]!r} must be nested, as "
                    f"{tuple(flat_keys[0].split('.'))!r}."
                )
            raise KeyError(message)
    flat, _ = td.pack(layout=layout)
    if not layout.keys:
        return torch.zeros(td.batch_size + (0,), device=device)
    return flat


def _dipole_directions(
    r_hat: Float[torch.Tensor, "... n_dims"],
    n_hats: Float[torch.Tensor, "... n_vectors n_dims"],
) -> Float[torch.Tensor, "... n_vectors n_dims"]:
    r"""``-n_hat`` projected onto the plane normal to ``r_hat``, for each ``n_hat``.

    This is ``e_theta`` of
    :func:`~physicsnemo.nn.functional.equivariant_ops.spherical_basis` in 3D
    and ``e_kappa`` of
    :func:`~physicsnemo.nn.functional.equivariant_ops.polar_and_dipole_basis`
    in 2D, both with ``normalize_basis_vectors=False``: the only directions
    of those bases that the kernels use.  It is zero where ``r_hat`` is.

    Parameters
    ----------
    r_hat : Float[torch.Tensor, "... n_dims"]
        Unit or zero displacement directions, of shape :math:`(*, D)`.
    n_hats : Float[torch.Tensor, "... n_vectors n_dims"]
        Unit directions of the other vectors, of shape :math:`(*, N_v, D)`.

    Returns
    -------
    Float[torch.Tensor, "... n_vectors n_dims"]
        Unnormalized dipole directions, of shape :math:`(*, N_v, D)`.
    """
    e = vector_project(-n_hats, r_hat.unsqueeze(-2))
    return e.masked_fill(torch.all(r_hat == 0.0, dim=-1)[..., None, None], 0.0)


def _aggregate_features(
    cluster_tree: "ClusterTree",
    source_points: Float[torch.Tensor, "n_sources n_dims"],
    source_areas: Float[torch.Tensor, " n_sources"],
    source_features: Float[torch.Tensor, "n_sources n_features"],
) -> tuple[
    Float[torch.Tensor, "n_nodes n_dims"], Float[torch.Tensor, "n_nodes n_features"]
]:
    r"""Area-weighted node centroids and mean packed features of a source tree.

    Averaging is linear in each column, so the packed features aggregate as
    one TensorDict leaf, in one pass, and stay packed.

    Parameters
    ----------
    cluster_tree : ClusterTree
        Tree over ``source_points``.
    source_points : Float[torch.Tensor, "n_sources n_dims"]
        Source coordinates, of shape :math:`(N_{sources}, D)`.
    source_areas : Float[torch.Tensor, "n_sources"]
        Source weights, of shape :math:`(N_{sources},)`.
    source_features : Float[torch.Tensor, "n_sources n_features"]
        Packed source features, of shape :math:`(N_{sources}, F)`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Node centroids of shape :math:`(N_{nodes}, D)`, and mean node features
        of shape :math:`(N_{nodes}, F)`.
    """
    aggregates = cluster_tree.compute_source_aggregates(
        source_points=source_points,
        areas=source_areas,
        source_data=TensorDict(
            {"features": source_features}, batch_size=source_features.shape[:1]
        ),
    )
    if aggregates.node_source_data is None:  # a tree without nodes
        return aggregates.node_centroid, source_features.new_zeros(
            (0, source_features.shape[-1])
        )
    return aggregates.node_centroid, aggregates.node_source_data["features"]


def _target_node_centroids(
    cluster_tree: "ClusterTree",
    target_tree: "ClusterTree",
    target_points: Float[torch.Tensor, "n_targets n_dims"],
    node_centroids: Float[torch.Tensor, "n_nodes n_dims"],
) -> Float[torch.Tensor, "n_target_nodes n_dims"]:
    r"""Centroids of the target-tree nodes, for the far-target phases.

    For self-interaction (``target_tree is cluster_tree``) these are the
    source tree's area-weighted ``node_centroids``; otherwise the plain mean
    of each target node's points.

    Parameters
    ----------
    cluster_tree : ClusterTree
        Source tree of the plan.
    target_tree : ClusterTree
        Target tree or forest of the plan, over ``target_points``.
    target_points : Float[torch.Tensor, "n_targets n_dims"]
        Target coordinates, of shape :math:`(N_{targets}, D)`.
    node_centroids : Float[torch.Tensor, "n_nodes n_dims"]
        Area-weighted centroids of the source tree's nodes, of shape
        :math:`(N_{nodes}, D)`.

    Returns
    -------
    Float[torch.Tensor, "n_target_nodes n_dims"]
        Centroids of the :math:`M` target-tree nodes, of shape :math:`(M, D)`.
    """
    if target_tree is cluster_tree:
        return node_centroids
    return target_tree.compute_source_aggregates(
        source_points=target_points,
        areas=torch.ones(
            target_points.shape[0],
            device=target_points.device,
            dtype=target_points.dtype,
        ),
        source_data=None,
    ).node_centroid


def _output_layout(
    schema: FieldSchema, n_spatial_dims: int, *, nested: bool = False
) -> PackLayout:
    r"""Layout of a kernel's packed output fields.

    Fields are in sorted name order, a scalar taking one column and a vector
    ``n_spatial_dims``.  Leaves are keyed by the dotted field names, or, with
    ``nested=True``, by the nested keys of :meth:`FieldSchema.key
    <physicsnemo.mesh.FieldSchema.key>`.

    Parameters
    ----------
    schema : FieldSchema
        The kernel's output fields.
    n_spatial_dims : int
        Number of components of each vector field.
    nested : bool, optional, default=False
        Whether to key the leaves by nested keys instead of dotted names.

    Returns
    -------
    PackLayout
        Layout whose ``unpack`` turns a packed tensor of shape
        :math:`(*, F_{out})` into the named fields.
    """
    ranks = schema.ranks
    names = tuple(sorted(ranks))
    return PackLayout(
        keys=tuple(FieldSchema.key(n) for n in names) if nested else names,
        shapes=tuple(() if ranks[n] == 0 else (n_spatial_dims,) for n in names),
        ### Any device: ``unpack`` then puts the fields on the device of the
        ### packed tensor.
        device=torch.device("cpu"),
    )


def _require_supported(schema: FieldSchema, *, label: str) -> None:
    """Refuse field declarations the kernels do not implement.

    The kernels form invariant features from scalars and the dot products of
    true vectors, and build vector outputs on a basis of true vectors.  A
    field of rank 2 or more would be silently left out, and a pseudotensor
    would transform with the wrong sign under reflections.

    Parameters
    ----------
    schema : FieldSchema
        Parsed field declarations to check.
    label : str
        Name of the schema argument, used in error messages.

    Raises
    ------
    NotImplementedError
        If a field has rank other than 0 or 1, or has ``parity='odd'``.
    """
    for name, spec in schema.items():
        if spec.rank not in (0, 1):
            raise NotImplementedError(
                f"{label}[{name!r}]: rank-{spec.rank} fields are not supported; "
                f"GLOBE implements scalars (rank 0) and vectors (rank 1)"
            )
        if spec.parity != "even":
            raise NotImplementedError(
                f"{label}[{name!r}]: pseudotensors (parity='odd') are not "
                f"supported; GLOBE's features and vector outputs are built from "
                f"true vectors, so a pseudotensor would transform with the wrong "
                f"sign under reflections"
            )


class Kernel(Module):
    r"""A kernel function for evaluating scalar and vector fields from source points.

    This class implements a learnable neural-network-based kernel function that
    computes scalar and vector fields at target points based on the influence of
    source points with associated normals and strengths. The kernel uses a Pade
    rational neural network to model the field interactions while preserving
    physical properties such as proper far-field decay rates, translational
    invariance, rotational invariance, parity invariance, and scale invariance.

    The kernel takes as input the relative positions, orientations, and magnitudes
    of source points, then outputs field values that are consistent with physical
    conservation laws. For vector fields, the output is automatically reprojected
    onto a local coordinate system to maintain rotational invariance.

    Parameters
    ----------
    n_spatial_dims : int
        Number of spatial dimensions (2 or 3).
    output_schema : FieldSchemaLike
        Field schema declaring the output fields, e.g.
        ``{"pressure": {"rank": 0}, "velocity": {"rank": 1}}`` (see
        :class:`~physicsnemo.mesh.FieldSchema`). Nesting is supported and
        mirrors the desired output structure. Only true scalars (rank 0) and
        vectors (rank 1) are implemented, here and in the other schemas;
        anything else raises :class:`NotImplementedError`. Derive from data
        via :meth:`FieldSchema.from_tensordict`.
    source_schema : FieldSchemaLike or None, optional
        Field schema describing per-source features. The number of rank-0
        fields determines scalar input width; rank-1 fields determine vector
        input width.
    global_schema : FieldSchemaLike or None, optional
        Field schema describing global conditioning features.
    smoothing_radius : float, optional, default=1e-8
        Small value used to smooth power functions near zero to avoid numerical
        instabilities.
    hidden_layer_sizes : Sequence[int] or None, optional, default=None
        Sequence of hidden layer sizes for the neural network. When ``None``,
        defaults to ``[64]``.
    n_spherical_harmonics : int, optional, default=4
        Number of spherical harmonic terms to use as features.
    network_type : {"pade", "mlp"}, optional, default="pade"
        Type of neural network to use for the kernel function.
    spectral_norm : bool, optional, default=False
        Whether to apply spectral normalization to network weights.
    use_gradient_checkpointing : bool, optional, default=True
        If ``True``, applies ``torch.utils.checkpoint.checkpoint`` during
        training to trade compute for memory. Disable for small models or
        when profiling.

    Forward
    -------
    reference_length : Float[torch.Tensor, ""]
        Scalar reference length scale used to convert position-based features
        into dimensionless quantities.
    source_points : Float[torch.Tensor, "n_sources n_dims"]
        Physical coordinates of the source points, which are the centers of
        the influence fields. Shape :math:`(N_{sources}, D)`.
    target_points : Float[torch.Tensor, "n_targets n_dims"]
        Physical coordinates of the target points where the field is evaluated.
        Shape :math:`(N_{targets}, D)`.
    source_strengths : Float[torch.Tensor, "n_sources"] or None, optional, default=None
        Scalar strength values associated with each source point. Shape
        :math:`(N_{sources},)`. Defaults to all ones if ``None``.
    source_data : TensorDict or None, optional, default=None
        Per-source features with ``batch_size=(N_sources,)``. Contains a mix
        of scalar (rank-0) and vector (rank-1) tensors; the kernel packs
        them into one tensor before evaluating. Its leaves must be exactly
        the fields of ``source_schema``, under their nested keys (a field
        ``"a.b"`` at ``("a", "b")``). All values must be dimensionless.
    global_data : TensorDict or None, optional, default=None
        Problem-level features with ``batch_size=()``. Contains a mix of
        scalar (rank-0) and vector (rank-1) tensors, packed the same way.
        Its leaves must be exactly the fields of ``global_schema``. All
        values must be dimensionless.

    Outputs
    -------
    TensorDict[str, Float[torch.Tensor, "n_targets ..."]]
        TensorDict with batch_size :math:`(N_{targets},)` containing the computed
        fields. Each scalar field has shape :math:`(N_{targets},)` and each vector
        field has shape :math:`(N_{targets}, D)`.

    Notes
    -----
    Internally the kernel works on packed tensors, not on TensorDicts: the
    scalar and vector features each sit in one tensor, and the output fields
    in one ``(..., n_output_features)`` tensor whose columns
    :attr:`output_layout` names. Features keep the column order the kernels
    have always used, so trained checkpoints stay valid.
    """

    def __init__(
        self,
        *,
        n_spatial_dims: int,
        output_schema: FieldSchemaLike,
        source_schema: FieldSchemaLike | None = None,
        global_schema: FieldSchemaLike | None = None,
        smoothing_radius: float = 1e-8,
        hidden_layer_sizes: Sequence[int] | None = None,
        n_spherical_harmonics: int = 4,
        network_type: Literal["pade", "mlp"] = "pade",
        spectral_norm: bool = False,
        use_gradient_checkpointing: bool = True,
        self_regularization_beta: float | None = None,
    ):
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [64]

        super().__init__()

        self.n_spatial_dims = n_spatial_dims
        self.output_schema = FieldSchema.parse(output_schema, label="output_schema")
        self.source_schema = FieldSchema.parse(
            {} if source_schema is None else source_schema, label="source_schema"
        )
        self.global_schema = FieldSchema.parse(
            {} if global_schema is None else global_schema, label="global_schema"
        )
        for label, schema in (
            ("output_schema", self.output_schema),
            ("source_schema", self.source_schema),
            ("global_schema", self.global_schema),
        ):
            _require_supported(schema, label=label)
        self.smoothing_radius = smoothing_radius
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_spherical_harmonics = n_spherical_harmonics
        self.use_gradient_checkpointing = use_gradient_checkpointing

        ### Pre-squared smoothing radius as a registered tensor buffer.
        ### A buffer (rather than a Python float) is required because
        ### ``_evaluate_packed`` adds this scalar to a tensor of squared
        ### magnitudes inside a checkpoint sub-graph; Python free variables cannot
        ### be lifted across nested Dynamo SubgraphTracers (the lift
        ### chain bottoms out at the root tracer with no parent and
        ### asserts ``lift_tracked_freevar_to_input should not be
        ### called on root SubgraphTracer``).  As a buffer, the
        ### tensor is a tracked module attribute that Dynamo treats as
        ### a graph leaf, so no lift is needed.
        self.register_buffer(
            "_smoothing_radius_sq",
            torch.tensor(smoothing_radius**2, dtype=torch.float32),
            persistent=False,
        )

        in_features = self.network_in_features
        hidden_features = list(self.hidden_layer_sizes)
        out_features = self.network_out_features

        if network_type == "pade":
            self.network = Pade(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
                spectral_norm=spectral_norm,
                numerator_order=2,
                denominator_order=2,
                use_separate_mlps=False,
                share_denominator_across_channels=False,
                self_regularization_beta=self_regularization_beta,
            )
        elif network_type == "mlp":
            self.network = nn.Sequential(
                Mlp(
                    in_features=in_features,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    spectral_norm=spectral_norm,
                    act_layer=nn.SiLU(),
                    final_dropout=False,
                ),
                nn.Tanh(),
            )
        else:
            raise ValueError(
                f"Invalid network type: {network_type=!r}; must be one of ['pade', 'mlp']"
            )

        ### Static description of the packed features (see
        ### ``_evaluate_packed``), built here so that compiled code only
        ### reads it.
        self._source_layout = _input_layout(
            self.source_schema, n_spatial_dims, sort_dotted_names=True
        )
        self._global_layout = _input_layout(
            self.global_schema, n_spatial_dims, sort_dotted_names=False
        )
        self._n_source_scalars = self.source_schema.count(0)
        self._n_source_vectors = self.source_schema.count(1)
        self._n_global_scalars = self.global_schema.count(0)
        self._n_global_vectors = self.global_schema.count(1)
        ### Pairs of input vectors, in the vector order of ``_evaluate_packed``.
        n_vectors_in = 1 + self._n_source_vectors + self._n_global_vectors
        ### A buffer, so that gathering the pairs needs no host-to-device
        ### index upload, and ``index_select`` rather than per-pair
        ### ``select`` views, whose backward allocates a full-size zero
        ### tensor per pair.
        pairs = list(itertools.combinations(range(n_vectors_in), 2))
        self.register_buffer(
            "_keypair_index",
            torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous(),
            persistent=False,
        )
        ### The spherical-harmonic feature groups were once keyed
        ### ``pairwise_spherical_harmonics_{i}`` and sorted as strings.
        self._harmonic_order: tuple[int, ...] = tuple(
            sorted(
                range(n_spherical_harmonics),
                key=lambda i: f"pairwise_spherical_harmonics_{i}",
            )
        )
        self.output_layout: PackLayout = _output_layout(
            self.output_schema, n_spatial_dims
        )
        """Columns of the packed output: the output fields in sorted name
        order, one column per scalar and ``n_spatial_dims`` per vector, keyed
        by their dotted names.  ``output_layout.unpack(packed)`` gives the
        output TensorDict."""

        ### Fill the lazily computed size now.  A first access inside a
        ### compiled forward adds it to the instance ``__dict__``; the next
        ### call then fails Dynamo's guard on that ``__dict__`` and recompiles
        ### every caller (3-5 minutes per DrivAerML training run on GB300).
        _ = self._floats_per_interaction

    @cached_property
    def _floats_per_interaction(self) -> int:
        r"""Identifiable float allocations per (target, source) interaction.

        Counts tensor elements from feature engineering, MLP evaluation,
        and post-processing that coexist at peak during ``Kernel.forward``.
        Used by :class:`BarnesHutKernel` to estimate chunk memory budgets.

        This is a lower bound - the actual peak is higher due to autograd
        saving input tensors for backward through each element-wise
        operation.  The caller applies a runtime multiplier to account for
        this (see ``BarnesHutKernel._auto_chunk_count``).

        Returns
        -------
        int
            The number of floats per interaction.
        """
        n_vec = 1 + self.source_schema.count(1) + self.global_schema.count(1)
        n_pairs = comb(n_vec, 2)

        return (
            ### Feature engineering: spatial vectors (n_targets, n_sources, 3, ...)
            3                                                  # r = target - source
            + 3 * n_vec * 2                                    # vectors + unit vectors
            ### Feature engineering: scalars (n_targets, n_sources, ...)
            + n_vec * 3                                        # magnitudes: squared, raw, log
            + n_pairs * (1 + 2 * self.n_spherical_harmonics)   # cos_theta + harmonics + products
            + self.network_in_features                         # concatenated MLP input
            ### MLP layers (sequential; peak is largest layer plus I/O)
            + self.network_in_features
            + sum(self.hidden_layer_sizes)
            + self.network_out_features
            ### Post-processing
            + self.network_out_features                        # reshaped output
            + 1                                                # far-field r_mag_sq
            + self.n_spatial_dims * max(1, 2 * n_vec - 1)      # basis vectors
        )

    @cached_property
    def network_in_features(self) -> int:
        r"""Number of input features for the kernel's internal network.

        Derived from the invariant feature engineering pipeline (Section 3.2.2):

        1. Raw source and global scalars
        2. Smoothed log-magnitudes of all input vectors (relative position ``r``,
           source vectors, global vectors)
        3. Pairwise spherical harmonic features for all :math:`\binom{n}{2}` vector
           pairs, each producing ``n_spherical_harmonics`` Legendre polynomial terms
        """
        n_vectors_in: int = (
            1 + self.source_schema.count(1) + self.global_schema.count(1)
        )  # +1 for r
        n_scalars_in: int = self.source_schema.count(0) + self.global_schema.count(0)
        n_vector_pairs_in: int = comb(n_vectors_in, 2)

        return (
            n_scalars_in + n_vectors_in + n_vector_pairs_in * self.n_spherical_harmonics
        )

    @cached_property
    def network_out_features(self) -> int:
        r"""Number of output features for the kernel's internal network.

        One channel per scalar output field, plus vector reprojection coefficients
        for each vector output field (1 radial + 2 per non-radial input vector).
        """
        n_vectors_in: int = (
            1 + self.source_schema.count(1) + self.global_schema.count(1)
        )  # +1 for r

        return self.output_schema.count(0) + self.output_schema.count(1) * (
            1  # r_hat
            + 2 * (n_vectors_in - 1)  # All non-r vectors
        )

    def _split_source_features(
        self, features: Float[torch.Tensor, "*batch n_features"]
    ) -> tuple[
        Float[torch.Tensor, "*batch n_scalars"],
        Float[torch.Tensor, "*batch n_vectors n_dims"],
    ]:
        r"""Views of packed source features as scalars and vectors.

        Parameters
        ----------
        features : Float[torch.Tensor, "*batch n_features"]
            Packed source features, of shape :math:`(*, N_s + N_v D)`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Scalars of shape :math:`(*, N_s)`, and vectors of shape
            :math:`(*, N_v, D)`.
        """
        n = self._n_source_scalars
        return features[..., :n], features[..., n:].unflatten(
            -1, (self._n_source_vectors, self.n_spatial_dims)
        )

    def _split_global_features(
        self, features: Float[torch.Tensor, " n_features"]
    ) -> tuple[
        Float[torch.Tensor, " n_scalars"], Float[torch.Tensor, "n_vectors n_dims"]
    ]:
        r"""Views of packed global features as scalars and vectors.

        Parameters
        ----------
        features : Float[torch.Tensor, "n_features"]
            Packed global features, of shape :math:`(G_s + G_v D,)`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Scalars of shape :math:`(G_s,)`, and vectors of shape
            :math:`(G_v, D)`.
        """
        n = self._n_global_scalars
        return features[..., :n], features[..., n:].unflatten(
            -1, (self._n_global_vectors, self.n_spatial_dims)
        )

    def forward(
        self,
        *,
        reference_length: Float[torch.Tensor, ""],
        source_points: Float[torch.Tensor, "n_sources n_dims"],
        target_points: Float[torch.Tensor, "n_targets n_dims"],
        source_strengths: Float[torch.Tensor, " n_sources"] | None = None,
        source_data: TensorDict | None = None,
        global_data: TensorDict | None = None,
    ) -> TensorDict[str, Float[torch.Tensor, "n_targets ..."]]:
        r"""Evaluates a field kernel at target points based on source point influences.

        Parameters
        ----------
        reference_length : Float[torch.Tensor, ""]
            Scalar tensor, shape :math:`()`. The reference length scale used
            to convert position-based features into dimensionless quantities.
        source_points : Float[torch.Tensor, "n_sources n_dims"]
            Tensor of shape :math:`(N_{sources}, D)`. The physical coordinates
            of the source points, which are the centers of the influence fields.
        target_points : Float[torch.Tensor, "n_targets n_dims"]
            Tensor of shape :math:`(N_{targets}, D)`. The physical coordinates
            of the target points where the field is evaluated.
        source_strengths : Float[torch.Tensor, "n_sources"] or None, optional
            Tensor of shape :math:`(N_{sources},)`. Scalar strength values
            associated with each source point. Defaults to all ones if ``None``.
        source_data : TensorDict or None, optional
            Per-source features with ``batch_size=(N_sources,)``. Contains a
            mix of scalar (rank-0) and vector (rank-1) tensors, exactly the
            fields of ``source_schema``. All values must be dimensionless.
            ``None`` (the default) indicates no per-source features.
        global_data : TensorDict or None, optional
            Problem-level features with ``batch_size=()``. Contains a mix of
            scalar (rank-0) and vector (rank-1) tensors, exactly the fields
            of ``global_schema``. All values must be dimensionless.
            ``None`` (the default) indicates no global conditioning.

        Returns
        -------
        TensorDict[str, Float[torch.Tensor, "n_targets ..."]]
            TensorDict with batch_size :math:`(N_{targets},)` containing the computed
            fields. Each scalar field has shape :math:`(N_{targets},)` and each vector
            field has shape :math:`(N_{targets}, D)`.
        """
        n_sources: int = len(source_points)
        n_targets: int = len(target_points)
        device = source_points.device

        ### Set defaults
        if source_strengths is None:
            source_strengths = torch.ones(n_sources, device=device)
        if source_data is None:
            source_data = TensorDict({}, batch_size=[n_sources], device=device)
        if global_data is None:
            global_data = TensorDict({}, device=device)

        source_features = _pack_inputs(
            source_data, self._source_layout, device, "source_data"
        )
        global_features = _pack_inputs(
            global_data, self._global_layout, device, "global_data"
        )

        ### Input validation
        # Skip validation when running under torch.compile for performance
        if not torch.compiler.is_compiling():
            if source_points.ndim != 2:
                raise ValueError(
                    f"Expected source_points to be 2-dimensional, "
                    f"got {source_points.ndim}D tensor with shape {source_points.shape}"
                )
            if target_points.ndim != 2:
                raise ValueError(
                    f"Expected target_points to be 2-dimensional, "
                    f"got {target_points.ndim}D tensor with shape {target_points.shape}"
                )
            if source_points.shape[-1] != self.n_spatial_dims:
                raise ValueError(
                    f"Expected source_points last dimension to be {self.n_spatial_dims}, "
                    f"got {source_points.shape[-1]}"
                )
            if target_points.shape[-1] != self.n_spatial_dims:
                raise ValueError(
                    f"Expected target_points last dimension to be {self.n_spatial_dims}, "
                    f"got {target_points.shape[-1]}"
                )

        source_scalars, source_vectors = self._split_source_features(source_features)
        global_scalars, global_vectors = self._split_global_features(global_features)

        ### Every (target, source) pair, as dense ``(n_targets, n_sources)``
        ### interaction dims.  ``r`` is computed before any autocast cast.
        r = (
            target_points[:, None, :]  # (n_targets, 1, n_dims)
            - source_points[None, :, :]  # (1, n_sources, n_dims)
        ) / reference_length  # (n_targets, n_sources, n_dims)
        output = self._evaluate_packed(
            source_scalars=source_scalars.expand(n_targets, *source_scalars.shape),
            source_vectors=source_vectors.expand(n_targets, *source_vectors.shape),
            global_scalars=global_scalars,
            global_vectors=global_vectors,
            r=r,
        )

        ### Aggregate over sources, weighted by source strengths
        return self.output_layout.unpack(
            torch.einsum("tsf,s->tf", output, source_strengths)
        )

    def _evaluate_packed(
        self,
        *,
        source_scalars: Float[torch.Tensor, "*interaction_dims n_source_scalars"],
        source_vectors: Float[
            torch.Tensor, "*interaction_dims n_source_vectors n_spatial_dims"
        ],
        global_scalars: Float[torch.Tensor, " n_global_scalars"],
        global_vectors: Float[torch.Tensor, "n_global_vectors n_spatial_dims"],
        r: Float[torch.Tensor, "*interaction_dims n_spatial_dims"],
    ) -> Float[torch.Tensor, "*interaction_dims n_output_features"]:
        r"""Core kernel computation: feature engineering, network, and post-processing.

        Operates on packed feature tensors with arbitrary leading interaction
        dimensions :math:`(*)`. Both ``Kernel.forward()`` (with dense
        :math:`(N_{tgt}, N_{src})` interactions) and ``BarnesHutKernel`` (with
        sparse :math:`(N_{pairs},)` interactions) call this method.

        Each feature group is one tensor, so every step below is one tensor
        operation for all of the group's leaves, where a TensorDict of
        features would take one per leaf.  The network's input columns follow
        the order of :func:`_sort_key`: global scalars, source scalars, the
        log-magnitudes of the vectors (global vectors, ``r``, source vectors),
        then the pairwise spherical-harmonic features.

        Parameters
        ----------
        source_scalars : Float[torch.Tensor, "*interaction_dims n_source_scalars"]
            Per-interaction source scalars in :func:`_sort_key` order, of
            shape :math:`(*, N_s)`.
        source_vectors : Float[torch.Tensor, "*interaction_dims n_source_vectors n_spatial_dims"]
            Per-interaction source vectors in :func:`_sort_key` order, of
            shape :math:`(*, N_v, D)`.
        global_scalars : Float[torch.Tensor, "n_global_scalars"]
            Global scalars, broadcast to every interaction, of shape
            :math:`(G_s,)`.
        global_vectors : Float[torch.Tensor, "n_global_vectors n_spatial_dims"]
            Global vectors, broadcast to every interaction, of shape
            :math:`(G_v, D)`.
        r : Float[torch.Tensor, "*interaction_dims n_spatial_dims"]
            Target-minus-source displacement divided by the reference length,
            of shape :math:`(*, D)`.  All inputs must be dimensionless.

        Returns
        -------
        Float[torch.Tensor, "*interaction_dims n_output_features"]
            Per-interaction output fields of shape :math:`(*, F_{out})`,
            packed as :attr:`output_layout` describes, NOT aggregated over
            sources.
        """
        # Cast to autocast dtype after the fp32-critical r computation
        if torch.is_autocast_enabled(r.device.type):
            dtype = torch.get_autocast_dtype(r.device.type)
            source_scalars = source_scalars.to(dtype)
            source_vectors = source_vectors.to(dtype)
            global_scalars = global_scalars.to(dtype)
            global_vectors = global_vectors.to(dtype)
            r = r.to(dtype)

        interaction_dims = r.shape[:-1]
        n_global_vectors = global_vectors.shape[-2]
        k1, k2 = self._keypair_index

        ### Vector magnitude, direction, and log-magnitude features
        with record_function("kernel::feature_engineering"):
            # All input vectors, in the order [global..., source..., r]
            # EVERY VECTOR HERE SHOULD BE PHYSICALLY UNITLESS to preserve units-invariance.
            vectors = torch.cat(
                [
                    global_vectors.expand(*interaction_dims, *global_vectors.shape),
                    source_vectors,
                    r.unsqueeze(-2),
                ],
                dim=-2,
            )  # (*interaction_dims, n_vectors_in, n_spatial_dims)
            vectors_mag_squared = (vectors * vectors).sum(dim=-1) + (
                self._smoothing_radius_sq
            )
            vectors_mag = vectors_mag_squared.sqrt()
            vectors_hat = vectors / vectors_mag.unsqueeze(-1)
            vectors_log_mag = smooth_log(vectors_mag)

            # TODO in 3D, add cross products of pairs of vectors as input features

            ### Pairwise spherical harmonic features from vector pairs
            cos_theta_pairs = torch.sum(
                vectors_hat.index_select(-2, k1) * vectors_hat.index_select(-2, k2),
                dim=-1,
            )  # (*interaction_dims, n_keypairs)

            # [1:] skips P_0(x) = 1 (constant), which carries no angular information
            spherical_harmonics: list[torch.Tensor] = legendre_polynomials(
                x=cos_theta_pairs, n=self.n_spherical_harmonics + 1
            )[1:]
            pair_log_mag = smooth_log(
                vectors_mag.index_select(-1, k1) * vectors_mag.index_select(-1, k2)
            )

            cat_input_tensors = torch.cat(
                [
                    global_scalars.expand(*interaction_dims, *global_scalars.shape),
                    source_scalars,
                    vectors_log_mag[..., :n_global_vectors],
                    vectors_log_mag[..., -1:],  # r
                    vectors_log_mag[..., n_global_vectors:-1],
                    *(pair_log_mag * spherical_harmonics[i] for i in self._harmonic_order),
                ],
                dim=-1,
            )  # (*interaction_dims, self.network_in_features)

        ### Validate and evaluate the neural network
        if not torch.compiler.is_compiling():
            if not cat_input_tensors.shape[-1] == self.network_in_features:
                raise RuntimeError(
                    f"The input tensor has {cat_input_tensors.shape[-1]=!r} features, but the network expects {self.network_in_features=!r} input features.\n"
                    f"This is due to a shape inconsistency between the `network_in_features` and `forward` methods of the {self.__class__.__name__!r} class."
                )

        flattened_input = cat_input_tensors.reshape(
            prod(interaction_dims), self.network_in_features
        )

        with record_function("kernel::network"):
            flattened_output = self.network(flattened_input)

        output = flattened_output.reshape(*interaction_dims, self.network_out_features)

        ### Far-field decay envelope and vector reprojection
        with record_function("kernel::postprocess"):
            r_mag_sq = vectors_mag_squared[..., -1:]
            output = output * (
                -torch.expm1(-r_mag_sq)
            )  # Lamb-Oseen vortex kernel, numerically stable via expm1
            if self.n_spatial_dims == 2:
                output = output / (r_mag_sq + 1).sqrt()
            elif self.n_spatial_dims == 3:
                output = output / (r_mag_sq + 1)
            else:
                output = output / (r_mag_sq + 1) ** ((self.n_spatial_dims - 1) / 2)

            ### Build the packed per-field outputs in one pass over the
            ### network output; the vector basis is built only when needed.
            basis_vectors: torch.Tensor | None = None
            fields: list[torch.Tensor] = []
            offset = 0
            for shape in self.output_layout.shapes:
                if not shape:
                    fields.append(output[..., offset : offset + 1])
                    offset += 1
                    continue
                if basis_vectors is None:
                    basis_vectors = self._vector_basis(vectors_hat, vectors_log_mag)
                n_coeffs = basis_vectors.shape[-1]
                coeffs = output[..., offset : offset + n_coeffs]
                fields.append(torch.sum(basis_vectors * coeffs.unsqueeze(-2), dim=-1))
                offset += n_coeffs

        return torch.cat(fields, dim=-1)

    def _vector_basis(
        self,
        vectors_hat: Float[torch.Tensor, "*interaction_dims n_vectors_in n_dims"],
        vectors_log_mag: Float[torch.Tensor, "*interaction_dims n_vectors_in"],
    ) -> Float[torch.Tensor, "*interaction_dims n_dims n_basis"]:
        r"""Local rotationally-equivariant basis for the vector outputs.

        Helmholtz-like decomposition: each vector field is expressed in a
        local basis derived from the input vectors: ``r_hat``, then for each
        other input vector (in the order of ``_evaluate_packed``) its
        direction and its dipole direction around ``r_hat``, both scaled by
        its log-magnitude.

        Parameters
        ----------
        vectors_hat : Float[torch.Tensor, "*interaction_dims n_vectors_in n_dims"]
            Unit directions of the :math:`V` input vectors, ``r_hat`` last,
            of shape :math:`(*, V, D)`.
        vectors_log_mag : Float[torch.Tensor, "*interaction_dims n_vectors_in"]
            Log-magnitudes of the input vectors, of shape :math:`(*, V)`.

        Returns
        -------
        Float[torch.Tensor, "*interaction_dims n_dims n_basis"]
            The basis vectors as columns, of shape :math:`(*, D, 2V - 1)`.
        """
        r_hat = vectors_hat[..., -1, :]
        n_hats = vectors_hat[..., :-1, :]
        n_other = n_hats.shape[-2]
        if n_other and self.n_spatial_dims not in (2, 3):
            raise NotImplementedError(
                f"The {self.__class__.__name__!r} class does not support {self.n_spatial_dims=!r}-dimensional problems."
            )
        scales = vectors_log_mag[..., :-1, None]
        scaled_directions = scales * n_hats
        scaled_dipoles = scales * _dipole_directions(r_hat, n_hats)
        components = [r_hat]
        for i in range(n_other):
            components.append(scaled_directions[..., i, :])
            components.append(scaled_dipoles[..., i, :])
        return torch.stack(components, dim=-1)


class BarnesHutKernel(Kernel):
    r"""Tree-accelerated kernel evaluation via Barnes-Hut monopole approximation.

    Reduces the :math:`O(N_{src} \cdot N_{tgt})` cost of the all-to-all kernel
    evaluation to :math:`O((N_{src} + N_{tgt}) \log N_{src})` by building a
    spatial cluster tree over source points and using aggregate (monopole)
    representations for distant clusters.

    For each target point, sources are classified as either:

    - **Near-field**: within the opening-angle threshold, evaluated exactly
      using the underlying :class:`Kernel`'s neural network.
    - **Far-field**: beyond the threshold, approximated by evaluating the
      same network with the cluster's area-weighted centroid, average normal,
      and average features as a "virtual source."

    The interactions fall into four phases (near, far, near-far, far-near).
    Each phase gathers its pairs from the packed features and evaluates them
    with :meth:`Kernel._evaluate_packed`, in chunks for the near and near-far
    phases, and scatters the results into one packed output buffer.

    The ``ClusterTree`` spatial structure can be precomputed per mesh geometry
    and reused across kernel branches and hyperlayers. The
    ``DualInteractionPlan`` can be cached when targets equal sources
    (communication hyperlayers).

    Parameters
    ----------
    Inherits all parameters from :class:`Kernel`.

    leaf_size : int, optional, default=1
        Maximum sources per tree leaf node. Larger values produce shallower
        trees (fewer traversal iterations) at the cost of more exact
        interactions per leaf.

    Forward
    -------
    Same parameters as :class:`Kernel`, with additions:

    theta : float, optional, default=1.5
        Barnes-Hut opening angle.  A node is approximated when
        ``D/r < theta``.  Larger values are more aggressive (more
        approximation, faster).  At ``theta = 0``, all interactions
        are exact.
    cluster_tree : ClusterTree or None, optional, default=None
        Precomputed spatial tree over source points. If ``None``, built
        from ``source_points`` on each call.
    target_tree : ClusterTree or None, optional, default=None
        Precomputed spatial tree over target points, or a forest from
        :meth:`ClusterTree.merge`. If ``None``, built from ``target_points``
        when needed: to plan, or for target-node centroids. For
        self-interaction, pass the same tree as ``cluster_tree``. For any
        other tree or forest, the target-node centroids are plain means of
        the nodes' points unless ``target_centroids`` is given.
    dual_plan : DualInteractionPlan or None, optional, default=None
        Precomputed dual traversal plan. If ``None``, computed from the
        trees and target points on each call.
    source_areas : Float[torch.Tensor, "n_sources"] or None, optional, default=None
        Per-source areas for aggregate weighting. Defaults to ones.
    target_centroids : Float[torch.Tensor, "n_target_nodes n_dims"] or None, optional, default=None
        Centroids of the target nodes that ``dual_plan`` refers to. If
        ``None``, the source tree's area-weighted centroids are used for
        self-interaction, and otherwise the plain mean of each target
        node's points.
    near_chunk_count : int or None, optional, default=None
        Number of chunks for the near-field phase, at least 1. If ``None``,
        chosen by :meth:`_auto_chunk_count`.
    expand_far_targets : bool, optional, default=False
        If ``True``, far-field node pairs are expanded to individual target
        points during plan construction.

    Outputs
    -------
    TensorDict[str, Float[torch.Tensor, "n_targets ..."]]
        Approximate kernel output, converging to the exact result as
        ``theta`` approaches zero.
    """

    def __init__(
        self,
        *,
        n_spatial_dims: int,
        output_schema: FieldSchemaLike,
        source_schema: FieldSchemaLike | None = None,
        global_schema: FieldSchemaLike | None = None,
        smoothing_radius: float = 1e-8,
        hidden_layer_sizes: Sequence[int] | None = None,
        n_spherical_harmonics: int = 4,
        network_type: Literal["pade", "mlp"] = "pade",
        spectral_norm: bool = False,
        use_gradient_checkpointing: bool = True,
        leaf_size: int = 1,
        self_regularization_beta: float | None = None,
    ):
        super().__init__(
            n_spatial_dims=n_spatial_dims,
            output_schema=output_schema,
            source_schema=source_schema,
            global_schema=global_schema,
            smoothing_radius=smoothing_radius,
            hidden_layer_sizes=hidden_layer_sizes,
            n_spherical_harmonics=n_spherical_harmonics,
            network_type=network_type,
            spectral_norm=spectral_norm,
            use_gradient_checkpointing=use_gradient_checkpointing,
            self_regularization_beta=self_regularization_beta,
        )
        self.leaf_size = leaf_size

    def forward(
        self,
        *,
        reference_length: Float[torch.Tensor, ""],
        source_points: Float[torch.Tensor, "n_sources n_dims"],
        target_points: Float[torch.Tensor, "n_targets n_dims"],
        source_strengths: Float[torch.Tensor, " n_sources"] | None = None,
        source_data: TensorDict | None = None,
        global_data: TensorDict | None = None,
        theta: float = 1.5,
        cluster_tree: "ClusterTree | None" = None,
        target_tree: "ClusterTree | None" = None,
        dual_plan: "DualInteractionPlan | None" = None,
        source_areas: Float[torch.Tensor, " n_sources"] | None = None,
        target_centroids: Float[torch.Tensor, "n_target_nodes n_dims"] | None = None,
        near_chunk_count: int | None = None,
        expand_far_targets: bool = False,
    ) -> TensorDict[str, Float[torch.Tensor, "n_targets ..."]]:
        r"""Evaluate the kernel with dual-tree Barnes-Hut acceleration.

        Uses four evaluation phases (see :meth:`_evaluate_plan`):

        - **Phase A (near-field)**: individual target-source pairs from
          nearby leaf nodes, evaluated exactly with chunked processing.
        - **Phase B (far-field node pairs)**: the kernel is evaluated ONCE
          at ``(centroid_T, centroid_S, avg_data_S)`` per well-separated
          node pair, then broadcast to all individual targets in the
          target node via scatter_add.
        - **Phase C (near,far)**: individual targets against source-node
          aggregates, chunked like Phase A.
        - **Phase D (far,near)**: target-node centroids against individual
          sources, broadcast like Phase B.

        Parameters
        ----------
        reference_length : Float[torch.Tensor, ""]
            Reference length scale for nondimensionalization.
        source_points : Float[torch.Tensor, "n_sources n_dims"]
            Source point coordinates.
        target_points : Float[torch.Tensor, "n_targets n_dims"]
            Target point coordinates.
        source_strengths : Float[torch.Tensor, "n_sources"] or None
            Per-source strength weights. Defaults to ones.
        source_data : TensorDict or None
            Per-source features (normals, latents).
        global_data : TensorDict or None
            Problem-level conditioning features.
        theta : float
            Barnes-Hut opening angle (larger = more aggressive).
        cluster_tree : ClusterTree or None
            Precomputed source tree. Built on-the-fly if ``None``.
        target_tree : ClusterTree or None
            Precomputed target tree, or a forest from :meth:`ClusterTree.merge`,
            built on-the-fly if ``None`` and needed (to plan, or for
            target-node centroids). For self-interaction (comm layers), pass
            the same tree as ``cluster_tree``. For any other tree or forest,
            the target-node centroids are plain means of the nodes' points
            unless ``target_centroids`` is given.
        dual_plan : DualInteractionPlan or None
            Precomputed dual traversal plan. Computed on-the-fly if ``None``.
        source_areas : Float[torch.Tensor, "n_sources"] or None
            Per-source areas for aggregate weighting. Defaults to ones.
        target_centroids : Float[torch.Tensor, "n_target_nodes n_dims"] or None
            Per-node centroids for the target tree. If ``None`` and
            ``target_tree is cluster_tree`` (self-interaction), source
            aggregates' centroids are reused. Otherwise computed from
            the target tree.
        near_chunk_count : int or None
            Number of chunks for near-field pair processing, at least 1.
            This is a count of chunks, not a chunk size. A count above the
            number of near pairs only adds empty chunks. When provided,
            overrides :meth:`_auto_chunk_count`, e.g. to fix the chunking
            from an outer scope.
        expand_far_targets : bool, optional, default=False
            If ``True``, far-field node pairs are expanded to individual
            target points during plan construction, eliminating the
            target-side centroid broadcast.  Passed through to
            :meth:`ClusterTree.find_dual_interaction_pairs`.

        Returns
        -------
        TensorDict[str, Float[torch.Tensor, "n_targets ..."]]
            Kernel output fields at target points.

        Raises
        ------
        ValueError
            If ``near_chunk_count`` is given and below 1.
        """
        ### A plain int, so this check also runs under torch.compile.
        if near_chunk_count is not None and near_chunk_count < 1:
            raise ValueError(
                f"near_chunk_count must be at least 1, got {near_chunk_count}"
            )

        if source_strengths is None:
            source_strengths = torch.ones(
                source_points.shape[0], device=source_points.device
            )
        packed = self._evaluate_plan(
            reference_length=reference_length,
            source_points=source_points,
            target_points=target_points,
            source_strengths=source_strengths,
            near_chunk_count=near_chunk_count,
            **self._prepare_plan_inputs(
                source_points=source_points,
                target_points=target_points,
                source_data=source_data,
                global_data=global_data,
                theta=theta,
                cluster_tree=cluster_tree,
                target_tree=target_tree,
                dual_plan=dual_plan,
                source_areas=source_areas,
                target_centroids=target_centroids,
                expand_far_targets=expand_far_targets,
            ),
        )
        return self.output_layout.unpack(packed)

    def _prepare_plan_inputs(
        self,
        *,
        source_points: Float[torch.Tensor, "n_sources n_dims"],
        target_points: Float[torch.Tensor, "n_targets n_dims"],
        source_data: TensorDict | None,
        global_data: TensorDict | None,
        theta: float,
        cluster_tree: "ClusterTree | None",
        target_tree: "ClusterTree | None",
        dual_plan: "DualInteractionPlan | None",
        source_areas: Float[torch.Tensor, " n_sources"] | None,
        target_centroids: Float[torch.Tensor, "n_target_nodes n_dims"] | None,
        expand_far_targets: bool,
    ) -> dict[str, object]:
        r"""Trees, plan, packed features and aggregates for :meth:`_evaluate_plan`.

        This is the setup of :meth:`forward`, shared with
        :meth:`MultiscaleKernel.forward_packed`: it fills in the defaults,
        builds what is not precomputed, packs the features once, and
        aggregates them over the source tree.

        Parameters
        ----------
        source_points : Float[torch.Tensor, "n_sources n_dims"]
            Source coordinates, of shape :math:`(N_{sources}, D)`.
        target_points : Float[torch.Tensor, "n_targets n_dims"]
            Target coordinates, of shape :math:`(N_{targets}, D)`.
        source_data : TensorDict or None
            Per-source features; none if ``None``.
        global_data : TensorDict or None
            Problem-level features; none if ``None``.
        theta : float
            Barnes-Hut opening angle, used only to plan.
        cluster_tree : ClusterTree or None
            Source tree, built if ``None``.
        target_tree : ClusterTree or None
            Target tree or forest, built if ``None`` and needed.
        dual_plan : DualInteractionPlan or None
            Interaction plan, computed if ``None``.
        source_areas : Float[torch.Tensor, "n_sources"] or None
            Per-source areas for aggregate weighting. Defaults to ones.
        target_centroids : Float[torch.Tensor, "n_target_nodes n_dims"] or None
            Target-node centroids, derived if ``None`` and the plan needs
            them (see :meth:`forward`).
        expand_far_targets : bool
            Passed to :meth:`ClusterTree.find_dual_interaction_pairs`.

        Returns
        -------
        dict[str, object]
            The ``source_features``, ``global_features``, ``dual_plan``,
            ``cluster_tree``, ``node_centroids``, ``node_features`` and
            ``target_centroids`` arguments of :meth:`_evaluate_plan`.
        """
        from physicsnemo.mesh.spatial.cluster_tree import ClusterTree

        n_sources = source_points.shape[0]
        device = source_points.device

        ### Set defaults
        if source_data is None:
            source_data = TensorDict({}, batch_size=[n_sources], device=device)
        if global_data is None:
            global_data = TensorDict({}, device=device)
        if source_areas is None:
            source_areas = torch.ones(n_sources, device=device)

        ### Build trees and the plan if not precomputed
        with record_function("bh_kernel::build_tree"):
            if cluster_tree is None:
                cluster_tree = ClusterTree.from_points(
                    source_points, leaf_size=self.leaf_size, areas=source_areas
                )
            if dual_plan is None:
                if target_tree is None:
                    target_tree = ClusterTree.from_points(
                        target_points, leaf_size=self.leaf_size,
                    )
                dual_plan = cluster_tree.find_dual_interaction_pairs(
                    target_tree=target_tree, theta=theta,
                    expand_far_targets=expand_far_targets, validate=False,
                )

        ### Pack the features once; every phase gathers from these tensors.
        with record_function("bh_kernel::prepare_data"):
            source_features = _pack_inputs(
                source_data, self._source_layout, device, "source_data"
            )
            global_features = _pack_inputs(
                global_data, self._global_layout, device, "global_data"
            )

        ### Source aggregates for far-field clusters, packed like the sources,
        ### and the target-node centroids that only the far-target phases use.
        with record_function("bh_kernel::compute_aggregates"):
            node_centroids, node_features = _aggregate_features(
                cluster_tree, source_points, source_areas, source_features
            )
            if (
                target_centroids is None
                and dual_plan.n_far_nodes + dual_plan.n_fn > 0
            ):
                if target_tree is None:
                    target_tree = ClusterTree.from_points(
                        target_points, leaf_size=self.leaf_size,
                    )
                target_centroids = _target_node_centroids(
                    cluster_tree, target_tree, target_points, node_centroids
                )

        return dict(
            source_features=source_features,
            global_features=global_features,
            dual_plan=dual_plan,
            cluster_tree=cluster_tree,
            node_centroids=node_centroids,
            node_features=node_features,
            target_centroids=target_centroids,
        )

    def _evaluate_plan(
        self,
        *,
        reference_length: Float[torch.Tensor, ""],
        source_points: Float[torch.Tensor, "n_sources n_dims"],
        target_points: Float[torch.Tensor, "n_targets n_dims"],
        source_strengths: Float[torch.Tensor, " n_sources"],
        source_features: Float[torch.Tensor, "n_sources n_source_features"],
        global_features: Float[torch.Tensor, " n_global_features"],
        dual_plan: "DualInteractionPlan",
        cluster_tree: "ClusterTree",
        node_centroids: Float[torch.Tensor, "n_nodes n_dims"],
        node_features: Float[torch.Tensor, "n_nodes n_source_features"],
        target_centroids: Float[torch.Tensor, "n_target_nodes n_dims"] | None,
        near_chunk_count: int | None = None,
    ) -> Float[torch.Tensor, "n_targets n_output_features"]:
        r"""Run the four interaction phases of a dual plan on packed data.

        This is the body of :meth:`forward` after the trees, the plan, the
        aggregates and the packed features exist. :class:`MultiscaleKernel`
        calls it directly, so that all of its branches share one packing and
        one set of aggregates.

        Parameters
        ----------
        reference_length : Float[torch.Tensor, ""]
            Reference length scale for nondimensionalization, a 0-d tensor.
        source_points : Float[torch.Tensor, "n_sources n_dims"]
            Source coordinates, of shape :math:`(N_{sources}, D)`.
        target_points : Float[torch.Tensor, "n_targets n_dims"]
            Target coordinates, of shape :math:`(N_{targets}, D)`.
        source_strengths : Float[torch.Tensor, "n_sources"]
            Per-source strength weights, of shape :math:`(N_{sources},)`.
        source_features : Float[torch.Tensor, "n_sources n_source_features"]
            Source features packed by :func:`_pack_inputs`, of shape
            :math:`(N_{sources}, F_{src})`.
        global_features : Float[torch.Tensor, "n_global_features"]
            Global features packed by :func:`_pack_inputs`, of shape
            :math:`(F_{glob},)`.
        dual_plan : DualInteractionPlan
            Interaction plan from ``cluster_tree`` to the targets.
        cluster_tree : ClusterTree
            Source tree, for per-node strength totals.
        node_centroids : Float[torch.Tensor, "n_nodes n_dims"]
            Area-weighted source-node centroids, of shape
            :math:`(N_{nodes}, D)`.
        node_features : Float[torch.Tensor, "n_nodes n_source_features"]
            Area-weighted mean ``source_features`` per source node, of shape
            :math:`(N_{nodes}, F_{src})`.
        target_centroids : Float[torch.Tensor, "n_target_nodes n_dims"] or None
            Centroids of the :math:`M` target nodes, of shape :math:`(M, D)`;
            required if the plan has far-target (``far`` or ``fn``) pairs.
        near_chunk_count : int or None
            Number of chunks for the near-field phase (see :meth:`forward`).

        Returns
        -------
        Float[torch.Tensor, "n_targets n_output_features"]
            Output fields at the targets, of shape
            :math:`(N_{targets}, F_{out})`, packed as :attr:`output_layout`
            describes.
        """
        n_sources = source_points.shape[0]
        n_targets = target_points.shape[0]
        device = source_points.device
        global_scalars, global_vectors = self._split_global_features(global_features)

        with record_function("bh_kernel::compute_strengths"):
            node_total_strength = self._compute_node_strengths(
                cluster_tree, source_strengths
            )

        n_near = dual_plan.n_near
        n_nf = dual_plan.n_nf
        n_fn = dual_plan.n_fn
        n_far_nodes = dual_plan.n_far_nodes

        if not torch.compiler.is_compiling():
            n_dense = n_sources * n_targets
            logger.debug(
                "BarnesHutKernel: %d near + %d nf + %d fn + %d far_node "
                "(%d sources x %d targets = %d dense, %.2f%% near-field)",
                n_near, n_nf, n_fn, n_far_nodes,
                n_sources, n_targets, n_dense,
                100.0 * n_near / max(n_dense, 1),
            )
            if target_centroids is None and n_far_nodes + n_fn > 0:
                raise ValueError(
                    "The plan has far-target pairs, so target_centroids is required."
                )

        ### Packed output buffer.  All four phases scatter into this single
        ### tensor, one ``index_add_`` per phase chunk for all output fields:
        ### ``indexing_backward`` was the top GPU kernel by time in profiling
        ### (~23% of total GPU time) when each output field had its own.
        ###
        ### Buffer dtype must match the dtype of ``weighted = chunk * weights``
        ### that the four phases will scatter in (``index_add_`` requires
        ### exact dtype match).  ``chunk`` comes from the MLP at the
        ### autocast dtype (or ``source_points.dtype`` outside autocast);
        ### ``weights`` carries ``source_strengths.dtype``.  Their product
        ### is the type-promoted dtype.
        chunk_dtype = (
            torch.get_autocast_dtype(device.type)
            if torch.is_autocast_enabled(device.type)
            else source_points.dtype
        )
        buffer_dtype = torch.promote_types(chunk_dtype, source_strengths.dtype)
        packed_buf = torch.zeros(
            (n_targets, self.output_layout.numel), dtype=buffer_dtype, device=device
        )

        # ==================================================================
        # Phase A: Near-field (individual target-source pairs, chunked)
        # ==================================================================
        if n_near > 0:
            near_tgt_ids = dual_plan.near_target_ids
            near_src_ids = dual_plan.near_source_ids
            n_chunks = (
                near_chunk_count
                if near_chunk_count is not None
                else self._auto_chunk_count(n_near, device)
            )
            chunk_size = _ceil_div(n_near, n_chunks)

            ### Loop over the chunk count (see ``_auto_chunk_count``), not
            ### ``range(0, n_near, chunk_size)``: torch.compile specializes
            ### ``range`` arguments, so that form would recompile for every new
            ### pair count (each new geometry).  Only the last chunk is shorter,
            ### and it ends at ``n_near`` itself: a slice past the end makes
            ### torch.compile guard on ``n_near % n_chunks``, and a ``min``
            ### gives each chunk its own symbolic size (slower to compile).
            for i_chunk in range(n_chunks):
                start = i_chunk * chunk_size
                end = n_near if i_chunk == n_chunks - 1 else start + chunk_size

                chunk_tgt_ids = near_tgt_ids[start:end]
                chunk_src_ids = near_src_ids[start:end]

                ### Gather + evaluate inside one checkpoint boundary.
                # By checkpointing a function that takes INDICES (int64,
                # ~8 bytes/pair) and references to the shared packed source
                # data (O(1)), the autograd graph saves only the indices -
                # not the gathered float data (~300 bytes/pair).  This is a
                # ~37x reduction in checkpoint-saved memory per branch.
                with record_function("bh_kernel::near_chunk"):
                    chunk_result = self._maybe_checkpointed_evaluate(
                        chunk_tgt_ids, chunk_src_ids,
                        target_points, source_points, source_features,
                        global_scalars, global_vectors, reference_length,
                    )

                with record_function("bh_kernel::near_scatter"):
                    chunk_strengths = torch.index_select(source_strengths, 0, chunk_src_ids)
                    self._weight_and_scatter(
                        chunk_result, chunk_strengths, chunk_tgt_ids, packed_buf
                    )

        # ==================================================================
        # Phase B: Far-field node pairs (evaluate once, broadcast to targets)
        # ==================================================================
        if n_far_nodes > 0:
            far_tgt_nids = dual_plan.far_target_node_ids
            far_src_nids = dual_plan.far_source_node_ids

            ### Evaluate kernel at (centroid_T, centroid_S, avg_data_S).
            # Same gather-inside-checkpoint pattern: the checkpoint saves
            # only the node ID indices, not the gathered aggregate data.
            with record_function("bh_kernel::far_node_evaluate"):
                far_result = self._maybe_checkpointed_evaluate(
                    far_tgt_nids, far_src_nids,
                    target_centroids, node_centroids, node_features,
                    global_scalars, global_vectors, reference_length,
                )

            ### Broadcast node-level results to individual targets via
            ### ``pair_ids`` (which point back into the per-node result
            ### tensor): one gather + one scatter for all output fields.
            with record_function("bh_kernel::far_node_broadcast"):
                far_strengths = torch.index_select(node_total_strength, 0, far_src_nids)
                self._weight_and_scatter(
                    far_result, far_strengths, dual_plan.far_broadcast_target_ids,
                    packed_buf, broadcast_pair_ids=dual_plan.far_broadcast_pair_ids,
                )

        # ==================================================================
        # Phase C: (near,far) - individual targets × source node centroids
        # ==================================================================
        if n_nf > 0:
            ### Chunked like Phase A: with ``expand_far_targets`` this stream
            ### carries almost every interaction, and evaluating it in one
            ### call materializes every hidden activation for all of it.
            ### It loops over the chunk count for the same reason as Phase A.
            n_nf_chunks = self._auto_chunk_count(n_nf, device)
            nf_chunk = _ceil_div(n_nf, n_nf_chunks)
            for i_chunk in range(n_nf_chunks):
                start = i_chunk * nf_chunk
                end = n_nf if i_chunk == n_nf_chunks - 1 else start + nf_chunk
                nf_tgt_ids = dual_plan.nf_target_ids[start:end]
                nf_src_nids = dual_plan.nf_source_node_ids[start:end]

                ### Same evaluation as Phase B (source centroids + aggregates),
                # but same scatter as Phase A (per-target, no broadcast).
                with record_function("bh_kernel::nf_evaluate"):
                    nf_result = self._maybe_checkpointed_evaluate(
                        nf_tgt_ids, nf_src_nids,
                        target_points, node_centroids, node_features,
                        global_scalars, global_vectors, reference_length,
                    )

                with record_function("bh_kernel::nf_scatter"):
                    nf_strengths = torch.index_select(node_total_strength, 0, nf_src_nids)
                    self._weight_and_scatter(
                        nf_result, nf_strengths, nf_tgt_ids, packed_buf
                    )

        # ==================================================================
        # Phase D: (far,near) - target node centroid × individual sources,
        #          broadcast to stage-1 survivors
        # ==================================================================
        if n_fn > 0:
            fn_tgt_nids = dual_plan.fn_target_node_ids
            fn_src_ids = dual_plan.fn_source_ids

            ### Evaluate K(target_centroid, source_point, source_data).
            # Uses target centroids (like Phase B) but individual source
            # points and data (like Phase A).
            with record_function("bh_kernel::fn_evaluate"):
                fn_result = self._maybe_checkpointed_evaluate(
                    fn_tgt_nids, fn_src_ids,
                    target_centroids, source_points, source_features,
                    global_scalars, global_vectors, reference_length,
                )

            ### Broadcast to stage-1 survivors via the ragged mapping.
            with record_function("bh_kernel::fn_broadcast"):
                fn_strengths = torch.index_select(source_strengths, 0, fn_src_ids)
                self._weight_and_scatter(
                    fn_result, fn_strengths, dual_plan.fn_expanded_target_ids,
                    packed_buf, broadcast_pair_ids=dual_plan.fn_expanded_pair_ids,
                )

        return packed_buf

    def _maybe_checkpointed_evaluate(self, *args: object) -> torch.Tensor:
        r"""Run :meth:`_gather_and_evaluate` with optional gradient checkpointing.

        The four BH-kernel phases (near, far_node, nf, fn) all wrap the
        same ``_gather_and_evaluate`` call in an
        ``if self.use_gradient_checkpointing`` branch with identical
        checkpoint kwargs.  Hoisting the branch keeps the
        gradient-checkpointing policy in one place.

        The wrap fires in eval as well as in training. In eval no autograd
        tape is active (``train.py`` wraps validation in
        ``torch.no_grad()``), so ``checkpoint(use_reentrant=False)``
        degenerates to a near-no-op forward call - no recompute, no
        extra memory, no extra compute.

        Parameters
        ----------
        *args : object
            Positional arguments of :meth:`_gather_and_evaluate`.

        Returns
        -------
        torch.Tensor
            The packed per-pair outputs of :meth:`_gather_and_evaluate`.
        """
        if self.use_gradient_checkpointing:
            return checkpoint(
                self._gather_and_evaluate, *args, use_reentrant=False
            )
        return self._gather_and_evaluate(*args)

    @staticmethod
    def _weight_and_scatter(
        chunk_result: Float[torch.Tensor, "n_pairs n_output_features"],
        weights: Float[torch.Tensor, " n_pairs"],
        tgt_ids: Int[torch.Tensor, " n_pairs_or_expanded"],
        packed_buf: Float[torch.Tensor, "n_targets n_output_features"],
        *,
        broadcast_pair_ids: Int[torch.Tensor, " n_pairs_or_expanded"] | None = None,
    ) -> None:
        r"""Weight one phase's packed outputs and ``index_add_`` them into ``packed_buf``.

        One scatter covers every output field, so the backward of the
        scatter (the largest GPU cost in profiling) runs once per phase
        chunk instead of once per field.

        Parameters
        ----------
        chunk_result : Float[torch.Tensor, "n_pairs n_output_features"]
            Output of one ``_evaluate_packed`` call, of shape
            :math:`(P, F_{out})`.
        weights : Float[torch.Tensor, "n_pairs"]
            Per-pair scalar multipliers (e.g. source strengths), of shape
            :math:`(P,)`.
        tgt_ids : Int[torch.Tensor, "n_pairs_or_expanded"]
            Target index for each scatter contribution, of shape
            :math:`(E,)`.  When ``broadcast_pair_ids`` is ``None``,
            :math:`E = P`; otherwise :math:`E` is the broadcast-expanded
            length and ``broadcast_pair_ids`` indexes back into the
            :math:`P` pairs.
        packed_buf : Float[torch.Tensor, "n_targets n_output_features"]
            Shared output buffer for the four-phase loop, of shape
            :math:`(N_{targets}, F_{out})`, updated in place.
        broadcast_pair_ids : Int[torch.Tensor, "n_pairs_or_expanded"] | None
            Used by Phase B and Phase D, where each evaluated pair is
            broadcast to many targets via a ragged-arange mapping, of shape
            :math:`(E,)`.  When given, weighting is applied first (over the
            :math:`P` pairs) and broadcasting second (lower memory than the
            reverse).
        """
        weighted = chunk_result * weights.unsqueeze(-1)
        if broadcast_pair_ids is not None:
            weighted = torch.index_select(weighted, 0, broadcast_pair_ids)

        ### ``index_add_`` rather than ``scatter_add_`` with broadcasted
        ### indices: equivalent semantics, but ``index_add_`` takes a 1-D
        ### ``index`` and avoids the ``unsqueeze`` + ``expand_as`` overhead
        ### that ``scatter_add_`` requires.  In addition to the direct
        ### speedup, ``index_add_`` has a more compact backward (no index
        ### broadcasting in the saved tensors).
        packed_buf.index_add_(0, tgt_ids, weighted)

    def _compute_node_strengths(
        self,
        tree: "ClusterTree",
        source_strengths: Float[torch.Tensor, " n_sources"],
    ) -> Float[torch.Tensor, " n_nodes"]:
        """Compute total source strength per tree node.

        Each node covers a contiguous range
        ``[node_range_start, node_range_start + node_range_count)`` in
        morton-sorted source order, so the total strength in a node's
        subtree is just a range sum: ``prefix_sum[end] - prefix_sum[start]``.
        This replaces the previous bottom-up Python loop over tree levels
        - which launched a fresh batch of gather + add + scatter kernels at
        every level - with a single ``cumsum`` + gather + subtract.  In
        profiling, the level loop was the largest single contributor to
        ``cluster_tree::bottom_up_propagation`` GPU and CPU time.

        Parameters
        ----------
        tree : ClusterTree
            The spatial cluster tree.
        source_strengths : Float[torch.Tensor, "n_sources"]
            Per-source strength values.

        Returns
        -------
        torch.Tensor
            Total strength per node, shape ``(n_nodes,)``.
        """
        device = source_strengths.device
        n_nodes = tree.n_nodes
        if n_nodes == 0:
            return torch.zeros(0, dtype=source_strengths.dtype, device=device)

        ### Cumsum and range-subtract in fp64 to avoid catastrophic
        ### cancellation when ``cumsum_total >> range_sum`` - the regime
        ### of small leaves in a large tree built over offset coordinates.
        ### See the matching note in :meth:`ClusterTree.compute_source_aggregates`.
        sorted_strengths_64 = source_strengths[tree.sorted_source_order].double()
        ### Pad with a leading zero so that ``cumsum[i]`` is the sum of
        ### sorted_strengths[:i] - both endpoints index identically.
        prefix_sum = torch.nn.functional.pad(
            torch.cumsum(sorted_strengths_64, dim=0), (1, 0)
        )

        starts = tree.node_range_start
        ends = starts + tree.node_range_count
        return (prefix_sum[ends] - prefix_sum[starts]).to(source_strengths.dtype)

    def _gather_and_evaluate(
        self,
        tgt_ids: Int[torch.Tensor, " n_pairs"],
        src_ids: Int[torch.Tensor, " n_pairs"],
        target_positions: Float[torch.Tensor, "n_target_rows n_dims"],
        source_positions: Float[torch.Tensor, "n_source_rows n_dims"],
        source_features: Float[torch.Tensor, "n_source_rows n_source_features"],
        global_scalars: Float[torch.Tensor, " n_global_scalars"],
        global_vectors: Float[torch.Tensor, "n_global_vectors n_dims"],
        reference_length: Float[torch.Tensor, ""],
    ) -> Float[torch.Tensor, "n_pairs n_output_features"]:
        r"""Gather source/target data by index and evaluate interactions.

        This function is the checkpoint boundary for memory-efficient
        training.  By wrapping both the gather (indexing into shared
        source data) and the evaluate (feature engineering + MLP) in one
        checkpointed call, the autograd graph saves only the int64 index
        tensors (~8 bytes/pair) and references to the shared source data
        (O(1)), instead of the gathered float features (~300 bytes/pair).

        The source features arrive packed (:func:`_pack_inputs`), so one
        gather fetches every scalar and vector feature of each pair's
        source, and the gathered rows split into scalars and vectors as
        views.

        Parameters
        ----------
        tgt_ids : Int[torch.Tensor, "n_pairs"]
            Row of ``target_positions`` for each pair, of shape :math:`(P,)`.
        src_ids : Int[torch.Tensor, "n_pairs"]
            Row of ``source_positions`` and ``source_features`` for each pair,
            of shape :math:`(P,)`.
        target_positions : Float[torch.Tensor, "n_target_rows n_dims"]
            Target points or target-node centroids, of shape
            :math:`(N_t, D)`.
        source_positions : Float[torch.Tensor, "n_source_rows n_dims"]
            Source points or source-node centroids, of shape
            :math:`(N_s, D)`.
        source_features : Float[torch.Tensor, "n_source_rows n_source_features"]
            Packed features of the source rows, of shape
            :math:`(N_s, F_{src})`.
        global_scalars : Float[torch.Tensor, "n_global_scalars"]
            Global scalars, of shape :math:`(G_s,)`.
        global_vectors : Float[torch.Tensor, "n_global_vectors n_dims"]
            Global vectors, of shape :math:`(G_v, D)`.
        reference_length : Float[torch.Tensor, ""]
            Reference length scale, a 0-d tensor.

        Returns
        -------
        Float[torch.Tensor, "n_pairs n_output_features"]
            Per-pair packed outputs of :meth:`_evaluate_packed`, of shape
            :math:`(P, F_{out})`.
        """
        ### ``index_select`` rather than ``x[idx]``: same forward, but its
        ### backward is an ``index_add_`` instead of the sort-based
        ### ``index_put_(accumulate=True)`` (``indexing_backward``), which was
        ### the largest GPU kernel of a DrivAerML training step on GB300.
        r = (
            torch.index_select(target_positions, 0, tgt_ids)
            - torch.index_select(source_positions, 0, src_ids)
        ) / reference_length
        source_scalars, source_vectors = self._split_source_features(
            torch.index_select(source_features, 0, src_ids)
        )
        return self._evaluate_packed(
            source_scalars=source_scalars,
            source_vectors=source_vectors,
            global_scalars=global_scalars,
            global_vectors=global_vectors,
            r=r,
        )

    def _auto_chunk_count(self, n_total_pairs: int, device: torch.device) -> int:
        r"""Determine the number of chunks for pair-batched kernel evaluation.

        Sizes chunks to fit within a fixed fraction of total device memory
        from the kernel's per-pair feature-engineering footprint estimate.
        During inference (no grad), the autograd overhead multiplier is
        dropped, allowing larger chunks.

        The count is a power of two.  Under ``torch.compile`` the chunk loop
        is unrolled, so the count is part of the graph and a geometry whose
        count differs recompiles (several minutes at DrivAerML scale).  With
        the doubling loop below, Dynamo's guards on the count become a range
        of pair counts that spans a factor of two, instead of one exact count.

        Uses a static budget derived from
        :func:`_device_total_memory_bytes` (cached) instead of
        ``torch.cuda.mem_get_info``.  ``mem_get_info`` is a synchronizing
        driver query - calling it on every kernel evaluation produced
        ~60 syncs per training step in profiling.  Trading "exactly fits
        in current free memory" for "fits in 25% of total device memory"
        is fine when total_memory is large (e.g. 197 GB GB200) and the
        non-kernel resident state (parameters + activations elsewhere) is
        well under the remaining 75%.

        On CPU the same algorithm runs against system RAM (debug-only;
        production runs on CUDA).

        Parameters
        ----------
        n_total_pairs : int
            Number of interaction pairs to evaluate.
        device : torch.device
            Device that evaluates the pairs.

        Returns
        -------
        int
            The number of chunks, a power of two and at least 1.
        """
        element_bytes = (
            torch.get_autocast_dtype(device.type).itemsize
            if torch.is_autocast_enabled(device.type)
            else 4  # fp32
        )

        autograd_overhead = 5 if torch.is_grad_enabled() else 1
        approx_peak_bytes = (
            n_total_pairs
            * self._floats_per_interaction
            * element_bytes
            * autograd_overhead
        )
        target_bytes = _device_chunk_budget_bytes(device)

        n_chunks = 1
        while n_chunks * target_bytes < approx_peak_bytes:
            n_chunks *= 2

        if not torch.compiler.is_compiling():
            logger.debug(
                "auto_chunk_count: %d pairs -> %d chunks of %d "
                "(%.1f MB est. peak, %.1f MB budget / %.1f MB total %s)",
                n_total_pairs, n_chunks, _ceil_div(n_total_pairs, n_chunks),
                approx_peak_bytes / 1e6,
                target_bytes / 1e6,
                _device_total_memory_bytes(device) / 1e6,
                device.type.upper(),
            )

        return n_chunks


class MultiscaleKernel(Module):
    r"""Multiscale kernel composition that linearly combines kernels at different length scales.

    This class implements the multiscale kernel architecture described in paper Section 3.3.
    Physical systems often exhibit phenomena at multiple characteristic length scales
    (e.g., viscous boundary layer thickness, geometric features, wakes).
    :class:`MultiscaleKernel` creates independent kernel branches for each reference
    length, allowing each to specialize at different spatial scales while sharing the
    same functional form.

    Each kernel branch:

    - Operates at a user-specified reference length (e.g., ``viscous_length``,
      ``chord_length``)
    - Has its own learnable parameters (separate neural network weights)
    - Has a learnable scale adjustment factor (``log_scalefactor``) that fine-tunes its
      effective reference length during training
    - Receives the same inputs but normalizes relative positions by its effective length
    - Has separate per-source, per-branch strength values

    The outputs from all branches are linearly summed, forming a multiscale superposition.
    This enables efficient representation of fields with disparate spatial scales without
    requiring a single network to span the entire range.

    Additionally, log-ratios of all reference length pairs are automatically added as
    global scalar features. This provides scale relationship information and enables the
    model to behave equivariantly under uniform scaling when all nondimensional parameters
    (e.g., Reynolds number) are held constant.

    Parameters
    ----------
    n_spatial_dims : int
        Number of spatial dimensions (2 or 3).
    output_schema : FieldSchemaLike
        Field schema declaring the output fields (see :class:`Kernel`).
    reference_length_names : Sequence[str]
        Sequence of identifiers for reference length scales. Each creates an
        independent kernel branch. Examples: ``["viscous", "geometric"]``.
        Must not be empty: a ``ValueError`` is raised otherwise.
    source_schema : FieldSchemaLike or None, optional
        Field schema for per-source features (see :class:`Kernel`).
    global_schema : FieldSchemaLike or None, optional
        Field schema for global features (see :class:`Kernel`).
        Log-ratios of reference lengths are automatically added as scalar
        entries before passing to each kernel branch.
    smoothing_radius : float, optional, default=1e-8
        Small value for numerical stability in magnitude computations.
    hidden_layer_sizes : Sequence[int] or None, optional, default=None
        Hidden layer sizes for kernel networks.
    n_spherical_harmonics : int, optional, default=4
        Number of Legendre polynomial terms for angle features.
    network_type : {"pade", "mlp"}, optional, default="pade"
        Type of network to use.
    spectral_norm : bool, optional, default=False
        Whether to apply spectral normalization to network weights.
    use_gradient_checkpointing : bool, optional, default=True
        Forwarded to each :class:`Kernel` branch. See
        :class:`Kernel` for details.

    Forward
    -------
    reference_lengths : dict[str, torch.Tensor]
        Mapping of reference length names to scalar tensors.
    source_points : Float[torch.Tensor, "n_sources n_dims"]
        Physical coordinates of the source points. Shape :math:`(N_{sources}, D)`.
    target_points : Float[torch.Tensor, "n_targets n_dims"]
        Physical coordinates of the target points. Shape :math:`(N_{targets}, D)`.
    source_strengths : TensorDict[str, Float[torch.Tensor, " n_sources"]] or None, optional, default=None
        Per-source, per-branch strength values. TensorDict keyed by
        ``reference_length_names``. Defaults to all ones.
    source_data : TensorDict or None, optional, default=None
        Per-source features with ``batch_size=(N_sources,)``. Mixed-rank
        TensorDict passed through to each :class:`BarnesHutKernel` branch.
    global_data : TensorDict or None, optional, default=None
        Problem-level features with ``batch_size=()``. Automatically
        augmented with log-ratios of reference lengths before being passed
        to each kernel branch.
    theta : float, optional, default=1.5
        Barnes-Hut opening angle (larger = more aggressive).
    cluster_tree : ClusterTree or None, optional, default=None
        Pre-built cluster tree for source points.  If ``None``, one is
        built from ``source_points`` using the kernel's ``leaf_size``.
    target_tree : ClusterTree or None, optional, default=None
        Pre-built target tree, or a forest from :meth:`ClusterTree.merge`.
        For self-interaction, pass the same tree as ``cluster_tree``. For
        any other tree or forest, the target-node centroids are plain means
        of the nodes' points unless ``target_centroids`` is given.
    dual_plan : DualInteractionPlan or None, optional, default=None
        Pre-computed dual traversal plan.  If ``None``, computed from trees.
    source_areas : Float[torch.Tensor, " n_sources"] or None, optional, default=None
        Area weight per source, used for cluster aggregation.
    expand_far_targets : bool, optional, default=False
        If ``True``, far-field node pairs are expanded to individual target
        points during plan construction.
    target_centroids : Float[torch.Tensor, "n_target_nodes n_dims"] or None, optional, default=None
        Centroids of the target nodes that ``dual_plan`` refers to. If
        ``None``, derived from ``target_tree`` when the plan needs them.

    Outputs
    -------
    TensorDict[str, Float[torch.Tensor, "n_targets ..."]]
        TensorDict with the summed results from all kernel branches. Each scalar
        field has shape :math:`(N_{targets},)` and each vector field has shape
        :math:`(N_{targets}, D)`.

    Examples
    --------
    >>> kernel = MultiscaleKernel(
    ...     n_spatial_dims=2,
    ...     output_schema={"phi": {"rank": 0}, "u": {"rank": 1}},
    ...     reference_length_names=["viscous_length", "chord_length"],
    ...     source_schema={"normal": {"rank": 1}},
    ...     hidden_layer_sizes=[64, 64],
    ... )
    >>> result = kernel(
    ...     source_points=boundary_face_centers,
    ...     target_points=query_points,
    ...     reference_lengths={"viscous_length": torch.tensor(0.001),
    ...                        "chord_length": torch.tensor(1.0)},
    ...     source_data=TensorDict({"normal": normals}, batch_size=[n_sources]),
    ...     source_strengths=TensorDict({"viscous_length": strengths_v,
    ...                                  "chord_length": strengths_c}, ...),
    ... )
    """

    def __init__(
        self,
        *,
        n_spatial_dims: int,
        output_schema: FieldSchemaLike,
        reference_length_names: Sequence[str],
        source_schema: FieldSchemaLike | None = None,
        global_schema: FieldSchemaLike | None = None,
        smoothing_radius: float = 1e-8,
        hidden_layer_sizes: Sequence[int] | None = None,
        n_spherical_harmonics: int = 4,
        network_type: Literal["pade", "mlp"] = "pade",
        spectral_norm: bool = False,
        use_gradient_checkpointing: bool = True,
        leaf_size: int = 1,
        self_regularization_beta: float | None = None,
    ):
        super().__init__()

        if len(reference_length_names) == 0:
            raise ValueError(
                "reference_length_names is empty; MultiscaleKernel needs at "
                "least one reference length, one per kernel branch"
            )

        self.n_spatial_dims = n_spatial_dims
        self.reference_length_names = reference_length_names
        self.smoothing_radius = smoothing_radius
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_spherical_harmonics = n_spherical_harmonics
        self.network_type = network_type
        self.spectral_norm = spectral_norm
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.leaf_size = leaf_size
        ### Parsed schemas; ``GLOBE`` reads ``source_schema`` to select the
        ### cell_data leaves each kernel consumes.
        self.output_schema = FieldSchema.parse(output_schema, label="output_schema")
        self.source_schema = FieldSchema.parse(
            {} if source_schema is None else source_schema, label="source_schema"
        )
        self.global_schema = FieldSchema.parse(
            {} if global_schema is None else global_schema, label="global_schema"
        )

        ### Augment global_schema with log-ratio entries for each
        # pair of reference lengths. These are rank-0 (scalar) features.
        augmented_global = {
            **self.global_schema,
            "log_reference_length_ratios": {
                f"{k1}_{k2}": {"rank": 0}
                for k1, k2 in itertools.combinations(reference_length_names, 2)
            },
        }

        self.kernels = nn.ModuleDict(
            {
                name: BarnesHutKernel(
                    n_spatial_dims=n_spatial_dims,
                    output_schema=self.output_schema,
                    source_schema=self.source_schema,
                    global_schema=augmented_global,
                    smoothing_radius=smoothing_radius,
                    hidden_layer_sizes=hidden_layer_sizes,
                    n_spherical_harmonics=n_spherical_harmonics,
                    network_type=network_type,
                    spectral_norm=spectral_norm,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    leaf_size=leaf_size,
                    self_regularization_beta=self_regularization_beta,
                )
                for name in reference_length_names
            }
        )

        self.log_scalefactors = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1)) for name in reference_length_names}
        )
        self.output_layout: PackLayout = _output_layout(
            self.output_schema, n_spatial_dims
        )
        """Columns of :meth:`forward_packed`'s result (see
        :attr:`Kernel.output_layout`)."""

    def forward(
        self,
        *,
        reference_lengths: dict[str, Float[torch.Tensor, ""]],
        source_points: Float[torch.Tensor, "n_sources n_dims"],
        target_points: Float[torch.Tensor, "n_targets n_dims"],
        source_strengths: TensorDict[str, Float[torch.Tensor, " n_sources"]]
        | None = None,
        source_data: TensorDict[str, Float[torch.Tensor, "n_sources ..."]]
        | None = None,
        global_data: TensorDict[str, Float[torch.Tensor, "..."]] | None = None,
        theta: float = 1.5,
        cluster_tree: "ClusterTree | None" = None,
        target_tree: "ClusterTree | None" = None,
        dual_plan: "DualInteractionPlan | None" = None,
        source_areas: Float[torch.Tensor, " n_sources"] | None = None,
        expand_far_targets: bool = False,
        target_centroids: Float[torch.Tensor, "n_target_nodes n_dims"] | None = None,
    ) -> TensorDict[str, Float[torch.Tensor, "n_targets ..."]]:
        r"""Evaluates the multiscale kernel by combining results from multiple scales.

        Builds a shared :class:`ClusterTree` and :class:`DualInteractionPlan`
        once, then evaluates each :class:`BarnesHutKernel` branch at its
        respective reference length.

        Parameters
        ----------
        reference_lengths : dict[str, Float[torch.Tensor, ""]]
            Mapping of reference length names to 0-d tensors.
        source_points : Float[torch.Tensor, "n_sources n_dims"]
            Source point coordinates, shape :math:`(N_{sources}, D)`.
        target_points : Float[torch.Tensor, "n_targets n_dims"]
            Target point coordinates, shape :math:`(N_{targets}, D)`.
        source_strengths : TensorDict or None, optional
            Per-source, per-branch strength values. Defaults to all ones.
        source_data : TensorDict or None, optional
            Per-source features with ``batch_size=(N_sources,)``.
        global_data : TensorDict or None, optional
            Problem-level features with ``batch_size=()``.
        theta : float
            Barnes-Hut opening angle (larger = more aggressive).
        cluster_tree : ClusterTree or None, optional
            Precomputed source tree. Built from ``source_points`` if ``None``.
        target_tree : ClusterTree or None, optional
            Precomputed target tree, or a forest from :meth:`ClusterTree.merge`.
            Built from ``target_points`` if ``None`` and needed: to plan, or
            for the target-node centroids of a plan with far-target pairs.
            Unless it is ``cluster_tree`` or ``target_centroids`` is given,
            the target-node centroids are plain means of the nodes' points.
        dual_plan : DualInteractionPlan or None, optional
            Precomputed dual traversal plan. Computed if ``None``.
        source_areas : Float[torch.Tensor, "n_sources"] or None, optional
            Per-source areas for aggregate weighting. Defaults to ones.
        expand_far_targets : bool, optional, default=False
            If ``True``, eliminates target-side centroid broadcast by
            expanding far-field node pairs to individual target points.
            Passed through to
            :meth:`ClusterTree.find_dual_interaction_pairs`.
        target_centroids : Float[torch.Tensor, "n_target_nodes n_dims"] or None, optional
            Centroids of the target nodes that ``dual_plan`` refers to. If
            ``None``, derived from ``target_tree`` (see
            :meth:`BarnesHutKernel.forward`) when the plan needs them.

        Returns
        -------
        TensorDict[str, Float[torch.Tensor, "n_targets ..."]]
            Summed results from all kernel branches.
        """
        return self.output_layout.unpack(
            self.forward_packed(
                reference_lengths=reference_lengths,
                source_points=source_points,
                target_points=target_points,
                source_strengths=source_strengths,
                source_data=source_data,
                global_data=global_data,
                theta=theta,
                cluster_tree=cluster_tree,
                target_tree=target_tree,
                dual_plan=dual_plan,
                source_areas=source_areas,
                expand_far_targets=expand_far_targets,
                target_centroids=target_centroids,
            )
        )

    def forward_packed(
        self,
        *,
        reference_lengths: dict[str, Float[torch.Tensor, ""]],
        source_points: Float[torch.Tensor, "n_sources n_dims"],
        target_points: Float[torch.Tensor, "n_targets n_dims"],
        source_strengths: TensorDict[str, Float[torch.Tensor, " n_sources"]]
        | None = None,
        source_data: TensorDict[str, Float[torch.Tensor, "n_sources ..."]]
        | None = None,
        global_data: TensorDict[str, Float[torch.Tensor, "..."]] | None = None,
        theta: float = 1.5,
        cluster_tree: "ClusterTree | None" = None,
        target_tree: "ClusterTree | None" = None,
        dual_plan: "DualInteractionPlan | None" = None,
        source_areas: Float[torch.Tensor, " n_sources"] | None = None,
        expand_far_targets: bool = False,
        target_centroids: Float[torch.Tensor, "n_target_nodes n_dims"] | None = None,
    ) -> Float[torch.Tensor, "n_targets n_output_features"]:
        r"""Same as :meth:`forward`, but returns the fields packed in one tensor.

        The columns are those of :attr:`output_layout`, so
        ``self.output_layout.unpack(self.forward_packed(...))`` equals
        ``self(...)``. Callers that add up several kernels' outputs, as
        :class:`~physicsnemo.experimental.models.globe.GLOBE` does over its
        boundary-condition types, add one tensor per kernel instead of one
        per field.

        Parameters
        ----------
        reference_lengths : dict[str, Float[torch.Tensor, ""]]
            Mapping of reference length names to 0-d tensors.
        source_points : Float[torch.Tensor, "n_sources n_dims"]
            Source point coordinates, of shape :math:`(N_{sources}, D)`.
        target_points : Float[torch.Tensor, "n_targets n_dims"]
            Target point coordinates, of shape :math:`(N_{targets}, D)`.
        source_strengths : TensorDict or None, optional
            Per-source, per-branch strength values of shape
            :math:`(N_{sources},)`. Defaults to all ones.
        source_data : TensorDict or None, optional
            Per-source features with ``batch_size=(N_sources,)``.
        global_data : TensorDict or None, optional
            Problem-level features with ``batch_size=()``.
        theta : float
            Barnes-Hut opening angle (larger = more aggressive).
        cluster_tree : ClusterTree or None, optional
            Precomputed source tree. Built from ``source_points`` if ``None``.
        target_tree : ClusterTree or None, optional
            Precomputed target tree, or a forest from :meth:`ClusterTree.merge`.
            Built from ``target_points`` if ``None`` and needed: to plan, or
            for the target-node centroids of a plan with far-target pairs.
            Unless it is ``cluster_tree`` or ``target_centroids`` is given,
            the target-node centroids are plain means of the nodes' points.
        dual_plan : DualInteractionPlan or None, optional
            Precomputed dual traversal plan. Computed if ``None``.
        source_areas : Float[torch.Tensor, "n_sources"] or None, optional
            Per-source areas for aggregate weighting, of shape
            :math:`(N_{sources},)`. Defaults to ones.
        expand_far_targets : bool, optional, default=False
            If ``True``, expands far-field node pairs to individual target
            points. Passed through to
            :meth:`ClusterTree.find_dual_interaction_pairs`.
        target_centroids : Float[torch.Tensor, "n_target_nodes n_dims"] or None, optional
            Centroids of the :math:`M` target nodes that ``dual_plan`` refers
            to, of shape :math:`(M, D)`. If ``None``, derived from
            ``target_tree`` when the plan needs them.

        Returns
        -------
        Float[torch.Tensor, "n_targets n_output_features"]
            Results summed over all kernel branches, of shape
            :math:`(N_{targets}, F_{out})`, packed as :attr:`output_layout`
            describes.
        """
        n_sources: int = len(source_points)
        device = source_points.device

        ### Set defaults
        if source_strengths is None:
            source_strengths = TensorDict(
                {
                    name: torch.ones(n_sources, device=device)
                    for name in self.reference_length_names
                },
                batch_size=torch.Size([n_sources]),
                device=device,
            )
        if global_data is None:
            global_data = TensorDict({}, device=device)

        # Skip validation when running under torch.compile for performance
        if not torch.compiler.is_compiling():
            for name, (actual, expected) in {
                "reference_lengths": (
                    set(reference_lengths.keys()),
                    set(self.reference_length_names),
                ),
                "source_strengths": (
                    set(source_strengths.keys()),
                    set(self.reference_length_names),
                ),
            }.items():
                if actual != expected:
                    raise ValueError(
                        f"This kernel was instantiated to expect {expected} {name},\n"
                        f"but the forward-method input gives {actual} {name}."
                    )

        ### Augment global_data with log-ratios of reference lengths.
        log_ratios = TensorDict(
            {
                f"{k1}_{k2}": (
                    reference_lengths[k1] / reference_lengths[k2]
                ).log()
                for k1, k2 in itertools.combinations(
                    self.reference_length_names, 2
                )
            },
            device=device,
        )
        global_data = global_data.copy()
        global_data["log_reference_length_ratios"] = log_ratios

        ### The shared tree, plan, packed features, aggregates and target-node
        ### centroids, built once for all branches.  Every branch has the
        ### same schemas and leaf size, so any branch can build them.
        with record_function("multiscale_kernel::prepare_data"):
            plan_inputs = next(iter(self.kernels.values()))._prepare_plan_inputs(
                source_points=source_points,
                target_points=target_points,
                source_data=source_data,
                global_data=global_data,
                theta=theta,
                cluster_tree=cluster_tree,
                target_tree=target_tree,
                dual_plan=dual_plan,
                source_areas=source_areas,
                target_centroids=target_centroids,
                expand_far_targets=expand_far_targets,
            )
        dual_plan = plan_inputs["dual_plan"]

        ### Precompute near-field chunk counts outside the checkpoint boundary,
        # so that each branch's chunking is a fixed checkpoint input.
        near_chunk_counts: dict[str, int] = {
            name: self.kernels[name]._auto_chunk_count(
                dual_plan.n_near, source_points.device
            )
            for name in self.reference_length_names
        }

        ### Decide whether branch-level checkpointing is worthwhile.
        # Each branch accumulates ~34 bytes/near-pair of autograd state
        # (int64 checkpoint-saved indices + multiply/scatter graph nodes).
        # Branch checkpointing avoids holding all branches' graphs
        # simultaneously, which is essential at large N (800k+ faces)
        # but a pure compute overhead at small N.  Compared against a
        # static fraction of total device memory rather than free memory
        # (mem_get_info is a synchronizing driver query).
        _AUTOGRAD_BYTES_PER_PAIR = 34
        n_branches = len(self.reference_length_names)
        use_branch_ckpt = False
        if self.training and self.use_gradient_checkpointing and n_branches > 1:
            n_total_pairs = dual_plan.n_near + dual_plan.n_nf + dual_plan.n_fn
            per_branch_bytes = n_total_pairs * _AUTOGRAD_BYTES_PER_PAIR
            all_branches_bytes = per_branch_bytes * n_branches
            ### Compares against a fraction of total device memory
            ### (CUDA: VRAM; CPU: RAM).  ``_device_chunk_budget_bytes``
            ### handles both via ``_device_total_memory_bytes``.
            budget_bytes = _device_chunk_budget_bytes(device)
            use_branch_ckpt = all_branches_bytes > budget_bytes

            if not torch.compiler.is_compiling():
                logger.debug(
                    "branch checkpoint: %s (est. %.1f MB/branch, "
                    "%.1f MB all branches, %.1f MB budget, %d branches)",
                    "ENABLED" if use_branch_ckpt else "DISABLED",
                    per_branch_bytes / 1e6,
                    all_branches_bytes / 1e6,
                    budget_bytes / 1e6,
                    n_branches,
                )

        ### Evaluate each branch with the shared tree, plan, and aggregates.
        # When enabled, branch-level checkpointing ensures only ONE branch's
        # autograd graph exists at a time during backward, preventing
        # autograd memory from accumulating across all branches.
        result: torch.Tensor | None = None
        for name in self.reference_length_names:
            with record_function(f"multiscale_kernel::branch/{name}"):
                kernel = self.kernels[name]
                branch_kwargs = dict(
                    reference_length=(
                        reference_lengths[name]
                        * torch.exp(self.log_scalefactors[name])
                    ),
                    source_points=source_points,
                    target_points=target_points,
                    source_strengths=source_strengths[name],
                    near_chunk_count=near_chunk_counts[name],
                    **plan_inputs,
                )
                if use_branch_ckpt:
                    piece = checkpoint(
                        kernel._evaluate_plan, use_reentrant=False, **branch_kwargs
                    )
                else:
                    piece = kernel._evaluate_plan(**branch_kwargs)
                result = piece if result is None else result + piece

        return result
