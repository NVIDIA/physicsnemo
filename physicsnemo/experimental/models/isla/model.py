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

r"""ISLA: Invariant Slice Attention.

An E(d)-equivariant soft-slice transformer (any spatial dimension d >= 2) for
surrogates of steady boundary-value PDEs, on the mesh-native API of the mesh
models (GLOBE's). It takes one case at a time:

- the boundary, a mapping of :class:`~physicsnemo.mesh.mesh.Mesh` objects keyed
  by boundary-condition name, each a ``Mesh[d-1, d]``. Its cells are the tokens,
  with their centroids, normals and effective measures: every cell for a surface
  task, and for an interior task a sample of the full-resolution cells, drawn
  inside the model;
- the problem's data, as *named fields of declared tensor rank*: global inputs in
  a TensorDict (a direction, a diffusivity, a dimensionless group), and per-cell
  boundary inputs in the meshes' ``cell_data``, declared per condition (an inlet's
  prescribed velocity, a loaded face's traction; a no-slip wall declares none);
- an interior task's prediction points, read by their positions alone (the model
  computes their geometry).

It returns a point-cloud ``Mesh`` whose ``point_data`` holds fields of declared
rank: scalars, vectors, rank-2 tensors, up to rank 4, each covariant by its rank.
The name states the design principle. The attention operates only on invariants
of the per-point vector set :math:`\{r_i, n_i, \hat g_1, \dots, \hat g_K, v_{i1},
\dots\}` (scaled position, unit normal, unit global vectors, the token's own
tensor inputs). The frame is re-attached only at the heads, so exact rotation,
reflection and translation covariance is paid for once, at the network's edges,
instead of in every layer.

**Where the covariance is exact.** The network is exactly covariant by
construction. The interior task draws its boundary tokens, lays out its boundary
read's forest and builds its exact geometry's winding-number tree in the
boundary's canonical frame (``canonical_frame``). The full model is therefore
exactly covariant, up to roundoff, when that frame is unique. The draw is not a
continuous function of the coordinates, though. A rotation applied in floating
point moves them by roundoff, which can change the drawn cells of a dense
boundary close to a tie (two nearly equal principal variances, or an axis that
is nearly symmetric), and the outputs then agree only approximately. The frame is
not unique on a boundary with a symmetry that reverses a principal axis (a mirror
plane, a two-fold axis, a centre of symmetry), or with equal spread along two
principal axes. The tied axes then take their direction (equal spreads) or their
sign (a symmetric axis) from the lab axes. On such a boundary a surface task,
which samples no cells and has neither read nor tree, stays exact. An interior
task is exact only with ``n_boundary_tokens`` at least the cell count,
``tree_read_theta=0`` and a closed boundary. Otherwise the drawn tokens, the
read's clusters or, near a hole in an open boundary, the distance's sign depend
on the pose, and the outputs agree only approximately.

**Rank specs.** Inputs and outputs are declared in the field-schema grammar that
every mesh model shares (``physicsnemo.mesh.fields``). A schema maps a field name
to a declaration: a :class:`~physicsnemo.mesh.fields.RankSpec`, or a mapping
``{"rank": r, "symmetric": bool, "parity": "even" | "odd"}`` (``r`` = 0 scalar,
1 vector, 2 rank-2 tensor, ...). A mapping of mappings is a nested group, and a
flat name may contain ``"."`` for the same nesting: ``{"surface": {"pressure":
{"rank": 0}}}`` and ``{"surface.pressure": {"rank": 0}}`` declare the same field.
Each parsed declaration is a :class:`~physicsnemo.mesh.fields.FieldSchema`.
``symmetric`` (rank :math:`\ge 2`) means fully symmetric under index permutation.
``parity="odd"`` (outputs only) declares a pseudotensor, which flips sign under a
reflection (a vorticity); pseudoscalar outputs (rank 0, odd) are not implemented
and are refused. One name declares one field, so several independent
scalars are several rank-0 names. The same grammar serves the constructor's
``output_schema``, each schema of ``boundary_schemas``, and ``global_schema``. The
same dotted names address the fields in the meshes' ``cell_data`` and in
``global_data`` (``td["surface", "pressure"]``).

**Inputs of any rank enter through invariants.** A rank-1 boundary input
:math:`v_i` at a token contributes the seed scalars :math:`v_i \cdot \hat n_i`,
:math:`v_i \cdot \hat g_k`, :math:`|v_i|` and its dot products with the token's
other vector inputs. Inside every slice layer, where the token-to-anchor
direction :math:`\hat\rho_{is}` exists, it also contributes :math:`v_i \cdot
\hat\rho_{is}`, next to :math:`\hat n_i \cdot \hat\rho_{is}` and :math:`\hat g_k
\cdot \hat\rho_{is}`. One shared helper contracts every per-token vector with the
anchor direction; the normal is the first of them. A rank-2 input :math:`T_i`
contributes the seed scalars :math:`a^{\mathsf T} T_i b` over the frame vectors
:math:`a, b \in \{\hat n_i, \hat g_k\}`, :math:`\operatorname{tr} T_i` and
:math:`\|T_i\|_F`, and in the layers :math:`\hat\rho^{\mathsf T} T_i \hat\rho`,
:math:`\hat\rho \cdot T_i \hat n_i` and :math:`\hat\rho \cdot T_i \hat g_k`. The
last two contract the derived true vectors :math:`T_i \hat n_i` and :math:`T_i
\hat g_k` with the anchor direction, through the same helper. The input side has
only true vectors and no cross products, so every input invariant is invariant
under improper rotations as well, and pseudotensor *inputs* are refused. Rank-0
inputs are the scalars themselves. Inputs of rank above 2 are refused, since the
contraction set grows combinatorially.

**Boundary conditions by name.** ``boundary_schemas`` declares the boundary
conditions by name, each with its own per-cell fields (an empty schema for a
condition without any, such as a no-slip wall). A condition's identity is
structural; no tag field names it. Its tokens enter through the condition's own
input layer, whose column on the constant is its learned offset. Its vector and
tensor inputs reach the slice layers through its own linear maps onto shared
channels. These are equivariant, since a linear combination with scalar weights
commutes with every orthogonal map. The slice layers are shared: every
condition's tokens route into the same slices and read the same anchors, which is
how the conditions interact. ``forward`` takes any non-empty subset of the
declared names. The tokens are the given conditions' cells in canonical (sorted)
order, so the order of the input mapping changes nothing. With more than one
declared condition the quadrature is per condition: a token's log weight is
:math:`\log a_i - \log A_k + \beta_k`, with :math:`A_k` the full-resolution
measure of its condition :math:`k` and :math:`\beta_k` a learned log share
(``condition_log_shares.beta``, zero at construction). Each condition then weighs
:math:`e^{\beta_k}` in every measure-weighted read, whatever its size. The
interior task's forest keeps the cells' own measures and builds one set of face
trees per condition, so a node holds cells of one condition, and its read weighs
its cells' measure shifted by that condition's share.

**Outputs of any rank leave through an equivariant basis.** One linear head on
the normalized final token state emits invariant coefficients. A rank-:math:`r`
output is the contraction of those coefficients with tensor products of :math:`r`
vectors from the token's basis. For even parity these are the *true* vectors
:math:`t_1, \dots, t_m`: the global directions, the normal, the radial direction
from the token's soft anchor and the :math:`e_\theta` complements (:math:`m = 2K
+ 3` in every dimension). For odd parity the products have exactly one
*pseudovector* factor. The pseudovectors are the Levi-Civita construction from
the primary true vectors :math:`\hat g_k, \hat n_i, \hat u_i`: their pairwise
cross products in three dimensions (:math:`p = \binom{K+2}{2}`), and their
perpendiculars in two (:math:`p = K + 2`). Odd parity is refused in four or more
dimensions, where a pseudovector needs :math:`d - 1` vectors. The basis also
holds the isotropic tensors of that rank and parity: the identity at rank 2, the
three products of Kronecker deltas at rank 4, and, for odd parity, the
Levi-Civita symbol of rank :math:`d`. A symmetric spec uses the symmetrized basis,
with one coefficient per unordered multi-index. The class docstring gives the
coefficient counts; ranks above 4 are refused. The complements add coefficients
but no rank: :math:`e_\theta(\hat u, \hat a)` is :math:`-\hat a` projected off
:math:`\hat u`, so it lies in the span of :math:`\hat u` and :math:`\hat a`, and
the true vectors span at most :math:`K + 2` directions. With no global direction
in three dimensions, every even-parity vector output therefore lies in the plane
of :math:`\hat n` and :math:`\hat u` (a known limit, ``test_isla_heads.py``).

**The frame.** ISLA has no frame origin: positions enter only as point-to-anchor
differences. The one length scale :math:`L` is the measure-weighted
root-mean-square pairwise distance of every cell of the given boundaries at full
resolution. It is computed before an interior task's token draw, so it is the
same for every draw. It is exact for the piecewise-flat boundary: each cell
enters with its own second moment about its centroid, so a subdivision of the
cells changes nothing. It is a normalized two-point statistic, so it is
invariant to the weights' overall scale and converges under refinement of any
boundary with a finite second moment, fractal ones included. No centroid, no
per-dataset reference length and no external frame appears anywhere in the
forward pass, so the frame holds no information about where the mesher placed
its cells.

**Scales (opt-in).** Next to its rank spec, a field of a condition's schema, of
``global_schema`` or of ``output_schema`` may declare ``units``: exponents of
named scales, such as ``{"L": -1, "E": 1}``. The scales are the built-in length
:math:`L` above and the scales the fields define (``E`` here). A field defines a
scale with ``defines: <name>``; a rank-1 one has the units ``{<name>: 1}``,
declared or not, and no others. Rank-1 boundary fields (pooled) or one rank-1
global define one scale, :math:`S = (\sum a |v|^2 / \sum a)^{1/2}`: the
measure-weighted RMS magnitude over the boundary sources' full-resolution cells,
or a global source's magnitude. With boundary sources, :math:`\hat g = \sum a v /
|\sum a v|` is appended to the global directions (a global source is a direction
already). Both are computed per case in float64 before the draw. An absent or
zero source is refused in eager mode; torch.compile skips the data-dependent
checks. A rank-0 global that defines a scale gives its value and is not appended.
Every declared boundary field and rank-0 global is divided by its units'
:math:`\prod \text{scale}^{\text{exp}}` and by its constant ``scale`` (the typical
size of the dimensionless field, default 1) before it enters the invariants.
Every output with units is the head's dimensionless prediction times both, then
standardized by the constants ``output_stats`` (vector means zero). These keys
are ISLA's own, popped before the shared grammar parses the field;
with none declared the model is the unscaled one. So a similarity of the problem
(lengths times :math:`s`, each defined scale times its own factor, every field
times its units' factor) leaves every internal state unchanged and scales each
output by its units.

**Routing.** Each block routes tokens to slices by bounded cosine logits:
``routing_temperature`` :math:`T` times the cosine between the normalized token
state and a prototype per slice. Every logit lies in :math:`[-T, T]`, so no block
can route every point to one slice however its weights grow. The prototypes are
synthesized per sample (``_RoutingLogits``): learned seeds attend over the token
states with a measure-weighted softmax, and the attended term, centered across
the slices and smoothly bounded to a learned fraction of the seeds' spread, is
added to the seeds. The quadrature measure enters the routing as a log-space
bias, so slice states are measure-weighted (quadrature) means; with no measures
the weights are uniform.

**Surface and interior tasks.** Both tasks read the whole boundary: every cell of
every mesh in ``boundary_meshes`` is a token (for the interior task, a sample of
them). A surface task (``interior=False``) predicts at the centroids of the cells
of the ``prediction_boundaries`` given to ``forward``, the names of the
conditions that carry the outputs. For a vehicle that is its wall, with the
tunnel's floor, walls, inlet and outlet read as conditions only. An interior task
(``interior=True``) predicts at the ``prediction_points`` given to ``forward``: a
point-cloud ``Mesh``, whose ``points`` alone are read, or a bare ``(N, d)``
tensor. They enter the encoder as tokens next to the boundary cells, and the
outputs are returned for them only. Their only inputs are their positions and the
geometry the model computes for them; each condition's per-cell boundary inputs
(declared in ``boundary_schemas``) are carried by that condition's cells.

The interior task takes the query-independent form: the prediction-point tokens
are passive and carry no routing weight. At every depth they read the slices,
anchors and prototypes the boundary formed, and never write into them. After
every block each one also reads every boundary cell (``_tree_read``): a learned
kernel of the pair's relative geometry and an attention over the cells' states at
that depth, weighted by each cell's share of the boundary measure and evaluated
through a Barnes-Hut forest over the cells. The prediction at a point is then a
function of that point and the boundary alone, so any request can be split into
chunks with the same result. The two tasks own different parameter sets (the
interior task adds the prediction-point input layer and the boundary read). The
choice is therefore a constructor argument: a parameter that takes no part in a
forward pass stops DDP.

**Interior geometry.** A prediction-point token needs a direction (its "normal"
in every invariant) and a distance to the wall. The model computes both from the
boundary *surface*, the union of the cells of ``boundary_meshes`` at full
resolution, and never takes them as inputs (``exact_boundary_geometry``). For
each prediction point :math:`x_q`, the exact closest point :math:`p_q` on the
surface's cells (point to face, never point to centroid) gives the distance
:math:`d_q = \pm |x_q - p_q|`. The generalized winding number of the oriented
cells gives its sign, positive on the side the cells' normals point to. An open
surface, such as a boundary given without its far field, is signed by the
threshold :math:`w = \sigma / 2`, with :math:`\sigma` the sign of the volume
the cells enclose, which depends on its shape: give the whole boundary for a
reliable sign. The direction :math:`u_q` is the cell's normal, oriented toward
the query, when :math:`p_q` is interior to a cell, and
:math:`(x_q - p_q) / |x_q - p_q|` otherwise. The token reads
:math:`s_q = d_q / L` and :math:`\operatorname{asinh}(s_q / 10^{-6})`
(``soft_log_length``: logarithmic away from the wall, linear through it and
within :math:`10^{-6} L` of it) as its two distance channels. Its direction is
:math:`\operatorname{sign}(d_q)\, u_q`, the gradient of the signed distance, with
:math:`\operatorname{sign}(0) = +1`. Both are continuous through the surface: a
point on it, whose distance is zero or a rounding error of either sign, gets the
code of the points just off it on either side. The boundary read also takes
:math:`\log(r / |d_q|)`, floored at :math:`|d_q| = 10^{-12} L`, which is not
continuous there: predict boundary values at a small offset into the domain, the
offset the model was trained at.

:math:`(d_q, u_q)` are functions of the surface and its orientation alone. They
are the same for every draw of the boundary tokens, invariant to splitting a
cell, and converge as :math:`O(h^2)` under refinement of the surface, as a stored
signed-distance field computed from the same surface does; the model therefore
does not read one. The search is a Warp BVH in three dimensions (a torch custom
op) and an exact brute force over the segments in two; the interior task is
implemented in two and three dimensions. The closest point is rebuilt from its
cell's vertices, so both quantities are differentiable. Both are built from
differences of positions, so every covariance contract holds.

**Boundary tokens of the interior task.** The boundary tokens are a sample of the
same cells, drawn inside the forward pass (``_sampling.sample_boundary_cells``).
The ``n_boundary_tokens`` (default 200,000) tokens per case are shared over the
named boundaries by water filling: a boundary with fewer cells than its share is
kept whole, and its unused share goes to the others. Each cell is drawn with
probability :math:`\pi_i` proportional to its measure (capped at one), by
systematic sampling along the Morton order of the cell centroids. The count is
exactly :math:`\min(\text{cells}, \text{n\_boundary\_tokens})`, a function of the
cell counts only, so the draw keeps static shapes. Each kept cell's measure is its
Horvitz-Thompson weight :math:`a_i / \pi_i`, so every measure-weighted read is an
unbiased quadrature of the full surface, exact on a boundary kept whole. The weight
takes the gradient of :math:`a_i` in the vertices with the kept cells and
:math:`\pi_i` held fixed (exact on a boundary kept whole, where :math:`\pi_i = 1`;
otherwise the kept set changes in jumps as the vertices move). The
sample is spread along the space-filling curve, so every part of the surface is
represented. In training the draw's start is uniform from the global torch RNG
(``torch.manual_seed`` reproduces it). In evaluation it is fixed, so inference is
deterministic. The draw reads the cells' measures and centroids in float64 from
the input meshes, so it is the same at every model precision. A surface task
samples none: it reads every cell it is given, and its outputs are predicted at
the cells of its prediction boundaries, which must all be tokens.

**Global inputs.** Rank-1 global inputs are the :math:`K` global directions. The
model normalizes each to a unit direction, since a direction is what the
invariants consume; a physically meaningful magnitude belongs among the rank-0
global inputs. A direction enters the seed through :math:`\hat n_i \cdot \hat
g_k`, the layers through :math:`\hat g_k \cdot \hat\rho_{is}`, and the head basis
as :math:`\hat g_k` with the complements of :math:`(\hat u_i, \hat g_k)`. Rank-0
global inputs are appended to every token's seed. Global inputs of rank 2 are
refused: broadcast the tensor to the boundary cells and declare it as a boundary
input instead.

**Kernels.** The per-layer geometry region has two exact implementations of the
same arithmetic. The eager PyTorch region rebuilds its intermediates in backward
instead of storing them. The fused one is one Triton kernel per direction that
never materializes the (B, N, S, .) tensor. It requires CUDA, float32 or float64
geometry, autocast off or in bfloat16 (then with float32 parameters), at most
three spatial dimensions, at most 1024 slices, and at most four per-point
vectors, counting the global directions (none counts as one) and the token's
rank-1 inputs together. It also requires no rank-2 inputs, since it does not
implement their per-token contractions (rho^T T rho). At the first forward pass
the model picks the fused kernel when the inputs are on CUDA, Triton is
importable and these constraints hold, and the eager region otherwise, logging
the choice once (on CUDA, a warning with its reason). The search of the prediction
points' boundary geometry likewise runs a fast implementation on CUDA when Triton
is importable, and its reference otherwise; the two give the same outputs
bitwise. Each piece of the interior task's tree read (the forest build, the walk,
and the read with its node means) runs its fast kernel where that piece's
requirements hold, and its reference otherwise, logging each choice once. The
forest and the pairs are the same either way, and the read is the same up to
summation order. Setting :attr:`~physicsnemo.experimental.models.isla.ISLA.reference_kernels` to ``True`` runs every
one of these on its reference path: the eager geometry region, the reference
search and every piece's reference read. It is an execution choice, not a
constructor argument, so a checkpoint does not store it and a loaded model starts
at ``False``.

Every constructor argument and every ``forward`` input is keyword-only, so a call
reads like the recipe's ``forward_kwargs`` mapping, and models can be swapped
without positional bookkeeping. Both ``forward(boundary_meshes=...,
prediction_points=..., global_data=...)`` (interior) and
``forward(boundary_meshes=..., prediction_boundaries=..., global_data=...)``
(surface) return a point-cloud ``Mesh``; see the class docstring of
:class:`ISLA`.

**Data layout.** Inside ``forward`` the tokens stay TensorDicts. Each boundary
condition's cells hold their centroid, normal and measure, and the invariants of
the condition's declared ``cell_data`` fields with its vectors and tensors. These
are in a block layout over every declared condition, so the cells of all
conditions, and the in-model draw's sample of them, are one token set with static
widths. The boundary cells and the prediction points are concatenated into one
token state over ``(batch, token)``, completed to a multiple of
``_TOKEN_ALIGNMENT`` tokens by padding tokens of no measure that nothing reads.
The field inputs enter the invariants, and the head's coefficients leave as the
output fields, through cached, schema-ordered :class:`~tensordict.PackLayout`
objects; the output ``Mesh`` is built from them. The token axis itself stays
dense where the arithmetic needs it: every softmax over points spans all tokens
at once (passive tokens excluded by ``log_w = -inf``), the slice anchors are
einsums over that axis, the fused geometry kernel reads its ``(B, N, .)``
strides, and the boundary read indexes the flattened boundary tokens. The slice
blocks take plain tensors.

The surface task's whole model, ``Mesh`` inputs and output included, compiles with
``model.compile(fullgraph=True, dynamic=True)`` into one graph for every boundary
count, boundary inputs up to rank 2 (``test_isla_compile.py``). **Compile it with**
:meth:`~physicsnemo.experimental.models.isla.ISLA.compile`, **not** ``torch.compile(model)``: on torch 2.13
Inductor's default options make the compiled model compute a different function
wherever it runs the eager geometry region, without an error or a warning. The interior task
does not: its boundary read sizes its pairs from the data and breaks a
fullgraph trace, so it compiles with ``compile_blocks``.
"""

import contextlib
import functools
import logging
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from tensordict import TensorDict

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.models.isla import _fused_read
from physicsnemo.experimental.models.isla._blocks import (
    MLP_RATIO,
    ROUTING_TEMPERATURE,
    _QueryTokens,
    _RoutingLogits,
    _SliceBlock,
    _softmax_over_points,
)
from physicsnemo.experimental.models.isla._geometry import (
    exact_boundary_geometry,
    soft_log_length,
)
from physicsnemo.experimental.models.isla._heads import (
    _assemble_output,
    _OutputLayout,
    _polar_complement,
    _pseudovectors,
)
from physicsnemo.experimental.models.isla._inputs import (
    _BoundaryCondition,
    _contract_vectors,
    _declare,
    _InputSet,
    _missing_fields,
    _pack,
    _schema_layout,
)
from physicsnemo.experimental.models.isla._sampling import sample_boundary_cells
from physicsnemo.experimental.models.isla._scales import (
    length_unit,
    merged,
    parse_scales,
    rms_moments,
    scale_factor,
    split_scale_keys,
)
from physicsnemo.experimental.models.isla._tree_read import (
    normal_error_bound,
    TREE_READ_LEAF_SIZE,
    TREE_READ_THETA,
    TreeReadWalk,
    _KernelBasis,
    _TreeKernelRead,
    kernel_geo_width,
    tree_read_walk,
    walk_pair_counts,
    walk_points_range,
    walk_range_pairs,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY, cell_measures
from physicsnemo.mesh.fields import FieldSchema, FieldSchemaLike
from physicsnemo.mesh.sampling.systematic_sampling import _draw_inputs
from physicsnemo.utils._tracing import is_tracing

logger = logging.getLogger(__name__)

#: Numerical floor for norms and logarithms of quantities without units (lengths
#: in frame units, unit normals) and, relative to the case's total, for the cells'
#: measures.
_EPS = 1e-12
#: The interior task's default boundary-token budget per case.
_BOUNDARY_TOKENS = 200_000
#: The systematic draw's start in evaluation mode (the middle of its range), so
#: that inference is deterministic.
_EVAL_START = 0.5
#: The token axis is completed to a multiple of this count by padding tokens
#: (:func:`_pad_tokens`). On CUDA in bfloat16 the slice blocks' GEMMs that reduce
#: over the token axis run on cuBLAS's fast kernels only when its rows span a
#: multiple of 16 bytes (8 tokens), and the LayerNorm backward's weight-gradient
#: reduction over more than 65,536 rows takes its aligned path only for a multiple
#: of its 32-row tile.
_TOKEN_ALIGNMENT = 32
#: The default of ``query_chunk_pairs``: the most walk pairs (prediction point,
#: boundary source) an interior task's :meth:`ISLA.forward` reads in one chunk of
#: prediction points. Above it the points run in chunks of consecutive points, each
#: holding only its own pair lists and kernel basis, so the read's memory stays flat
#: in the point count and each chunk stays within the fast walk's array limit.
#: Follow-up: the default is not tied to the device's memory (the docstring gives
#: the cost per pair); one derived from it would avoid a first out-of-memory error.
#: A surface forward has no such control: every cell goes through each block at once.
_QUERY_CHUNK_PAIRS = 2**28


### Inductor options of the compiled slice blocks (compile_blocks) and of the whole
### model (ISLA.compile). Mix-order reduction must stay off until the installed torch
### contains the fix for pytorch/pytorch#195845 (fixed on PyTorch main after 2.13).
### With it on, blocks in the eager geometry region add the padded rows of the last
### token tile to the per-slice sums whenever the token count is not a multiple of the
### tile height, and so compute a different function from the eager model. Remove
### these options and ISLA.compile once the minimum torch has the fix.
_BLOCK_COMPILE_OPTIONS = {"triton.mix_order_reduction": False}
#: The values of each execution-choice attribute (module docstring, *Kernels*), checked
#: when it is resolved (:meth:`ISLA._kernel_choice`).
_KERNEL_CHOICES = {
    "_geo_kernel": ("auto", "eager", "fused"),
    "_geometry_kernel": ("auto", "reference", "fast"),
    "_tree_read_kernel": ("auto", "reference"),
}
#: Each execution-choice attribute's reference path, which ``reference_kernels=True`` sets.
_REFERENCE_KERNELS = {"_geo_kernel": "eager", "_geometry_kernel": "reference", "_tree_read_kernel": "reference"}
#: The code object of the activation budget's wrapper for each block function it wraps
#: (:meth:`ISLA._compile_block`).
_BUDGET_CODES: dict = {}


def _hint_token_axis(n, widths) -> None:
    """Tell torch.compile that ``k * n`` is a multiple of the token count ``n``.

    With dynamic shapes a token axis made of several inputs (boundary meshes,
    boundary and query tokens) has a length that is a sum of sizes, ``a + b``,
    and Inductor cannot prove that ``k * a + k * b`` divides by ``a + b``: its
    code generation fails with CantSplit where it tiles a flattened ``(k n,)``
    loop of a ``(B, n, k)`` tensor as ``(n, k)``. :func:`torch._check` records
    each (always true) fact as a guard, which those static checks then read.
    Skipped in eager mode, where every fact is trivially true; the test is
    ``is_tracing()`` since dynamo traces a SymInt as an int.

    The CantSplit failure is an Inductor bug in torch 2.13, fixed
    from torch 2.14.0 on. Its trigger in ISLA is an affine LayerNorm over a
    concatenated token axis whose output feeds both a dtype cast and a
    reduction over the hidden dimension (``norm_assign(h)`` into ``k_proj``
    and ``F.normalize``). Remove this function once the minimum torch is
    2.14.0, after checking that compiles across token counts still produce
    one graph: without these hints, torch 2.14 compiled a second graph when
    the token count doubled."""
    if is_tracing():
        for k in widths:
            torch._check(k * n % n == 0)


def _pad_tokens(tokens: TensorDict, alignment: int) -> TensorDict:
    """``tokens`` ``(B, N)`` followed by ``-N mod alignment`` padding tokens: every
    field zero and the log-weight ``-inf``. Like passive tokens they hold no measure
    in any softmax over points, so no other token reads them, and no output is read
    from them: every output and gradient is the unpadded one up to the summation
    order of the longer reductions."""
    b, n = tokens.batch_size
    pad = -n % alignment
    ### In eager mode an aligned count keeps the tokens as they are. A compiled graph
    ### always pads: a test of pad == 0 would guard the dynamic token count on it.
    if not is_tracing() and pad == 0:
        return tokens

    ### F.pad rather than a concatenated block of pad tokens: under dynamic shapes
    ### a separate (B, pad, .) tensor makes torch.compile guard on pad == 0.
    def fill(v, value=0.0):
        return F.pad(v, (0, 0) * (v.dim() - 2) + (0, pad), value=value)

    padded = TensorDict({k: fill(v) for k, v in tokens.items()}, batch_size=[b, n + pad])
    padded.set("log_w", fill(tokens["log_w"], -math.inf))
    return padded


def _integer_knob(name: str, value) -> int:
    """A constructor knob that is a positive integer, as an int (a ValueError naming it otherwise)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return int(value)


def _real_knob(name: str, value, positive: bool) -> float:
    """A constructor knob that is a finite number at least 0 (above 0 when ``positive``),
    as a float (a ValueError naming it otherwise)."""
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        bound = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite {bound} number; got {value!r}")
    return float(value)


def _token_count(name: str, value) -> int:
    """A positive integer (an integral float included), as an int (a ValueError naming
    it otherwise)."""
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return int(value)


def _checked_knob(name: str, check, doc: str, default, surface: str) -> property:
    """A knob the forward pass reads, as a property whose every assignment, the
    constructor's and a later one, is checked by ``check(name, value)``. The knob
    belongs to the interior task, so a surface task refuses any value but
    ``default``, with the message ``surface`` (formatted with ``default`` and
    ``value``)."""
    attr = f"_{name}"

    def set_(self, value):
        value = check(name, value)
        if getattr(self, "interior", True) is False and value != default:
            raise ValueError(surface.format(name=name, default=default, value=value))
        setattr(self, attr, value)

    return property(lambda self: getattr(self, attr), set_, doc=doc)


#: A surface task's refusal of a boundary-read knob (:func:`_checked_knob`).
_READ_KNOB_ON_SURFACE = (
    "{name} sets the interior task's boundary read; a surface task (interior=False) "
    "has none, so it takes the default ({default!r}); got {value!r}"
)


def _refuse_earlier_state_dict(module, state_dict, prefix, local_metadata, *args) -> None:
    """A load_state_dict pre-hook: a state dict whose state-dict version (torch's
    ``nn.Module._version``, recorded in its metadata) is not ``ISLA._version`` is
    refused on every loading path (``Module.load``, ``load_checkpoint``,
    ``load_state_dict``), not only ``Module.from_checkpoint``, which checks the
    recorded checkpoint version. ISLA keeps no upgrade path. A state dict without
    metadata (a copy into a plain ``dict`` drops it, and so does
    ``torch.distributed.checkpoint``'s full state dict) loads with a warning that
    nothing was checked."""
    version = local_metadata.get("version")
    if version is None:
        logger.warning(
            "ISLA: this state dict has no metadata (torch's state_dict()._metadata, which a copy into a "
            "plain dict or torch.distributed.checkpoint's full state dict drops), so neither its "
            "state-dict version nor its fields' order is checked"
        )
    elif version != type(module)._version:
        raise RuntimeError(
            f"this ISLA state dict has state-dict version {version}; ISLA "
            f"{module.__model_checkpoint_version__} loads version {type(module)._version} only (no upgrade path)"
        )


def _field_order(module) -> dict[str, list[str]]:
    """The declared fields' names in the orders the trained weights follow, one list
    per set of columns: the output schema's (the head's), and the global schema's and
    each boundary condition's of each rank (the condition names themselves are sorted)."""
    order = {
        "output_schema": list(module.output_schema),
        "global_schema rank-1": list(module._global_vector_names),
        "global_schema rank-0": list(module._global_scalar_names),
    }
    for name, schema in module.boundary_schemas.items():
        for rank in range(module.MAX_INPUT_RANK + 1):
            order[f"boundary_schemas[{name!r}] rank-{rank}"] = [k for k, s in schema.items() if s.rank == rank]
    return {group: names for group, names in order.items() if names}


def _record_field_order(module, state_dict, prefix, local_metadata) -> None:
    """A state_dict post-hook: the declared fields' order (:func:`_field_order`) in the
    state dict's metadata, beside its version, for :func:`_refuse_reordered_fields`."""
    local_metadata["fields"] = _field_order(module)


def _refuse_reordered_fields(module, state_dict, prefix, local_metadata, *args) -> None:
    """A load_state_dict pre-hook: a state dict whose metadata records a field of this
    model at another place in its schema is refused, since its weights would load
    into another field's columns without a shape mismatch. A renamed field, or a
    state dict without the record (an earlier save, or no metadata, as from a plain
    ``dict`` or ``torch.distributed.checkpoint``), is not checked."""
    stored = local_metadata.get("fields")
    if stored is None:
        return
    for group, names in _field_order(module).items():
        saved = stored.get(group, [])
        if any(n in saved and saved.index(n) != i for i, n in enumerate(names)):
            raise RuntimeError(
                f"this ISLA state dict was saved with the {group} fields in the order {saved!r} and this "
                f"model declares {names!r}: the weights follow the declaration order, so they would load "
                "into the wrong fields; declare the fields in the saved order"
            )


class _CaseInputs(NamedTuple):
    """One case's inputs to the network (:meth:`ISLA._case_inputs`): the surface task's
    :meth:`ISLA._forward_tokens`, the interior task's :class:`QueryChunks`."""

    #: The boundary tokens over ``(1, N)`` (the interior task's drawn ones).
    boundary: TensorDict
    #: The prediction-point tokens over ``(1, Q)`` (interior), else None.
    query: TensorDict | None
    #: The boundary surface ``(points, cells)`` the prediction points' geometry is
    #: measured from (interior), else None.
    surface: tuple[torch.Tensor, torch.Tensor] | None
    #: The unit global directions ``(1, K, d)`` and the global scalars ``(1, S)`` or None.
    g_unit: torch.Tensor
    g_scalars: torch.Tensor | None
    #: The length unit ``(1, 1, 1)``.
    gauge: torch.Tensor
    #: The case's scales (:meth:`ISLA._case_scales`), None when none are declared.
    scales: dict | None
    #: The output Mesh's points and global data.
    out_points: torch.Tensor
    global_td: TensorDict
    #: The surface task's output token rows as ``(start, stop)`` ranges, None for every row.
    out_rows: list[tuple[int, int]] | None


class ISLA(Module):
    r"""ISLA (Invariant Slice Attention): invariant backbone, equivariant edges.

    A soft-slice transformer for steady boundary-value problems. Its attention
    operates only on invariants of the per-point vector set, and its heads
    re-assemble outputs of any rank in an equivariant basis, so the network is
    exactly covariant under rotations, reflections and translations by
    construction (the module docstring gives the contracts). The full model is
    exactly covariant, up to roundoff, when the boundary's canonical frame is
    unique and not close to a tie. At a tie (a symmetry that reverses a principal
    axis: a mirror plane, a two-fold axis, a centre of symmetry; or equal spread
    along two principal axes) a surface task stays exact. An interior task is
    then exact only with ``n_boundary_tokens`` at least the cell count,
    ``tree_read_theta=0`` and a closed boundary, since its draw, read and
    winding-number tree are laid out in that frame. Otherwise it is approximate
    (module docstring, *Where the covariance is exact*). Every argument is
    keyword-only.

    For more details, see the paper: Peter Sharpe, *ISLA: A Coordinate-Free,
    Measure-Weighted Transformer for Boundary-Driven PDEs*, arXiv preprint
    `arXiv:2611.17141 <https://arxiv.org/abs/2611.17141>`__, 2026.

    **Setting up a new problem.** ISLA learns the map from a boundary-value
    problem's boundary data to its solution, so its inputs must pose that problem
    completely. Its inputs differ from a point-cloud model's. These are the
    usual ways a setup goes wrong:

    1. *Every condition, with its data.* Pass every part of the domain's
       boundary (inlets, outlets, walls, far field, tunnel walls, symmetry
       planes) as a condition. Declare in ``boundary_schemas`` every value
       each one prescribes (an inlet's velocity and turbulence values, an
       outlet's pressure, a moving wall's velocity). ``forward`` accepts any
       subset of the declared conditions, so a pipeline that drops one runs
       without an error and trains on an underposed problem. If a dataset
       stores only the body, build the missing parts from its own documented
       constants. The inputs must determine the solution: a pressure defined
       up to a constant needs the condition that fixes its level (an outlet's
       prescribed pressure). A field that is zero in every training case
       contributes nothing and its weights get no gradient, so a later
       fine-tune on cases where it varies starts from untrained weights.
    2. *A condition is a boundary-condition type.* Use one name per kind of
       condition (what is prescribed, and how), with the same partition in
       every case. Each declared condition has its own input layer and, with
       more than one, its own learned share of every measure-weighted read
       (module docstring, *Boundary conditions by name*), so splitting one
       condition into parts, or lumping different ones together, changes
       the model.
    3. *Cells at full resolution.* Each boundary is a ``Mesh[n_spatial_dims -
       1, n_spatial_dims]`` of cells, not a point cloud: the model reads the
       cells' centroids, normals and measures. For an interior task, pass every
       cell and let ``n_boundary_tokens`` draw the tokens: decimating or
       subsampling the boundary first changes the surface the prediction
       points measure their distance to, and the length unit. A surface task
       reads every cell it is given, so subsample a large surface in the data
       pipeline, with each kept cell's Horvitz-Thompson measure in
       ``cell_data["_effective_measure"]``.
    4. *One orientation.* A cell's normal follows its vertex order. Orient
       every boundary the same way in every case, into the domain where the
       outputs live (into the fluid, for a flow). The interior task's signed
       distance is positive on the side the normals point to, and the input
       invariants read each vector input against the normal. Reorient in the data
       pipeline, not in the stored data: reverse the vertex order of a
       boundary's cells to flip its normals. The model does not check the
       orientation, so a boundary oriented the other way gives the opposite
       signed distance without an error.
    5. *No derived geometry and no coordinates as fields.* The model computes
       every geometric quantity it uses (the length unit, the frame, each
       prediction point's signed distance and direction) from the boundary
       cells. Coordinates given as fields break the covariance contracts, a
       stored signed-distance field duplicates what the model computes, and
       the prediction points' ``point_data`` is never read.
    6. *Declared ranks.* A vector is one rank-1 field and a stress one rank-2
       field (``symmetric`` when it is), never several scalars: the
       covariance holds per declared field.
    7. *Units.* Declare ``units`` on every dimensional field, and let a field
       define each scale: an inlet's velocity with ``defines: U`` (its
       measure-weighted RMS speed, and its mean direction as a global
       direction), a loaded face's traction the same way for a stress scale, a
       density global with ``defines: rho``. The length ``L`` is built in: the
       RMS pairwise distance of all the given boundaries, so it measures the
       whole domain when its far field is given. Each input is divided by its
       units' scale and each output multiplied by its own, so the model is
       covariant under similarity. A dimensional field without ``units``
       enters, or leaves, as raw numbers.
       :func:`~physicsnemo.experimental.models.isla.case_scales` gives a case's
       scales for preprocessing or for another model.
    8. *Global inputs.* A rank-0 global enters every token's seed as given
       (after its units and ``scale``, when declared), so pass it
       dimensionless and of order one: a standardized log Reynolds number,
       not ``Re`` or ``nu``. A rank-1 global keeps only its direction, so give
       a magnitude that matters as a rank-0 global or a scale.
    9. *Interior or surface.* ``interior=True`` predicts at prediction points
       in the domain, ``interior=False`` at the cells of named boundaries.
       The two have different parameters, so the choice is made at
       construction.
    10. *One case per call.* ``forward`` takes one case, with no batch axis,
        and returns a point-cloud ``Mesh``, not a ``(B, N, C)`` tensor: train
        with one case per GPU. An interior task's prediction points are
        independent of one another, so train on a random subset of each
        case's points, and predict any number of them in chunks
        (``query_chunk_pairs``, :meth:`query_chunks`).
    11. *Targets.* With ``output_stats`` the outputs are standardized physical
        fields, ``(y - mean) / std``; without it, physical fields (when units
        are declared). Train against targets in the same form.

    A model configuration for a dataset records these choices once, for every
    case: the conditions and their fields, the units and defined scales, the
    global inputs, the task and the output statistics.

    **Rank specs.** ``output_schema``, each schema of ``boundary_schemas`` and
    ``global_schema`` are :data:`~physicsnemo.mesh.fields.FieldSchemaLike`
    declarations (module docstring, *Rank specs*); each parsed declaration is the
    :class:`~physicsnemo.mesh.fields.FieldSchema` attribute of the same name.

    Parameters
    ----------
    output_schema : FieldSchemaLike
        Output fields, a rank spec per name (module docstring, *Rank specs*). Each
        field is a leaf of the returned Mesh's ``point_data``, nested per the dotted
        name: a rank-0 field has shape :math:`(N,)`, a rank-:math:`r` field
        :math:`(N, d, \dots, d)` with :math:`r` trailing axes (:math:`d` is
        ``n_spatial_dims``); a symmetric spec gives a symmetric tensor. Ranks up to
        :attr:`MAX_OUTPUT_RANK`. Each field's coefficients come from the shared
        linear head; with :math:`m = 2K + 3` true basis vectors and :math:`p`
        pseudovectors (:math:`(K+2)(K+1)/2` in three dimensions, :math:`K + 2` in
        two) the counts per field are 1 at rank 0; :math:`m` (even) or :math:`p`
        (odd) at rank 1; :math:`m^2 + 1` (even, the identity included) or
        :math:`m(m+1)/2 + 1` (even symmetric), :math:`2pm` (odd; :math:`2pm + 1`
        in two dimensions, where the Levi-Civita symbol is an odd rank-2 tensor)
        or :math:`pm` (odd symmetric) at rank 2. For :math:`K = 1`
        (:math:`m = 5, p = 3`) a vector costs 5, a symmetric rank-2 tensor 16
        and a pseudovector 3.
    boundary_schemas : Mapping[str, FieldSchemaLike]
        The boundary conditions by name, each with the per-cell fields it carries:
        a rank spec per field with ranks 0 to 2 and even parity (an inlet's
        prescribed velocity as ``{"rank": 1}``, an outlet's pressure as
        ``{"rank": 0}``), or an empty schema for a condition without fields (a
        no-slip wall: ``{}``). The names are fixed here, in canonical (sorted)
        order; a name is a non-empty string without ``"."``. ``forward`` takes any
        non-empty subset of them, each mesh carrying its own condition's fields.
        Each condition has its own input layer (``boundary_conditions[name].seed``).
        It maps the condition's seed ``[1, n.g_k, its fields' invariants, the
        global scalars]`` to the token width, ahead of the shared GELU and linear
        layer (``embed``). Its first column, on the constant, is the condition's
        learned offset, so a condition's identity is in the parameters its tokens
        read even when it declares no field. A condition's rank-1 and rank-2
        inputs (and the vectors derived from its rank-2 inputs) reach the slice
        layers through its own linear maps onto shared vector and tensor channels,
        as many as the widest condition has. There they contract with the
        token-to-anchor direction (module docstring); every condition shares the
        slice layers themselves. Declare every condition of the domain's
        boundary and every value it prescribes, one name per condition type with
        the same partition in every case (*Setting up a new problem*, items 1 and
        2).
    global_schema : FieldSchemaLike, optional, default=None
        Per-case fields, a rank spec per name with rank 0 (appended to every
        token's seed) or rank 1 (a global direction, normalized to unit length
        inside the model). The :math:`K` rank-1 entries, in declaration order, are
        the global directions of the geometry region and the head basis. Read from
        ``global_data`` by name; extra leaves there are ignored. A rank-0 field
        without ``units`` enters as given, so it should be dimensionless and of
        order one; a rank-1 field's magnitude is dropped (*Setting up a new
        problem*, item 8).
    interior : bool, optional, default=False
        The task structure. ``False``: a surface task. The outputs are predicted
        at the cells of the boundaries ``forward`` names in
        ``prediction_boundaries`` (every given boundary is read); ``forward`` takes
        no ``prediction_points``, and the model has no prediction-point parameters.
        ``True``: an interior task. The outputs are predicted at the
        ``prediction_points`` that ``forward`` requires, which enter the encoder
        as passive tokens (the query-independent form; module docstring,
        *Surface and interior tasks*), so each prediction depends on its own
        point and the boundary only. The boundary read breaks the whole-model
        fullgraph compile (use ``compile_blocks``), and rank-2 boundary inputs
        are refused.
        The two tasks have different parameter sets (a parameter without a
        gradient stops DDP), so the choice is made at construction.
    n_boundary_tokens : int, optional, default=200000
        Interior task only: the number of boundary tokens per case. ``forward``
        takes each named boundary at full resolution and draws its tokens from
        the cells inside the model, each kept cell with its Horvitz-Thompson
        measure (module docstring, *Boundary tokens of the interior task*). The
        draw's Morton order is taken in the boundary's canonical frame, so the
        drawn cells move with the case unless the boundary has a symmetry that
        reverses a principal axis (a mirror plane, a two-fold axis, a centre of
        symmetry), or equal spread along two principal axes (module docstring,
        *Where the covariance is exact*). The token count
        is exactly ``min(total cells, n_boundary_tokens)``; a boundary with all
        its cells kept is read as given. In training the draw's start is uniform
        from the global torch RNG (``torch.manual_seed`` reproduces it). In
        evaluation (``model.eval()``) it is fixed, so inference is deterministic.
        Pass the boundaries undecimated and unsubsampled: the draw is the
        subsampling. A surface task reads every cell it is given and samples
        none, so it takes only the default.
    tree_read_theta : float, optional, default=1.0
        Interior task only: the Barnes-Hut opening angle :math:`\theta` of the
        boundary read. A cluster of boundary cells is read as one virtual cell
        when its box diagonal is below :math:`\theta` times its distance from the
        point and the RMS chord of its normals about their mean is below
        :math:`\theta`. A smaller angle reads more cells one by one, more exactly
        and at more pairs per point; 0 is the exact read (every cell of every
        point). The orientation split of the read's forest already bounds the
        chord, so above :math:`\theta = \sqrt{2 (1 - 1/\sqrt{d})}` (0.92 in three
        dimensions, 0.77 in two), the default included, the coherence test
        rejects no cluster. At :math:`\theta > 0` the read is piecewise smooth:
        the output jumps where a cluster's admission changes, in the point and in
        the boundary's vertices, and a cluster's position and normal take no
        gradient. At 0 the read is smooth, and the model's derivatives in the
        prediction points and in the boundary's vertices are complete. When the
        interior task keeps fewer boundary tokens than cells, the kept set
        changes in jumps as the vertices move, and the vertex derivative holds
        the kept cells and their inclusion probabilities fixed (the token draw,
        in the module docstring). Read a model at the angle it was trained
        at: a cluster's aggregate is what the kernel learns to read. A finite
        number at least 0; a surface task takes only the default.
    tree_read_leaf_size : int, optional, default=4
        Interior task only: the boundary cells per leaf of the read's trees. At
        :math:`\theta > 0` a leaf that is not read as a cluster is read cell by
        cell, so a larger leaf reads more cells exactly near each point, at more
        pairs per point and a shallower tree; at :math:`\theta = 0` it changes
        nothing. A positive integer; a surface task takes only the default.
    query_chunk_pairs : int, optional, default=2**28
        Interior task only: the most walk pairs (prediction point, boundary
        source) ``forward`` reads in one chunk of prediction points. Above it the
        points run in chunks of consecutive points, each holding only its own
        pairs, so a smaller value lowers the peak memory of a large point set at
        the cost of more, smaller launches; the outputs are the same up to
        summation order. A chunk's peak memory grows linearly in its pairs and its
        points: at ``hidden=256`` in float32 about 0.3 GiB per million pairs where
        each point reads many sources and about 1.5 GiB where each reads ten, so the
        default can need tens of GiB; lower it to fit the device.
        :meth:`query_chunks` takes its chunks from the caller. A positive integer;
        a surface task takes only the default. This knob, ``n_boundary_tokens``
        and ``tree_read_theta`` may also be set after construction, where their
        ranges are checked as here (a value set so is not saved with the model;
        pass it to ``Module.from_checkpoint``'s ``override_args`` instead).
    hidden : int, optional, default=512
        Token width of the slice blocks.
    n_layers : int, optional, default=12
        Number of slice-attention blocks.
    n_slices : int, optional, default=128
        Number of soft slices (data-adaptive anchors) per block.
    mlp_ratio : int, optional, default=4
        Expansion ratio of each block's per-token and per-slice MLPs: their hidden
        width is ``mlp_ratio * hidden``, so their parameters and compute grow
        linearly in it. A positive integer.
    routing_temperature : float, optional, default=10.0
        The bound :math:`T` of the cosine routing logits: each block's
        point-to-slice logit, and the head frame's, is :math:`T` times a cosine,
        so it lies in :math:`[-T, T]`. A larger :math:`T` lets a block route each
        point more sharply to one slice; a smaller one keeps the routing closer to
        uniform. It costs nothing. A finite positive number.
    compile_blocks : bool, optional, default=False
        Run every slice block through :func:`torch.compile`. The compiled function
        is the block's forward, built at the first forward pass, so parameter
        names, checkpoints and the eager path are unchanged; the fused geometry
        kernel stays a custom-op leaf. In an interior task the boundary read's
        LayerNorms and its key, value and query maps compile too (dynamic shapes,
        one graph for every block), so float32 training gives gradients that
        differ from eager at rounding level. Compiles once per distinct token count
        (an interior task whose cases fill its token budget, with a fixed number of
        prediction points per step, compiles once); with a token count that
        changes every case, leave it off.
    n_spatial_dims : int, optional, default=3
        Spatial dimension :math:`d` of the domain, Mesh's ``n_spatial_dims``.
        Every boundary mesh is a ``Mesh[n_spatial_dims - 1, n_spatial_dims]`` and
        every prediction point has ``n_spatial_dims`` coordinates. Every geometric
        quantity is built for ``n_spatial_dims`` dimensions and the head's basis
        follows it (pseudovectors by the Levi-Civita construction: cross products
        in three dimensions, perpendiculars in two; odd-parity outputs are refused
        for ``n_spatial_dims >= 4``). The fused Triton geometry kernel runs two and
        three dimensions; higher ones run the eager region. The interior task (its
        exact boundary geometry) is implemented in two and three dimensions. Pass
        a two-dimensional problem with ``n_spatial_dims=2`` (polyline
        boundaries), not embedded with a zero third coordinate.
    output_stats : Mapping[str, Mapping], optional, default=None
        ``{name: {"mean", "std"}}`` over declared outputs (the format of
        ``NormalizeMeshFields.stats``; ``type`` is ignored): each output, after its
        units are restored, is returned as ``(y - mean) / std``, so the model
        predicts a dataset's standardized physical targets. The mean of a field of
        rank 1 or more must be 0. ``None``: the outputs as the head gives them (in
        physical units when declared). The schemas declare the units and the
        defined scales (module docstring, *Scales*): a field's mapping may carry
        ``units`` (``"L"`` and the defined scales' names to exponents),
        ``defines`` (a scale's name, on a rank-1 field or a rank-0 global) and
        ``scale`` (an output's positive constant :math:`c`: :math:`y = c \hat y
        \prod \text{scale}^{\text{exp}}`). With any of them, or ``output_stats``,
        the head and the output assembly run in the parameters' precision under
        autocast. A loss compares the outputs with targets in the same form:
        standardized by the same stats when they are given, physical otherwise.

    Forward
    -------
    boundary_meshes : Mapping[str, Mesh]
        The boundary, as ``Mesh[n_spatial_dims - 1, n_spatial_dims]`` objects keyed
        by boundary-condition name (a polyline in two dimensions, a surface in
        three): any non-empty subset of the names declared in
        ``boundary_schemas``, in any order. An unknown name is a ``ValueError``;
        an absent condition, like one with no cells, contributes no tokens. The
        subset is for a case that has no part of that kind (a domain without slip
        walls). A condition the case has but the pipeline leaves out is not
        reported, and the model then solves a different problem. Every mesh's
        cell normals point the same way in every case, into the domain where the
        outputs live (*Setting up a new problem*, item 4). Their floating dtype
        is the parameters' (another is a ``ValueError``). The cells are the
        boundary tokens, the conditions in canonical (sorted) order and each
        condition's cells in its mesh's order. A token's position is the cell
        centroid, its normal the cell normal, and its quadrature measure the
        cell's effective measure (``cell_data["_effective_measure"]``) when
        present, else the cell's geometric measure. Each condition's
        ``boundary_schemas`` fields are read from its mesh's ``cell_data`` by
        their (possibly dotted) names; other cell data are ignored. A surface
        task's tokens are every cell of every given boundary: those of its
        prediction boundaries and those read as conditions only (a tunnel's
        floor, walls, inlet and outlet). An interior task takes the boundaries at
        full resolution: their cells are the surface of the prediction points'
        geometry, and its tokens are ``n_boundary_tokens`` of them drawn inside
        the model (module docstring), each kept cell with its Horvitz-Thompson
        measure.
    prediction_points : Mesh or torch.Tensor, optional, default=None
        The interior task's prediction points (``interior=True``, required there):
        a point-cloud ``Mesh`` whose ``points`` are the prediction points (its
        ``point_data`` is not read, so features attached there are ignored), or a
        bare tensor of shape :math:`(N, d)`, in the boundary meshes' coordinates,
        of a floating dtype no wider than the parameters' (narrower points are
        promoted to it, the returned Mesh's points included; wider ones are a
        ``ValueError``). A rotation, reflection,
        translation or scaling applied to both transforms the outputs covariantly
        (exactly where the module docstring's *Where the covariance is exact*
        says), provided a reflection also reverses each boundary cell's vertex
        order, since that order sets the cell's normal; one applied to the points
        alone poses another problem. Each
        point's prediction depends on that point and the boundary only, so any
        subset of a case's points (a training sample, one chunk of a large
        evaluation) gives the same values at those points, up to summation order.
        The points enter the encoder as passive readers of the boundary (the
        query-independent form). Each one's signed distance to the boundary
        surface (the cells of ``boundary_meshes``) and the direction from it are
        computed inside the model, never taken as inputs (module docstring,
        *Interior geometry*). The points run after the boundary side, in chunks
        of at most ``query_chunk_pairs`` walk pairs each. ``None`` for a surface
        task.
    prediction_boundaries : list[str] or tuple[str, ...], optional, default=None
        The surface task's output boundaries (``interior=False``, required there):
        a non-empty list or tuple of distinct names of given boundaries with
        cells, at whose cells the outputs are predicted (DrivAerML Surface:
        ``["vehicle"]``, its floor, walls, inlet and outlet read as conditions
        only; a surface whose conditions together carry the outputs names them
        all). Their order does not matter. The argument owns no parameters: the
        head reads the same tokens whichever boundaries carry the outputs, and
        only the rows it returns differ. The prediction is a quadrature over the
        cells given for these boundaries (the request), so a request must cover
        the body. A larger request, or the mean of several, lowers the error: the
        spread between requests falls about as one over the square root of their
        cell count. ``None`` for an interior task.
    global_data : TensorDict or Mapping[str, torch.Tensor], optional, default=None
        The case's per-case fields, a TensorDict (or mapping) with the
        ``global_schema`` fields as leaves (a rank-0 field a scalar, a rank-1 field
        a tensor of shape :math:`(d,)`); extra leaves are ignored, a missing one is
        an error. ``None`` when nothing is declared. A rank-0 field without
        ``units`` should be dimensionless and of order one (a standardized log
        Reynolds number); a rank-1 field gives a direction only.

    Outputs
    -------
    Mesh
        A point-cloud ``Mesh`` whose ``points`` are the prediction points
        (interior) or the cell centroids of the prediction boundaries (surface:
        those boundaries in sorted name order, each one's cells in its mesh's
        order), whose ``point_data`` holds one leaf per declared output at its
        natural shape (:math:`(N,)` at rank 0, :math:`(N, d)` at rank 1,
        :math:`(N, d, d)` at rank 2, nested per the dotted names), and whose
        ``global_data`` is the selected ``global_data`` passed through. The
        outputs are physical fields when ``units`` are declared, and standardized
        by ``output_stats`` when it is given. A field declared with rank :math:`r`
        and even parity transforms as a true rank-:math:`r` tensor under every
        orthogonal map :math:`R` of the inputs (:math:`T \to R^{\otimes r} T`;
        rotations and reflections alike, a reflected boundary keeping its normals'
        side by reversing its cells' vertex order) and is invariant under
        translations. An
        odd field picks up the extra sign :math:`\det R`. Both hold exactly where
        the module docstring's *Where the covariance is exact* says, and
        approximately otherwise. Components in any frame are only ever a view of
        such a field.

    Notes
    -----
    The input checks of ``forward`` run in eager mode only; under
    :func:`torch.compile` a malformed case is not refused.

    :attr:`reference_kernels` (``False``) selects how the geometry region, the
    boundary geometry's search and the tree read run, not what they compute
    (module docstring, *Kernels*); checkpoints do not store it. ``block_activation_budget`` (1.0; above 0 and at
    most 1) is the share of their default saved activations that the compiled
    blocks keep for the backward pass (``compile_blocks``; the rest is recomputed
    there).

    *Public attributes.* The public surface is the constructor, ``forward``,
    :meth:`query_chunks`, :meth:`compile`, :attr:`MAX_OUTPUT_RANK`,
    :attr:`MAX_INPUT_RANK`, :attr:`reference_kernels` and
    ``block_activation_budget``. Each constructor argument is also kept as an
    attribute of its name, and ``boundary_names`` lists the declared conditions in
    canonical order. The submodules (``blocks``, ``head``, ``qt``, ``kernel_basis``,
    ``kernel_reads``, ...) name the checkpoint's entries; every other attribute is
    internal.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.isla import ISLA
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    >>> body = sphere_icosahedral.load(subdivisions=1)
    >>> model = ISLA(
    ...     output_schema={"pressure": {"rank": 0}, "wss": {"rank": 1}},
    ...     boundary_schemas={"body": {}},
    ...     global_schema={"U_inf_dir": {"rank": 1}},
    ...     hidden=32,
    ...     n_layers=2,
    ...     n_slices=8,
    ... )
    >>> out = model(
    ...     boundary_meshes={"body": body},
    ...     prediction_boundaries=["body"],
    ...     global_data={"U_inf_dir": torch.tensor([1.0, 0.0, 0.0])},
    ... )
    >>> out.point_data["pressure"].shape, out.point_data["wss"].shape
    (torch.Size([80]), torch.Size([80, 3]))
    """

    @dataclass
    class MetaData(ModelMetaData):
        jit: bool = False
        amp: bool = True

    #: Largest output rank; the ordered basis grows as ``m**rank``.
    MAX_OUTPUT_RANK = 4
    #: Largest per-token input rank; see the module docstring.
    MAX_INPUT_RANK = 2

    #: The checkpoint version ``Module`` records and ``Module.from_checkpoint`` checks.
    __model_checkpoint_version__ = "0.16.0"
    #: torch's state-dict version of the module (``nn.Module._version``, recorded in
    #: every state dict's metadata), so a state dict of another version is refused on
    #: load (:func:`_refuse_earlier_state_dict`).
    _version = 16

    #: The knobs the forward pass reads (the docstring), checked on every assignment as
    #: the constructor checks them, and overridable in ``Module.from_checkpoint``.
    n_boundary_tokens = _checked_knob(
        "n_boundary_tokens",
        _token_count,
        "The interior task's boundary tokens per case.",
        _BOUNDARY_TOKENS,
        "n_boundary_tokens is the interior task's budget of boundary tokens; a surface "
        "task (interior=False) reads every boundary cell it is given and samples "
        "none, so it takes the default ({default}); got {value!r}",
    )
    tree_read_theta = _checked_knob(
        "tree_read_theta",
        lambda name, value: _real_knob(name, value, positive=False),
        "The boundary read's Barnes-Hut opening angle.",
        TREE_READ_THETA,
        _READ_KNOB_ON_SURFACE,
    )
    query_chunk_pairs = _checked_knob(
        "query_chunk_pairs",
        _integer_knob,
        "The most walk pairs a chunk of prediction points reads in forward.",
        _QUERY_CHUNK_PAIRS,
        _READ_KNOB_ON_SURFACE,
    )
    _overridable_args = {"n_boundary_tokens", "tree_read_theta", "query_chunk_pairs"}

    def __init__(
        self,
        *,
        output_schema: FieldSchemaLike,
        boundary_schemas: Mapping[str, FieldSchemaLike],
        global_schema: FieldSchemaLike | None = None,
        interior: bool = False,
        n_boundary_tokens: int = _BOUNDARY_TOKENS,
        tree_read_theta: float = TREE_READ_THETA,
        tree_read_leaf_size: int = TREE_READ_LEAF_SIZE,
        query_chunk_pairs: int = _QUERY_CHUNK_PAIRS,
        hidden: int = 512,
        n_layers: int = 12,
        n_slices: int = 128,
        mlp_ratio: int = MLP_RATIO,
        routing_temperature: float = ROUTING_TEMPERATURE,
        compile_blocks: bool = False,
        n_spatial_dims: int = 3,
        output_stats: Mapping[str, Mapping] | None = None,
    ) -> None:
        super().__init__(meta=self.MetaData())
        ### compile_blocks (the docstring): the blocks see only the (B, n_tokens, .)
        ### tensors; the mesh-facing forward, whose vertex count varies per case, stays eager.
        self.compile_blocks = bool(compile_blocks)
        self._block_forwards: list | None = None
        self._block_reads: list | None = None
        ### Filled once a block with an activation budget has run outside torch.compile
        ### with gradients on and the model has said so (_compile_block); a list, so
        ### that the blocks' wrappers hold it and not the model.
        self._budget_eager_warned: list = []
        ### Whether an interior forward pass of this model has logged its chunks (once).
        self._forward_chunks_logged = False
        ### The head's basis and coefficient count depend on the spatial dimension d
        ### (pseudovectors by the Levi-Civita construction), so d is a constructor
        ### argument and forward checks the points against it.
        self.n_spatial_dims = _integer_knob("n_spatial_dims", n_spatial_dims)
        if self.n_spatial_dims < 2:
            raise ValueError(f"n_spatial_dims must be at least 2, got {n_spatial_dims!r}")
        ### Every input and output is a named field of declared rank; the grammar
        ### is the mesh models' shared one (physicsnemo.mesh.fields), flattened to
        ### dotted names here. The ISLA-local keys of a field (units, defines, scale;
        ### _scales.py) are popped before the shared grammar parses it.
        output_schema, out_extras = split_scale_keys("output_schema", output_schema)
        self.output_schema = _declare(
            "output_schema",
            output_schema,
            max_rank=self.MAX_OUTPUT_RANK,
            allow_odd=True,
        )
        if not self.output_schema:
            raise ValueError("output_schema must declare at least one field")
        ### The boundary conditions by name, in canonical (sorted) order, each with the
        ### per-cell fields it alone carries (an empty schema is a condition without
        ### fields, a no-slip wall).
        if not isinstance(boundary_schemas, Mapping) or not boundary_schemas:
            raise ValueError(
                "boundary_schemas must be a non-empty mapping of boundary-condition name to "
                f"field schema; got {boundary_schemas!r}"
            )
        for name in boundary_schemas:
            if not isinstance(name, str) or not name or "." in name:
                raise ValueError(
                    f"boundary_schemas: a condition name is a non-empty string without '.'; got {name!r}"
                )
            ### A name keys the conditions' nn.ModuleDict, so it cannot be one of its
            ### attributes ("keys", "update", "training", ...).
            if hasattr(nn.ModuleDict(), name):
                raise ValueError(
                    f"boundary_schemas: the condition name {name!r} is an attribute of nn.ModuleDict, "
                    "which holds the conditions' parameters; choose another name"
                )
        split = {
            name: split_scale_keys(f"boundary_schemas[{name!r}]", boundary_schemas[name])
            for name in sorted(boundary_schemas)
        }
        b_extras = {name: extras for name, (_, extras) in split.items()}
        self.boundary_schemas = {
            name: _declare(
                f"boundary_schemas[{name!r}]",
                spec,
                max_rank=self.MAX_INPUT_RANK,
                allow_odd=False,
            )
            for name, (spec, _) in split.items()
        }
        global_schema, g_extras = split_scale_keys("global_schema", global_schema)
        self.global_schema = _declare(
            "global_schema", global_schema, max_rank=1, allow_odd=False
        )
        ### The opt-in scales (module docstring, *Scales*), checked against the schemas.
        sc = parse_scales(
            out_extras, b_extras, g_extras, output_stats, self.output_schema, self.boundary_schemas, self.global_schema
        )
        self._output_units, self._output_scale = sc.output_units, sc.output_scale
        self._global_units, self._global_scale = sc.global_units, sc.global_scale
        self._boundary_units, self._boundary_scale = sc.boundary_units, sc.boundary_scale
        self._boundary_sources, self._global_sources = sc.boundary_sources, sc.global_sources
        self._defined_values, self._rms_scale = sc.defined_values, sc.rms_scale
        self.output_stats, self._scaled, self._fp32_head = sc.output_stats, sc.scaled, sc.fp32_head
        ### The task structure: a surface task predicts at the cells of the boundaries
        ### forward names and has no prediction-point parameters; an interior task
        ### predicts at prediction points that enter the encoder as tokens, passive
        ### readers of the boundary (the query-independent form).
        self.interior = bool(interior)
        if self.interior and self.n_spatial_dims not in (2, 3):
            raise NotImplementedError(
                "the interior task's built-in geometry (the exact distance to the boundary "
                f"surface) is implemented in two and three dimensions; got n_spatial_dims={self.n_spatial_dims}"
            )
        self.n_boundary_tokens = n_boundary_tokens
        ### The tunable knobs (the docstring). The boundary read's are the interior
        ### task's, so a surface task takes their defaults, as it does n_boundary_tokens'.
        self.tree_read_theta = tree_read_theta
        self.tree_read_leaf_size = _integer_knob("tree_read_leaf_size", tree_read_leaf_size)
        self.query_chunk_pairs = query_chunk_pairs
        self.mlp_ratio = _integer_knob("mlp_ratio", mlp_ratio)
        self.routing_temperature = _real_knob("routing_temperature", routing_temperature, positive=True)
        if not self.interior and self.tree_read_leaf_size != TREE_READ_LEAF_SIZE:
            raise ValueError(
                _READ_KNOB_ON_SURFACE.format(
                    name="tree_read_leaf_size", default=TREE_READ_LEAF_SIZE, value=self.tree_read_leaf_size
                )
            )
        ### Checkpoints store the constructor arguments as JSON. Recording the parsed
        ### schemas (a FieldSchema is a dict), merged with their ISLA-local keys, keeps
        ### the metadata JSON-safe whatever container the caller passed (a Hydra
        ### DictConfig, for instance).
        self._args["__args__"]["output_schema"] = merged(self.output_schema, out_extras) or None
        self._args["__args__"]["global_schema"] = merged(self.global_schema, g_extras) or None
        self._args["__args__"]["boundary_schemas"] = {
            name: merged(schema, b_extras[name]) for name, schema in self.boundary_schemas.items()
        }
        self._args["__args__"]["output_stats"] = self.output_stats
        ### The global DIRECTIONS (rank 1) and SCALARS (rank 0; module docstring,
        ### *Global inputs*). With rank-1 boundary sources the derived direction g is
        ### the last direction; a rank-0 global that defines a scale is consumed, not
        ### appended.
        self._global_vector_names = [
            n for n, s in self.global_schema.items() if s.rank == 1
        ]
        self._global_scalar_names = [
            n for n, s in self.global_schema.items() if s.rank == 0 and n not in self._defined_values.values()
        ]
        self._n_global_vectors = K = len(self._global_vector_names) + bool(self._boundary_sources)
        self._n_global_scalars = len(self._global_scalar_names)
        ### Each kind's columns in schema order (cached layouts, see _global_fields).
        self._global_vector_layout = _schema_layout({k: (self.n_spatial_dims,) for k in self._global_vector_names})
        self._global_scalar_layout = _schema_layout({k: () for k in self._global_scalar_names})
        ### The execution choices (module docstring, *Kernels*), resolved at the first
        ### forward pass (_resolve_geo_kernel, _resolve_geometry_kernel); reference_kernels
        ### sets them, and tests force one.
        self._geo_kernel = "auto"
        self._geometry_kernel = "auto"
        ### How much of their default saved activations the compiled blocks
        ### (compile_blocks) keep for the backward pass: below 1, torch.compile's
        ### partitioner recomputes the cheapest of them in the backward pass instead
        ### (torch.autograd.graph.region_activation_memory_budget). Read when the blocks
        ### are first compiled.
        self.block_activation_budget = 1.0
        self.hidden = _integer_knob("hidden", hidden)
        self.n_layers = _integer_knob("n_layers", n_layers)
        self.n_slices = _integer_knob("n_slices", n_slices)
        ### Each boundary condition's fields, each set with its own invariant layout
        ### (_InputSet). A condition's block vectors and tensors reach the slice
        ### layers through its own linear maps onto shared channels, as many as the
        ### widest condition has (_BoundaryCondition); every token carries them
        ### (zeros for the prediction points), so one geometry region serves every token.
        n_frame = 1 + K  # {n_hat, g_1..g_K}
        inputs = {
            name: _InputSet(schema, n_frame, dim=self.n_spatial_dims)
            for name, schema in self.boundary_schemas.items()
        }
        self._n_boundary_vecs = max(i.n_block_vecs for i in inputs.values())
        self._n_boundary_tens = max(i.n_block_tens for i in inputs.values())
        self._n_block_vecs = self._n_boundary_vecs
        self._n_block_tens = self._n_boundary_tens
        ### A condition's seed: [1, n.g_k (the K origin-free scalars of {n, g_k};
        ### positions enter through the anchors), its fields' invariants, the global
        ### scalars], mapped to the token width by its own input layer and then by the
        ### shared GELU and linear layer (embed). A boundary token carries its seed in
        ### its condition's columns of one block layout over every declared condition,
        ### zeros elsewhere, so one product with the concatenated input layers encodes
        ### a mixed set of tokens whose conditions the in-model draw decides, and every
        ### condition's parameters take part in every pass (DDP), present or not.
        self.boundary_conditions = nn.ModuleDict(
            {
                name: _BoundaryCondition(
                    inp,
                    K + inp.n_features + self._n_global_scalars,
                    hidden,
                    self._n_boundary_vecs,
                    self._n_boundary_tens,
                )
                for name, inp in inputs.items()
            }
        )
        self.boundary_names = tuple(self.boundary_schemas)
        self._condition_index = {name: i for i, name in enumerate(self.boundary_names)}
        self._seed_columns = self._offsets(c.seed.in_features for c in self.boundary_conditions.values())
        self._vec_columns = self._offsets(i.n_block_vecs for i in inputs.values())
        self._tens_columns = self._offsets(i.n_block_tens for i in inputs.values())
        self.embed = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList(
            _SliceBlock(
                hidden,
                n_slices,
                n_global_vectors=K,
                n_block_vecs=self._n_block_vecs,
                n_block_tens=self._n_block_tens,
                routing_temperature=self.routing_temperature,
                mlp_ratio=self.mlp_ratio,
            )
            for _ in range(n_layers)
        )
        ### Head frame: per-point soft slice anchor for the radial basis vector.
        ### Built only when an output of rank >= 1 reads the token basis: a
        ### scalar-only model never touches the radial direction, and a parameter
        ### that receives no gradient stops DDP at the second step.
        self._needs_frame = any(spec.rank >= 1 for spec in self.output_schema.values())
        self.frame_assign = (
            _RoutingLogits(hidden, n_slices, self.routing_temperature) if self._needs_frame else None
        )
        self.norm_out = nn.LayerNorm(hidden)
        ### One linear head emits every output's invariant coefficients, which are
        ### contracted with the token's basis vectors (_OutputLayout, _assemble_output;
        ### module docstring, *Outputs*). True vectors: g_1..g_K, n, rhat,
        ### e_th(rhat, n), e_th(rhat, g_k), so m = 2K+3. Pseudovectors: p = K + 2 in
        ### two dimensions, C(K+2, 2) in three; for d >= 4, odd-parity outputs are
        ### refused until a dataset needs them. For K = 1 in 3-D and outputs {scalar: 0, vector: 1} the head
        ### has width 1 + 5.
        self._n_true_basis = 3 + 2 * K
        odd_scalars = [n for n, sp in self.output_schema.items() if sp.rank == 0 and sp.parity == "odd"]
        if odd_scalars:
            raise NotImplementedError(
                f"odd-parity rank-0 outputs {odd_scalars!r}: pseudoscalar outputs are not implemented"
            )
        if self.n_spatial_dims == 3:
            self._n_pseudo_basis = (K + 2) * (K + 1) // 2
        elif self.n_spatial_dims == 2:
            self._n_pseudo_basis = K + 2
        else:
            self._n_pseudo_basis = 0
            odd = [n for n, sp in self.output_schema.items() if sp.parity == "odd"]
            if odd:
                raise NotImplementedError(
                    f"odd-parity outputs {odd!r} in n_spatial_dims={self.n_spatial_dims}: a pseudovector needs "
                    f"{self.n_spatial_dims - 1} vectors (the Levi-Civita construction); only two and "
                    "three dimensions are implemented"
                )
        self._output_layouts = {
            name: _OutputLayout(spec, self._n_true_basis, self._n_pseudo_basis, self.n_spatial_dims)
            for name, spec in self.output_schema.items()
        }
        self._n_head_coeffs = sum(
            layout.n_coeffs for layout in self._output_layouts.values()
        )
        ### The head's columns field by field in schema order, unpacked into the
        ### output fields' coefficients (a rank-0 field is its one coefficient).
        self._coeff_layout = _schema_layout(
            {n: (lay.n_coeffs,) if lay.spec.rank else () for n, lay in self._output_layouts.items()}
        )
        ### Each layout's coefficient index as a (non-persistent) buffer, so it
        ### follows the model's device instead of being copied there every pass.
        for i, layout in enumerate(self._output_layouts.values()):
            if layout.index is not None:
                self.register_buffer(f"_output_index_{i}", layout.index, persistent=False)
        self.head = nn.Linear(hidden, self._n_head_coeffs)
        ### Prediction-point tokens (interior=True): the same {n_q, g_k} invariant seeds
        ### as the boundary tokens, through the parameters of _QueryTokens; passive (the
        ### query-independent form). Registered after the boundary side, so its
        ### parameter order is a surface model's. A surface model has none of them, so
        ### every parameter it owns takes part in every forward pass.
        self._n_query_features = 2  # the built-in distance channels
        self.qt = (
            _QueryTokens(hidden, K + self._n_global_scalars + self._n_query_features)
            if self.interior
            else None
        )
        ### The interior task's boundary read (_tree_read.py): after every block the
        ### prediction points read every boundary cell through a Barnes-Hut forest.
        ### Registered after the blocks, the head and the prediction-point tokens, so
        ### those keep their places.
        ### _tree_read_kernel names the read's implementation, as _geo_kernel does the
        ### geometry region's: "reference" (the per-face cluster trees, GLOBE's walk,
        ### the kernel term a matmul read by _segment_softmax.py) or "auto", where the
        ### forest build (_tree_build.py), the walk (the library's Warp walk) and the read
        ### (_fused_read.py: the kernel term computed inside the read, nothing per pair
        ### kept but the basis) each take their fast path where their own requirements
        ### hold. reference_kernels sets it.
        self._tree_read_kernel = "auto"
        self.kernel_basis = None
        self.kernel_reads = None
        if self.interior:
            if self._n_block_tens:
                raise NotImplementedError(
                    "the interior task (interior=True) with rank-2 boundary inputs "
                    "(the boundary read's pair contractions are not implemented)"
                )
            n_geo = kernel_geo_width(K, self._n_boundary_vecs)
            self.kernel_basis = _KernelBasis(n_geo)
            self.kernel_reads = nn.ModuleList(_TreeKernelRead(hidden) for _ in range(n_layers))
        ### Per-condition quadrature (module docstring): with more than one declared
        ### condition, the learned log shares beta_k (canonical order; zeros, no RNG
        ### draw). Held by a submodule registered last, so they come after every other
        ### parameter in parameters() and every other parameter keeps its place. One
        ### declared condition has none: its shift is a constant, which cancels in every
        ### softmax over the tokens.
        self.condition_log_shares = None
        if len(self.boundary_names) > 1:
            self.condition_log_shares = nn.Module()
            self.condition_log_shares.beta = nn.Parameter(torch.zeros(len(self.boundary_names)))
        ### A state dict of another version fails with its version named on every
        ### loading path, not with a list of missing and unexpected keys.
        self.register_load_state_dict_pre_hook(_refuse_earlier_state_dict)
        ### The weights follow the schemas' declaration order, so a state dict records it
        ### and loads only where the fields it shares with the model keep their places.
        self.register_state_dict_post_hook(_record_field_order)
        self.register_load_state_dict_pre_hook(_refuse_reordered_fields)
        ### The trailing widths k of the per-token tensors: every small width (the
        ### input features, frames and vector stacks) and every layer's (see
        ### _hint_token_axis).
        self._token_widths = tuple(
            sorted(
                set(range(1, max(32, self._seed_columns[-1]) + 1))
                | {n_slices}
                | {f for m in self.modules() if isinstance(m, nn.Linear) for f in (m.in_features, m.out_features)}
                | {m.normalized_shape[-1] for m in self.modules() if isinstance(m, nn.LayerNorm)}
            )
        )

    @staticmethod
    def _offsets(widths) -> tuple[int, ...]:
        r"""The column offsets of consecutive blocks of the given ``widths``, one block
        per declared condition in canonical order: ``(0, w_0, w_0 + w_1, ..., total)``."""
        out = [0]
        for w in widths:
            out.append(out[-1] + int(w))
        return tuple(out)

    @staticmethod
    def _dots(x, g_hat):
        r"""x (B, N, d) against the K unit global directions g_hat (B, N, K, d) -> (B, N, K)
        (:func:`~physicsnemo.experimental.models.isla._inputs._contract_vectors`)."""
        return _contract_vectors(x[:, :, None], g_hat)[:, :, 0]

    @staticmethod
    def _in_block(x, cols: tuple[int, ...], i: int, dim: int):
        r"""``x`` placed in block ``i`` of the block layout ``cols`` along ``dim``
        (zeros in every other block)."""
        b, n = x.shape[:2]
        before = x.new_zeros(b, n, cols[i], *x.shape[dim + 1 :])
        after = x.new_zeros(b, n, cols[-1] - cols[i + 1], *x.shape[dim + 1 :])
        return torch.cat([before, x, after], dim)

    def _condition_tokens(
        self, name: str, points, normals, measure, fields: TensorDict, g_unit, g_scalars
    ) -> TensorDict:
        r"""One condition's boundary tokens over ``(B, n)``: their ``points``,
        ``normals`` and ``measure`` and, in the block layout
        over every declared condition (zeros in the other conditions' columns), the
        ``seed`` ``[1, n.g_k, the fields' invariants, global scalars]`` and the
        condition's block vectors ``vecs`` and rank-2 tensors ``tens`` (when any
        condition has them). ``fields`` holds the condition's declared fields over
        ``(B, n)``; ``g_unit`` ``(B, K, d)`` and ``g_scalars`` ``(B, S)`` or ``None``
        are the case's global inputs (:meth:`_global_fields`)."""
        i = self._condition_index[name]
        inputs = self.boundary_conditions[name].inputs
        b, n, d = points.shape
        n_hat = normals / normals.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        g_hat = g_unit[:, None].expand(b, n, -1, d).to(points.dtype)
        seed = [points.new_ones(b, n, 1), self._dots(n_hat, g_hat)]
        if inputs:
            feat, vecs, tens = inputs.features(fields, torch.cat([n_hat[:, :, None], g_hat], dim=2))
            seed.append(feat)
        if g_scalars is not None:
            seed.append(g_scalars[:, None, :].expand(b, n, -1).to(points.dtype))
        tokens = TensorDict(
            points=points,
            normals=normals,
            seed=self._in_block(torch.cat(seed, -1), self._seed_columns, i, 2),
            batch_size=[b, n],
        )
        tokens["measure"] = measure
        if self._vec_columns[-1]:
            if not inputs.n_block_vecs:
                vecs = points.new_zeros(b, n, 0, d)
            tokens["vecs"] = self._in_block(vecs, self._vec_columns, i, 2)
        if self._tens_columns[-1]:
            if not inputs.n_block_tens:
                tens = points.new_zeros(b, n, 0, d, d)
            tokens["tens"] = self._in_block(tens, self._tens_columns, i, 2)
        return tokens

    def _boundary_channels(self, boundary: TensorDict):
        r"""The boundary tokens' initial state and their shared vector and tensor
        channels: the block layout's seed through the conditions' concatenated input
        layers and the shared GELU and linear layer, and each condition's block
        vectors and tensors through its linear maps (``None`` where the model has
        none). Every declared condition's parameters enter the products."""
        conditions = list(self.boundary_conditions.values())
        weight = torch.cat([c.seed.weight for c in conditions], dim=1)
        h = self.embed(torch.nn.functional.linear(boundary["seed"], weight))
        vecs = tens = None
        with torch.autocast(h.device.type, enabled=False):
            if self._vec_columns[-1]:
                maps = [c.vector_map for c in conditions if c.vector_map is not None]
                m = torch.cat(maps, dim=1).to(boundary["vecs"].dtype)
                vecs = torch.einsum("vp,bnpi->bnvi", m, boundary["vecs"])
            if self._tens_columns[-1]:
                maps = [c.tensor_map for c in conditions if c.tensor_map is not None]
                m = torch.cat(maps, dim=1).to(boundary["tens"].dtype)
                tens = torch.einsum("vp,bnpij->bnvij", m, boundary["tens"])
        return h, vecs, tens

    def _block_extras(self, b_vecs, b_tens, b: int, n: int, like) -> dict:
        r"""The block-level stacks of one token class, as its token fields: the
        boundary channels ``vecs`` (B, N, P', d) and ``tens`` (B, N, Q, d, d);
        zeros for the prediction points, which carry none, and neither when no
        token carries one."""
        if not (self._n_block_vecs or self._n_block_tens):
            return {}
        d = self.n_spatial_dims
        if b_vecs is None:
            b_vecs = like.new_zeros(b, n, self._n_boundary_vecs, d)
        if b_tens is None:
            b_tens = like.new_zeros(b, n, self._n_boundary_tens, d, d)
        return {"vecs": b_vecs, "tens": b_tens}

    def _kernel_choice(self, name: str) -> str:
        r"""The execution-choice attribute ``name`` (``_geo_kernel``, ``_geometry_kernel`` or
        ``_tree_read_kernel``), a ValueError naming it unless it is one of its values
        (:data:`_KERNEL_CHOICES`)."""
        value, choices = getattr(self, name), _KERNEL_CHOICES[name]
        if value not in choices:
            raise ValueError(f"{name} must be one of {choices!r}, not {value!r}")
        return value

    @property
    def reference_kernels(self) -> bool:
        r"""Whether every region runs its reference path; ``False`` by default.

        Under ``False`` the geometry region of every slice layer, the search of the
        prediction points' boundary geometry and each piece of the interior task's
        tree read run their fast kernels where their requirements hold (CUDA and
        Triton among them), and their reference paths otherwise. ``True`` runs every
        one on its reference path: the eager geometry region, the reference search
        and the reference tree read. Both compute the same outputs up to summation
        order (module docstring, *Kernels*). A checkpoint does not store this
        execution choice, and a loaded model starts at ``False``. Second-order
        gradients need ``True``: the fast kernels' backward passes are not
        differentiable (a second ``backward()`` through one raises).

        Returns
        -------
        bool
            ``True`` when every region is set to its reference path.
        """
        return all(getattr(self, name) == path for name, path in _REFERENCE_KERNELS.items())

    @reference_kernels.setter
    def reference_kernels(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"reference_kernels must be True or False, not {value!r}")
        for name, path in _REFERENCE_KERNELS.items():
            setattr(self, name, path if value else "auto")

    def _resolve_geo_kernel(self, points: torch.Tensor) -> str:
        r"""The geometry-kernel implementation this forward pass will use.

        ``"eager"`` or ``"fused"`` when the ``_geo_kernel`` attribute forces it;
        under the default (``"auto"``) the fused kernel where
        :meth:`_geo_kernel_unsupported` finds no reason against it, else the eager
        region. The decision is cached per device, autocast dtype and parameter
        dtype, and logged once; on CUDA the eager region is a warning with its reason.
        """
        if self._kernel_choice("_geo_kernel") != "auto":
            return self._geo_kernel
        dev = points.device.type
        amp = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else None
        key = (dev, points.device.index, amp, self.head.weight.dtype)
        cached = getattr(self, "_geo_kernel_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        reason = self._geo_kernel_unsupported(points, amp)
        chosen = "eager" if reason else "fused"
        if is_tracing():
            ### Inside a traced region: no cache write and no logging (a logger
            ### call breaks fullgraph=True, a module write adds a recompile guard).
            return chosen
        self._geo_kernel_cache = (key, chosen)
        logger.info("ISLA geometry region resolved to %r on %s (%s)", chosen, points.device, reason or "supported")
        if points.is_cuda and reason:
            logger.warning("ISLA: the fused geometry kernel does not take this forward pass (%s); running the eager region.", reason)
        return chosen

    def _geo_kernel_unsupported(self, points: torch.Tensor, amp: torch.dtype | None) -> str | None:
        r"""Why the fused geometry kernel cannot run this forward pass, ``None`` when it
        can: it needs ``points`` on CUDA, Triton, CUDA autocast off or in bfloat16 (its
        only autocast dtype, and then float32 parameters: autocast leaves float64 ones
        as they are), at most three spatial dimensions, at most four per-point vectors
        (global directions, none counted as one, plus rank-1 token inputs), at most
        1024 slices and no rank-2 inputs. ``amp`` is the autocast dtype or ``None``."""
        from physicsnemo.experimental.models.isla._geo_kernel import (
            HAS_TRITON,
            MAX_DIRECTIONS,
            MAX_SLICES,
        )

        vectors = max(self._n_global_vectors, 1) + self._n_block_vecs
        reasons = (
            (not points.is_cuda, f"the points are on {points.device.type}, not CUDA"),
            (not HAS_TRITON, "Triton is not importable"),
            (amp not in (None, torch.bfloat16), f"{amp} autocast (the kernel's only autocast dtype is bfloat16)"),
            (amp is not None and self.head.weight.dtype == torch.float64, "float64 parameters under autocast"),
            (self.n_spatial_dims > 3, f"{self.n_spatial_dims} spatial dimensions (at most 3)"),
            (vectors > MAX_DIRECTIONS, f"{vectors} per-point vectors (at most {MAX_DIRECTIONS})"),
            (self.n_slices > MAX_SLICES, f"{self.n_slices} slices (at most {MAX_SLICES})"),
            (bool(self._n_block_tens), "rank-2 boundary inputs"),
        )
        return next((why for unsupported, why in reasons if unsupported), None)

    def _resolve_tree_read_kernel(self, points: torch.Tensor) -> str:
        r"""The interior task's boundary read implementation this forward pass will use.

        ``"reference"`` when the ``_tree_read_kernel`` attribute is ``"reference"``;
        under the default (``"auto"``) ``"fused"`` when ``points`` is on CUDA, not
        float64, and Triton is importable, else the reference read. Both compute the
        same read up to summation order. On CUDA the reference read sums its node means
        with atomics, so a float64 interior model on CUDA is not bitwise the same run
        to run. The decision is cached per device and dtype and logged once.
        """
        if self._kernel_choice("_tree_read_kernel") == "reference":
            return "reference"
        key = (points.device.type, points.device.index, points.dtype)
        cached = getattr(self, "_tree_read_kernel_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        ok = points.is_cuda and _fused_read.HAS_TRITON and points.dtype != torch.float64
        chosen = "fused" if ok else "reference"
        if is_tracing():
            return chosen
        self._tree_read_kernel_cache = (key, chosen)
        logger.info(
            "ISLA tree read resolved to %r on %s (cuda=%s, triton=%s, dtype=%s)",
            chosen,
            points.device,
            points.is_cuda,
            _fused_read.HAS_TRITON,
            points.dtype,
        )
        return chosen

    def _resolve_geometry_kernel(self, points: torch.Tensor) -> str:
        r"""The implementation of the boundary geometry's search this forward pass will
        use (``kernel`` of :func:`exact_boundary_geometry`).

        ``"reference"`` or ``"fast"`` when the ``_geometry_kernel`` attribute forces it;
        under the default (``"auto"``) the fast search when ``points`` is on CUDA and
        Triton is importable (its weld and canonical-cell kernels), else the
        reference. The decision is cached per device and logged once.
        """
        if self._kernel_choice("_geometry_kernel") != "auto":
            return self._geometry_kernel
        key = (points.device.type, points.device.index)
        cached = getattr(self, "_geometry_kernel_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        from physicsnemo.experimental.models.isla._geometry_fast import HAS_TRITON

        chosen = "fast" if points.is_cuda and HAS_TRITON else "reference"
        if is_tracing():
            return chosen
        self._geometry_kernel_cache = (key, chosen)
        logger.info(
            "ISLA boundary geometry search resolved to %r on %s (cuda=%s, triton=%s)", chosen, points.device, points.is_cuda, HAS_TRITON
        )
        return chosen

    def __getstate__(self):
        """The state without the compiled blocks, which hold this model's blocks: a
        deep copy or an unpickled model compiles its own at its first forward."""
        state = super().__getstate__()
        state["_block_forwards"] = state["_block_reads"] = None
        return state

    def _block_callables(self):
        r"""The slice blocks as callables: the modules themselves, or their forwards
        under :func:`torch.compile` when ``compile_blocks`` is set (built once)."""
        if not self.compile_blocks or is_tracing():
            return self.blocks
        if self._block_forwards is None:
            self._block_forwards = [self._compile_block(block.forward) for block in self.blocks]
        return self._block_forwards

    def _reads_compiled(self) -> bool:
        """Whether the boundary reads run their projections compiled (``compile_blocks``
        outside a whole-model compile)."""
        return self.compile_blocks and not is_tracing()

    def _block_read_callables(self):
        r"""The slice blocks' passive reads (:meth:`_SliceBlock.read`), compiled as
        :meth:`_block_callables` compiles their forwards."""
        if not self.compile_blocks or is_tracing():
            return [block.read for block in self.blocks]
        if self._block_reads is None:
            self._block_reads = [self._compile_block(block.read) for block in self.blocks]
        return self._block_reads

    def _compile_block(self, fn):
        r"""A block's forward or read under :func:`torch.compile` with the blocks'
        Inductor options and, when ``block_activation_budget`` is below 1, that
        activation memory budget for its graph."""
        budget = float(self.block_activation_budget)
        if not 0.0 < budget <= 1.0:
            raise ValueError(
                "block_activation_budget must be above 0 and at most 1; "
                f"got {self.block_activation_budget!r}"
            )
        if budget < 1.0:
            inner, warned = fn, self._budget_eager_warned

            def fn(*args, **kwargs):
                if not is_tracing():
                    ### Past Dynamo's recompile limit this frame runs eager, where the
                    ### budget's context raises and no partitioner saves anything:
                    ### say so once, when a backward pass can follow.
                    if torch.is_grad_enabled() and not warned:
                        warned.append(True)
                        logger.warning(
                            "ISLA: a block's %s runs outside torch.compile (Dynamo's recompile limit?), so"
                            " block_activation_budget=%s saves no memory there",
                            inner.__name__,
                            budget,
                        )
                    return inner(*args, **kwargs)
                with torch.autograd.graph.region_activation_memory_budget(budget):
                    return inner(*args, **kwargs)

            ### Dynamo counts a code object's graphs against its recompile limit, and
            ### every closure of one def shares its code object: the wrapper gets one
            ### copy per wrapped function (all blocks' forwards, all blocks' reads), as
            ### the blocks share the functions' own code objects without the budget.
            fn.__code__ = _BUDGET_CODES.setdefault(inner.__func__, fn.__code__.replace())
        return torch.compile(fn, options=_BLOCK_COMPILE_OPTIONS)

    def compile(self, *args, **kwargs) -> None:
        r"""Compile the whole model in place, as :meth:`torch.nn.Module.compile`, with
        Inductor's mix-order reduction off.

        **Use this, not** ``torch.compile(model)``. On torch 2.13 Inductor's
        default ``triton.mix_order_reduction`` (pytorch/pytorch#195845) makes the slice
        blocks compute a different function from the eager model wherever they run
        the eager geometry region (on CUDA wherever ``_geo_kernel_unsupported`` gives a
        reason, such as rank-2 boundary inputs or float16 autocast, or with
        ``model.reference_kernels = True``), and nothing warns: the model cannot see the options it is compiled with.
        ``torch.compile(model, options={"triton.mix_order_reduction": False, ...})`` is
        the same as this.

        The arguments are those of :func:`torch.compile`. With the Inductor backend
        (named, or the default one when ``backend`` is absent or ``None``) a ``mode``
        is passed as its options (``torch._inductor.list_mode_options``) and the
        option is added to ``options`` unless they set it; another backend gets the
        arguments as given.

        Parameters
        ----------
        *args, **kwargs
            The arguments of ``torch.compile``, handled as above.

        Returns
        -------
        None
            The model is compiled in place.
        """
        from torch._dynamo.backends.registry import get_default_backend

        if (kwargs.get("backend") or get_default_backend()) == "inductor":
            from torch._inductor import list_mode_options

            mode = kwargs.pop("mode", None)
            kwargs["options"] = {
                **(list_mode_options(mode) if mode else {}),
                **_BLOCK_COMPILE_OPTIONS,
                **(kwargs.get("options") or {}),
            }
        super().compile(*args, **kwargs)

    def _global_rows(self, data: TensorDict, like) -> TensorDict:
        r"""The declared global fields of ``data`` as one case's row, in ``like``'s
        dtype at their declared shapes."""
        rows = TensorDict(batch_size=[1])
        for name, spec in self.global_schema.items():
            shape, width = spec.shape(self.n_spatial_dims), spec.numel(self.n_spatial_dims)
            t = data.get(FieldSchema.key(name))
            if not is_tracing():
                if not isinstance(t, torch.Tensor) or t.numel() % width:
                    raise ValueError(
                        f"global_data[{name!r}]: expected {width} values per case, got "
                        f"{tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__}"
                    )
                if t.numel() != width:
                    raise ValueError(
                        f"global_data[{name!r}] gives {t.numel() // width} rows for a batch of 1"
                    )
            rows.set(FieldSchema.key(name), t.reshape(1, *shape).to(like.dtype))
        return rows

    def _global_fields(self, global_rows: TensorDict, like, scales: dict | None = None, g_dir=None):
        r"""The unit global directions (B, K, d) and the global scalars (B, S)
        (``None`` when none are declared) from the global fields over the batch
        (:meth:`_global_rows`), each kind's columns in schema order; with
        ``scales`` (:meth:`_case_scales`) the rank-0 globals divided by their
        units' scale, and the derived direction ``g_dir``, if any, appended to the
        directions."""
        if self._global_vector_names:
            gv = _pack(global_rows, self._global_vector_layout).unflatten(-1, (-1, self.n_spatial_dims))
            ### Each direction is normalized; the magnitude, if it means anything,
            ### is a rank-0 global input.
            ### (A floor of the dtype's smallest normal number: it guards a zero vector
            ### only, whatever the global's units.)
            g_unit = gv / gv.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(gv.dtype).tiny)  # (B, K, d)
        else:
            g_unit = like.new_zeros(global_rows.batch_size[0], 0, self.n_spatial_dims)
        g_scalars = _pack(global_rows, self._global_scalar_layout) if self._global_scalar_names else None
        if scales is None:
            return g_unit, g_scalars
        if self._global_units or self._global_scale:
            g_scalars = g_scalars / torch.stack(
                [
                    self._input_factor(self._global_units.get(n), self._global_scale.get(n), scales).to(like.dtype)
                    for n in self._global_scalar_names
                ]
            )
        if g_dir is not None:
            g_unit = torch.cat([g_unit, g_dir.to(g_unit.dtype).expand(g_unit.shape[0], 1, -1)], 1)
        return g_unit, g_scalars

    @staticmethod
    def _gauge(meshes, measures) -> torch.Tensor:
        r"""The length unit ``(1, 1, 1)`` of the given boundary meshes: the RMS pairwise
        distance of their piecewise-flat cells, each cell with its centroid, measure and
        own second moment (:func:`~physicsnemo.experimental.models.isla._scales.length_unit`,
        which :func:`~physicsnemo.experimental.models.isla.case_scales` shares)."""
        return length_unit(meshes, measures)

    def _case_scales(self, names, meshes, measures, global_rows: TensorDict, gauge, tests: list) -> tuple:
        r"""The case's scales in float64 from its full-resolution boundary and its
        globals, before the token draw, by name: ``L`` (the gauge), the scale the
        rank-1 fields define (the measure-weighted RMS magnitude pooled over the
        boundary sources, or the global source's magnitude) and each scale a rank-0
        global defines (its value); and apart from them, so that no scale's name
        can collide with it, the derived direction ``g`` ``(1, 1, d)`` (``None``
        unless the sources are boundary fields). An absent source is refused
        here, in eager mode and under torch.compile alike (it is structural); a zero
        source or a non-positive value in eager mode, as flags on the device with
        their messages, appended to ``tests`` for :meth:`_case_inputs` to read. Under
        torch.compile these data-dependent checks are skipped (as every input check
        of ``forward`` is), so a compiled model reads such a case as NaN or inf."""
        check = not is_tracing()
        scales, g_dir = {"L": gauge.reshape(()).double()}, None
        if self._boundary_sources:
            present = dict(zip(names, zip(meshes, measures)))
            s0, s2, s1 = rms_moments(
                (present[c][0].cell_data.get(FieldSchema.key(n)), present[c][1])
                for c, n in self._boundary_sources
                if c in present
            )
            absent = (
                f"the sources {self._boundary_sources!r} of the scale {self._rms_scale!r} are absent from this "
                "case, have no cells or are zero everywhere: the scale is undefined"
            )
            if not isinstance(s0, torch.Tensor):
                raise ValueError(absent)
            norm = s1.norm()
            if check:
                tests += [
                    ((s0 > 0) & (s2 > 0), lambda: absent),
                    (
                        norm > 0,
                        lambda: f"the sources {self._boundary_sources!r} of the scale {self._rms_scale!r} cancel "
                        "(sum of a v is zero): the direction g is undefined",
                    ),
                ]
            scales[self._rms_scale] = (s2 / s0).sqrt()
            g_dir = (s1 / norm)[None, None]
        elif self._global_sources:
            v = global_rows.get(FieldSchema.key(self._global_sources[0]))[0].double()
            scales[self._rms_scale] = v.norm()
            if check:
                tests.append(
                    (
                        scales[self._rms_scale] > 0,
                        lambda: f"the source {self._global_sources[0]!r} of the scale {self._rms_scale!r} is zero: "
                        "the scale is undefined",
                    )
                )
        for key, n in self._defined_values.items():
            scales[key] = global_rows.get(FieldSchema.key(n))[0].double().reshape(())
            if check:
                tests.append(
                    (
                        scales[key] > 0,
                        lambda key=key, n=n: f"the scale {key!r} ({n!r}) must be positive; got {scales[key].item()}",
                    )
                )
        return scales, g_dir

    @staticmethod
    def _input_factor(units, c, scales) -> torch.Tensor:
        r"""An input's divisor: its units' scale times its constant ``scale``."""
        factor = scale_factor(units, scales)
        if factor is None:
            factor = scales["L"].new_ones(())
        return factor if c is None else factor * c

    def _nondimensional(self, condition: str, fields: TensorDict, scales: dict | None) -> TensorDict:
        r"""A condition's declared fields, each divided by its units' scale and then by
        its constant ``scale``."""
        if scales is None:
            return fields
        units, cs = self._boundary_units.get(condition, {}), self._boundary_scale.get(condition, {})
        names = [n for n in self.boundary_schemas[condition] if n in units or n in cs]
        if not names:
            return fields
        fields = fields.clone(False)
        for name in names:
            key = FieldSchema.key(name)
            t = fields.get(key)
            fields.set(key, t / self._input_factor(units.get(name), cs.get(name), scales).to(t.dtype))
        return fields

    def _redimensionalize(self, fields: TensorDict, scales: dict | None) -> TensorDict:
        r"""The outputs times their constant ``scale`` and their units' scale, then
        standardized by ``output_stats``, in the outputs' precision (the head's, see
        :meth:`_head`)."""
        for name, spec in self.output_schema.items():
            factor = scale_factor(self._output_units.get(name), scales) if scales is not None else None
            c = self._output_scale.get(name)
            stats = (self.output_stats or {}).get(name)
            if factor is None and c is None and stats is None:
                continue
            key = FieldSchema.key(name)
            y = fields.get(key)
            if c is not None:
                y = y * c
            if factor is not None:
                y = y * factor.to(y.dtype)
            if stats is not None:
                y = (y - stats["mean"]) / stats["std"] if spec.rank == 0 else y / stats["std"]
            fields.set(key, y)
        return fields

    ### -----------------------------------------------------------------------
    ### The mesh API
    ### -----------------------------------------------------------------------

    def _check_fields(self, specs, data: TensorDict, n: int, label: str) -> None:
        r"""Every declared field is a leaf of ``data`` (by dotted name) with the
        shape its rank prescribes over ``n`` entities; extra leaves are ignored."""
        missing = _missing_fields(specs, data)
        if missing:
            raise ValueError(f"{label} is missing the declared fields {missing!r}")
        for name, spec in specs.items():
            t = data.get(FieldSchema.key(name))
            expected = (n, *spec.shape(self.n_spatial_dims))
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"{label}[{name!r}]: expected shape {expected} (rank {spec.rank}), "
                    f"got {tuple(t.shape)}"
                )

    def _validate_meshes(self, boundary_meshes, prediction_points, prediction_boundaries) -> None:
        r"""Mesh-level checks for :meth:`forward`: dimensions, declared fields
        and the task structure (``interior``, where the outputs are)."""
        d = self.n_spatial_dims
        if not isinstance(boundary_meshes, Mapping) or not boundary_meshes:
            raise ValueError("boundary_meshes must be a non-empty mapping of name to Mesh")
        unknown = sorted(str(name) for name in boundary_meshes if name not in self.boundary_schemas)
        if unknown:
            raise ValueError(
                f"boundary_meshes names {unknown!r}, which are not declared boundary conditions; "
                f"this model declares {list(self.boundary_names)!r} (boundary_schemas)"
            )
        for name, mesh in boundary_meshes.items():
            if not isinstance(mesh, Mesh):
                raise TypeError(
                    f"boundary_meshes[{name!r}] must be a physicsnemo.mesh.Mesh, got "
                    f"{type(mesh).__name__}"
                )
            if mesh.n_spatial_dims != d:
                raise ValueError(
                    f"boundary_meshes[{name!r}] has {mesh.n_spatial_dims} spatial dimensions; "
                    f"this model has n_spatial_dims={d}"
                )
            if mesh.n_manifold_dims != d - 1:
                raise ValueError(
                    f"boundary_meshes[{name!r}]: the boundary of a {d}-dimensional domain is a "
                    f"Mesh[{d - 1}, {d}] (its cells carry the normals); got a "
                    f"Mesh[{mesh.n_manifold_dims}, {d}]"
                )
            self._check_dtype(mesh.points.dtype, f"boundary_meshes[{name!r}]")
            self._check_fields(
                self.boundary_schemas[name],
                mesh.cell_data,
                mesh.n_cells,
                f"boundary_meshes[{name!r}].cell_data",
            )
        if prediction_points is None:
            if self.interior:
                raise ValueError(
                    "this model was built with interior=True and predicts at prediction "
                    "points; pass prediction_points (a point-cloud Mesh or an (N, n_spatial_dims) "
                    "tensor)"
                )
            self._validate_prediction_boundaries(prediction_boundaries, boundary_meshes)
            return
        if not self.interior:
            raise ValueError(
                "this model was built with interior=False (a surface task: predictions at "
                "the boundary cells), so prediction_points must be None; build with "
                "interior=True for an interior task"
            )
        if prediction_boundaries is not None:
            raise ValueError(
                "this model was built with interior=True and predicts at prediction_points, "
                "so prediction_boundaries must be None (a surface task names the boundaries "
                "it predicts on)"
            )
        self._validate_point_set(prediction_points, "prediction_points")
        if not any(mesh.n_cells for mesh in boundary_meshes.values()):
            raise ValueError(
                "boundary_meshes have no cells: the interior task's prediction points "
                "measure their distance to the boundary surface"
            )

    def _validate_prediction_boundaries(self, names, boundary_meshes) -> None:
        r"""A surface task's prediction boundaries: a non-empty sequence of distinct
        names of given boundaries (not a bare string)."""
        if names is None:
            raise ValueError(
                "this model was built with interior=False (a surface task) and predicts at "
                "the cells of named boundaries; pass prediction_boundaries, the names of "
                f"the given boundaries that carry the outputs (given: {sorted(boundary_meshes)!r})"
            )
        if isinstance(names, str) or not isinstance(names, (list, tuple)) or not names:
            raise ValueError(
                "prediction_boundaries must be a non-empty list or tuple of boundary names; "
                f"got {names!r}"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"prediction_boundaries names a boundary twice: {list(names)!r}")
        absent = [n for n in names if n not in boundary_meshes]
        if absent:
            raise ValueError(
                f"prediction_boundaries names {absent!r}, which are not in boundary_meshes "
                f"({sorted(boundary_meshes)!r}); the outputs are predicted at given cells"
            )
        if not any(boundary_meshes[n].n_cells for n in names):
            raise ValueError(
                f"prediction_boundaries {list(names)!r} have no cells, so there is nothing to "
                "predict at: name at least one boundary with cells"
            )

    def _check_dtype(self, dtype: torch.dtype, label: str, promoted: bool = False) -> None:
        r"""An input's points in the parameters' dtype (autocast changes neither), or with
        ``promoted`` in any dtype that promotes to it (narrower points are promoted)."""
        want = self.head.weight.dtype
        if (torch.promote_types(dtype, want) if promoted else dtype) != want:
            slower = ", which on CUDA runs the slower reference boundary read" if self.interior and dtype == torch.float64 else ""
            raise ValueError(
                f"{label} has {dtype} points and this model's parameters are {want}: "
                f"cast the input (.to({want})) or the model (model.to({dtype}){slower})"
            )

    def _validate_point_set(self, pts, label: str) -> None:
        r"""An interior point set of at least one point: a point-cloud Mesh (its
        ``points``) or a bare ``(N, n_spatial_dims)`` tensor."""
        d = self.n_spatial_dims
        if isinstance(pts, Mesh):
            if pts.n_spatial_dims != d:
                raise ValueError(
                    f"{label} has {pts.n_spatial_dims} spatial "
                    f"dimensions; this model has n_spatial_dims={d}"
                )
        elif not isinstance(pts, torch.Tensor) or pts.ndim != 2 or pts.shape[-1] != d:
            raise ValueError(
                f"{label} must be a point-cloud Mesh or a tensor of shape (N, {d}); "
                f"got {tuple(pts.shape) if isinstance(pts, torch.Tensor) else type(pts).__name__}"
            )
        dtype = (pts.points if isinstance(pts, Mesh) else pts).dtype
        if not dtype.is_floating_point:
            raise TypeError(f"{label} must have a floating-point dtype; got {dtype}: cast the points")
        self._check_dtype(dtype, label, promoted=True)
        if (pts.n_points if isinstance(pts, Mesh) else pts.shape[0]) == 0:
            raise ValueError(f"{label} has no points; pass at least one")

    def _select_global_data(self, global_data, device) -> TensorDict:
        r"""The declared leaves of ``global_data`` as a TensorDict (extra leaves
        dropped, a missing one an error); this is also what the output Mesh carries."""
        if global_data is None:
            td = TensorDict({}, batch_size=[], device=device)
        elif isinstance(global_data, TensorDict):
            td = global_data
        elif isinstance(global_data, Mapping):
            ### A plain mapping may name its leaves with dotted strings; TensorDict
            ### takes those as nested keys only in tuple form.
            td = TensorDict({}, batch_size=[], device=device)
            for name, value in global_data.items():
                td.set(FieldSchema.key(name) if isinstance(name, str) else name, value)
        else:
            raise TypeError(
                f"global_data must be a TensorDict or a mapping of name to tensor, got "
                f"{type(global_data).__name__}"
            )
        ### Skipped under torch.compile, where listing the leaves would guard on
        ### their order (the strict select below still fails on a missing leaf).
        if not is_tracing():
            missing = _missing_fields(self.global_schema, td)
            if missing:
                raise ValueError(f"global_data is missing the declared fields {missing!r}")
        return td.select(*(FieldSchema.key(name) for name in self.global_schema))

    def forward(
        self,
        *,
        boundary_meshes: Mapping[str, Mesh],
        prediction_points: Mesh | Float[torch.Tensor, "n_points n_spatial_dims"] | None = None,
        prediction_boundaries: list[str] | tuple[str, ...] | None = None,
        global_data: TensorDict | Mapping[str, torch.Tensor] | None = None,
    ) -> Mesh:
        r"""Predict the declared output fields for one case.

        Both tasks read the whole boundary in ``boundary_meshes``. The outputs are
        at ``prediction_points`` for an interior task and at
        ``prediction_boundaries`` for a surface task; pass exactly the one the task
        requires. The inputs pose one boundary-value problem, so they should
        determine its solution: every condition with its prescribed values, the
        boundaries at full resolution and oriented alike (class docstring,
        *Setting up a new problem*). The input checks run in eager mode only;
        under :func:`torch.compile` a malformed case is not refused.

        The class docstring's *Forward* and *Outputs* sections document the inputs
        and the output.
        """
        c = self._case_inputs(boundary_meshes, prediction_points, global_data, prediction_boundaries)
        if c.query is None:
            fields = self._redimensionalize(
                self._forward_tokens(c.boundary, c.g_unit, c.g_scalars, gauge=c.gauge, out_rows=c.out_rows), c.scales
            )
        else:
            ### The interior task: the training step's path (query_chunks) without its leaves, the
            ### prediction points in chunks of at most query_chunk_pairs walk pairs.
            chunks = QueryChunks(self, c, detach=False)
            spans = chunks.spans(self.query_chunk_pairs)
            if len(spans) > 1 and not self._forward_chunks_logged:
                self._forward_chunks_logged = True
                logger.info(
                    "ISLA forward runs %d prediction points in %d chunks of at most %d walk pairs",
                    chunks.n_points, len(spans), self.query_chunk_pairs,
                )  # fmt: skip
            parts = [chunks._fields(a, b) for a, b in spans]
            fields = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        return Mesh(
            points=c.out_points,
            point_data=fields[0].to(c.out_points.device),
            global_data=c.global_td,
        )

    def query_chunks(
        self,
        *,
        boundary_meshes: Mapping[str, Mesh],
        prediction_points: Mesh | Float[torch.Tensor, "n_points n_spatial_dims"],
        global_data: TensorDict | Mapping[str, torch.Tensor] | None = None,
    ) -> "QueryChunks":
        r"""The interior task's ``forward`` in chunks of prediction points, for a training
        step that holds the activations of one chunk of points at a time.

        It takes ``forward``'s arguments and runs the boundary side of the forward
        pass once: the case's inputs (the token draw included, which consumes the
        global RNG as ``forward`` does), the boundary tokens through every block,
        and the forest, which each chunk walks for its own points. What the
        prediction points read of each block (its prototypes, anchors and slice
        states, and the boundary read's keys and values) and the head frame's
        prototypes and anchors become leaves that accumulate gradients.

        Call the result ``chunks``, a :class:`QueryChunks`, as
        ``chunks(start, stop)`` for the output Mesh of
        ``prediction_points[start:stop]``. It equals those rows of ``forward`` up
        to summation order (each point reads exactly what it reads among all of
        them), and is differentiable in the parameters and the leaves. After the
        chunks' backward passes, ``chunks.backward()`` carries the leaves'
        gradients through the boundary side once. The parameters' gradients are
        then those of the unchunked step whose loss is the sum of the chunks'
        (``chunks.n_points`` points in all), and so are the boundary points' and
        the global data's where they require gradients. The prediction points stay
        in the caller's graph, so a chunk's backward pass frees any graph upstream
        of them: pass them as a leaf, or call each chunk's backward with
        ``retain_graph=True`` when that graph must survive.

        Each chunk computes its own points' geometry (their signed distance and
        direction), so the geometry's memory grows with the chunk, not with all
        the points. The geometry's boundary-side setup (the weld and the search's
        trees) is repeated per chunk. Interior task only: a surface model has no
        prediction points to chunk.

        Parameters
        ----------
        boundary_meshes : Mapping[str, Mesh]
            The boundary, as in ``forward`` (class docstring, *Forward*).
        prediction_points : Mesh or torch.Tensor
            The prediction points, a point-cloud ``Mesh`` or a tensor of shape
            :math:`(N, d)`, as in ``forward``.
        global_data : TensorDict or Mapping[str, torch.Tensor], optional, default=None
            The case's per-case fields, as in ``forward``.

        Returns
        -------
        QueryChunks
            The boundary side's result, called per chunk of prediction points.

        Raises
        ------
        ValueError
            If the model is a surface model (``interior=False``).
        """
        if not self.interior:
            raise ValueError(
                "query_chunks needs an interior model (interior=True): a surface model has "
                "no prediction points to chunk"
            )
        return QueryChunks(self, self._case_inputs(boundary_meshes, prediction_points, global_data))

    def _case_inputs(
        self, boundary_meshes, prediction_points, global_data, prediction_boundaries=None
    ) -> "_CaseInputs":
        r"""One case's inputs to the network from :meth:`forward`'s
        (:class:`_CaseInputs`: the surface task's :meth:`_forward_tokens`, the
        interior task's :class:`QueryChunks`)."""
        ### Validation at the API boundary (skipped under torch.compile): fail
        ### here with the offending mesh, not inside the network.
        if not is_tracing():
            self._validate_meshes(boundary_meshes, prediction_points, prediction_boundaries)
        ### The given conditions in canonical order.
        names = [name for name in self.boundary_names if name in boundary_meshes]
        meshes = [boundary_meshes[name] for name in names]
        ### A surface task's outputs: the token rows of its prediction boundaries, as
        ### (start, stop) ranges in token order, adjacent conditions' ranges merged;
        ### None when every given boundary is one, so that the head reads the token
        ### state itself (decided by name: a test on the counts would guard on them).
        out_rows = None
        if prediction_points is None and not set(names) <= set(prediction_boundaries):
            out_rows, start = [], 0
            for name, m in zip(names, meshes):
                if name in prediction_boundaries:
                    if out_rows and out_rows[-1][1] == start:
                        out_rows[-1] = (out_rows[-1][0], start + m.n_cells)
                    else:
                        out_rows.append((start, start + m.n_cells))
                start += m.n_cells
        measures = [cell_measures(m) for m in meshes]
        ### The case's data-dependent tests, in order, each a flag on the device with
        ### its message. They are read together once the tokens are built (below): one
        ### transfer to the host, and none while the device still runs what was queued
        ### before this call. Nothing in between raises on the values they test.
        tests: list = []
        if not is_tracing():
            for name, m, w in zip(names, meshes, measures):
                if m.n_cells:
                    tests.append(
                        (
                            torch.isfinite(m.points).all(),
                            lambda name=name: f"boundary_meshes[{name!r}] has a non-finite point coordinate: "
                            "repair or remove its points before the forward pass",
                        )
                    )
                    tests.append(
                        (
                            (torch.isfinite(w) & (w >= 0)).all(),
                            lambda name=name: f"boundary_meshes[{name!r}] has a negative or non-finite cell measure "
                            f"(its cells' areas, or cell_data[{EFFECTIVE_MEASURE_KEY!r}]): a quadrature weight "
                            "is finite and at least 0, so repair its cells or the measures given",
                        )
                    )
                    tests.append(
                        (
                            w.sum() > 0,
                            lambda name=name: f"boundary_meshes[{name!r}] has cells but no measure (their areas, or "
                            f"cell_data[{EFFECTIVE_MEASURE_KEY!r}], sum to zero): the boundary's "
                            "quadrature weights are undefined",
                        )
                    )
                    for field in self.boundary_schemas[name]:
                        tests.append(
                            (
                                torch.isfinite(m.cell_data.get(FieldSchema.key(field))).all(),
                                lambda name=name, field=field: f"boundary_meshes[{name!r}].cell_data[{field!r}] has a "
                                "non-finite value: repair or remove its cells before the forward pass",
                            )
                        )
        like = meshes[0].points
        global_td = self._select_global_data(global_data, like.device)
        global_rows = self._global_rows(global_td, like)
        if not is_tracing():
            for name in self.global_schema:
                tests.append(
                    (
                        torch.isfinite(global_rows.get(FieldSchema.key(name))).all(),
                        lambda name=name: f"global_data[{name!r}] has a non-finite value in this model's dtype",
                    )
                )
        ### The length unit (_gauge), from the full-resolution boundary before the draw;
        ### with declared scales, the case's other scales (_case_scales).
        gauge = self._gauge(meshes, measures)
        scales, g_dir = (
            self._case_scales(names, meshes, measures, global_rows, gauge, tests) if self._scaled else (None, None)
        )
        g_unit, g_scalars = self._global_fields(global_rows, like, scales, g_dir)
        ### The boundary cells, each condition's with its own inputs, concatenated in
        ### canonical order over a batch axis of one.
        boundary = torch.cat(
            [
                self._condition_tokens(
                    name,
                    m.cell_centroids[None],
                    m.cell_normals[None],
                    w[None],
                    self._nondimensional(
                        name, m.cell_data.select(*(FieldSchema.key(k) for k in self.boundary_schemas[name])), scales
                    )[None],
                    g_unit,
                    g_scalars,
                )
                for name, m, w in zip(names, meshes, measures)
            ],
            dim=1,
        )
        ### With more than one declared condition, each cell's condition (canonical
        ### order, so an absent condition keeps its number) and the log of its
        ### condition's full-resolution measure (float64, before the draw).
        if len(self.boundary_names) > 1:
            boundary["condition"] = torch.cat(
                [torch.full((1, m.n_cells), self._condition_index[name], device=like.device) for name, m in zip(names, meshes)],
                dim=1,
            )
            boundary["condition_log_measure"] = torch.cat(
                [torch.log(w.double().sum()).expand(1, m.n_cells) for m, w in zip(meshes, measures)], dim=1
            )
        if prediction_points is not None and not is_tracing():
            pts = prediction_points.points if isinstance(prediction_points, Mesh) else prediction_points
            tests.append(
                (
                    torch.isfinite(pts).all().to(like.device),  # read with the boundary's flags
                    lambda: "prediction_points has a non-finite coordinate: pass finite points",
                )
            )
        if tests:
            for ok, (_, message) in zip(torch.stack([flag for flag, _ in tests]).tolist(), tests):
                if not ok:
                    raise ValueError(message())
        query = surface = None
        if prediction_points is None:
            out_points = boundary["points"][0]
            if out_rows is not None:
                out_points = torch.cat([out_points[a:b] for a, b in out_rows])
        else:
            query = self._point_set(prediction_points)
            out_points = query["points"]
            ### The boundary surface the points' geometry is measured from (each
            ### chunk's, QueryChunks._fields): the union of the named meshes' cells
            ### at full resolution, at the parameters' precision.
            q_points = query["points"]
            with torch.autocast(q_points.device.type, enabled=False):
                offsets = [0]
                for m in meshes[:-1]:
                    offsets.append(offsets[-1] + m.n_points)
                points = torch.cat([m.points.to(q_points.device, q_points.dtype) for m in meshes])
                cells = torch.cat([m.cells.to(q_points.device) + o for m, o in zip(meshes, offsets)])
            surface = (points, cells)
            ### The boundary tokens: the sample of the cells (sample_boundary_cells),
            ### each kept cell with its Horvitz-Thompson measure. A uniform start
            ### from the global torch RNG in training, a fixed one in evaluation; in
            ### float64 from the input meshes whatever the model's precision or the
            ### autocast, the weights cast to the tokens' measures at the end.
            if self.training:
                start = torch.rand((), dtype=torch.float64, device=q_points.device)
            else:
                start = torch.full((), _EVAL_START, dtype=torch.float64, device=q_points.device)
            with torch.autocast(q_points.device.type, enabled=False):
                draw = [_draw_inputs(m) for m in meshes]
                ### The draw's measures take the gradient of the meshes' own (w - w.detach()
                ### is exactly 0), so the weights carry it in the vertices.
                kept, weights = sample_boundary_cells(
                    [a + (w - w.detach()).double() for (a, _), w in zip(draw, measures)],
                    [x for _, x in draw],
                    self.n_boundary_tokens,
                    start,
                )
            boundary = boundary.apply(lambda t: t[:, kept], batch_size=[1, kept.shape[0]])
            boundary["measure"] = weights[None].to(boundary["measure"].dtype)
            if self.kernel_reads is not None:
                ### The forest's orientation faces read each kept cell's normal with the
                ### bound on its rounding (tree_read.normal_error_bound), from the meshes'
                ### vertices as given, measured from its condition's tokens' weighted
                ### centroid, as the forest builds one set of trees per condition.
                ### (cells: the union's, from the surface above.)
                with torch.no_grad(), torch.autocast(q_points.device.type, enabled=False):
                    verts = torch.cat([m.points.to(q_points.device) for m in meshes])
                    simplex = verts[cells[kept]]
                    w = weights.double()
                    if "condition" in boundary.keys():
                        ### Each condition's centroid by a one-hot mask over the declared
                        ### conditions (no data-dependent shapes, so it compiles).
                        cond = boundary["condition"][0]
                        mask = (cond[:, None] == torch.arange(len(self.boundary_names), device=cond.device)).double()
                        sums = (mask[:, :, None] * (w[:, None] * simplex.double().mean(1))[:, None]).sum(0)
                        centre = (sums / (mask * w[:, None]).sum(0)[:, None])[cond]
                    else:
                        centre = (w[:, None] * simplex.double().mean(1)).sum(0) / w.sum()
                    boundary["normal_tol"] = normal_error_bound(simplex, centre)[None]
            query = query[None]
        return _CaseInputs(boundary, query, surface, g_unit, g_scalars, gauge, scales, out_points, global_td, out_rows)

    def _point_set(self, pts) -> TensorDict:
        r"""An interior point set's tokens: the points of a point-cloud Mesh or a bare
        tensor, promoted to the parameters' dtype."""
        points = (pts.points if isinstance(pts, Mesh) else pts).to(self.head.weight.dtype)
        return TensorDict(points=points, batch_size=points.shape[:1])

    def _log_shift(self, boundary: TensorDict, dtype: torch.dtype) -> torch.Tensor | None:
        r"""Each boundary token's log share shift ``-log A_k + beta_k`` ``(B, N)`` in
        ``dtype`` (the per-condition quadrature, module docstring), ``None`` for a model
        of one declared condition. beta_k is a one-hot mask of the token's condition
        times beta, summed: exactly beta_k, with a backward of a product and a sum (no
        scatter: Inductor's CPU codegen fails on an indexing's atomic-add backward), and
        an absent condition's share takes a zero gradient."""
        if self.condition_log_shares is None:
            return None
        beta = self.condition_log_shares.beta.to(dtype)
        one_hot = boundary["condition"][..., None] == torch.arange(beta.shape[0], device=beta.device)
        return (-boundary["condition_log_measure"]).to(dtype) + (one_hot.to(dtype) * beta).sum(-1)

    def _condition_shift(self, boundary: TensorDict, dtype: torch.dtype) -> torch.Tensor:
        r"""Each declared condition's log share shift ``-log A_k + beta_k`` ``(B, C)`` in
        ``dtype``, for the interior task's tree nodes (:func:`walk_range_pairs`): ``-log A_k`` that of
        the condition's tokens, cast as :meth:`_log_shift` casts it (so a node of one
        condition takes exactly its cells' shift), zero for a condition without tokens
        (no node holds it)."""
        beta = self.condition_log_shares.beta.to(dtype)
        with torch.no_grad():
            one_hot = boundary["condition"][..., None] == torch.arange(beta.shape[0], device=beta.device)
            neg = (-boundary["condition_log_measure"]).to(dtype)[..., None]
            neg = torch.where(one_hot, neg, -math.inf).amax(1)
            neg = torch.where(one_hot.any(1), neg, 0.0)
        return neg + beta

    def _tree_read_walk(
        self,
        bnd: TensorDict,
        q_r,
        g_unit,
        normal_tol=None,
        conditions=None,
    ) -> TreeReadWalk:
        r"""The boundary read's forest, its points not yet walked
        (:func:`~physicsnemo.experimental.models.isla._tree_read.tree_read_walk` with
        ``walk_points=False``: :func:`walk_points_range` walks a range of them, with
        that range's geometry); ``bnd`` are the boundary's tokens, ``q_r`` ``(B, Q,
        d)`` the points (frame units) and ``normal_tol`` ``(B, N)`` the bound on the
        rounding of each boundary token's normal (``None``: the normals as given are
        exact); ``conditions`` ``(B, N)`` (or ``None``) each token's declared
        condition, one set of face trees per condition."""
        r, n_hat, vecs, log_w = bnd["r"], bnd["n_hat"], bnd.get("vecs", None), bnd["log_w"]
        log_frac = log_w[..., 0] - torch.logsumexp(log_w[..., 0], dim=1, keepdim=True)
        return tree_read_walk(
            query_r=q_r,
            boundary_r=r,
            boundary_n=n_hat,
            boundary_vecs=vecs,
            log_frac=log_frac,
            g_unit=g_unit,
            theta=self.tree_read_theta,
            eps=_EPS,
            normal_tol=normal_tol,
            leaf_size=self.tree_read_leaf_size,
            kernel=self._resolve_tree_read_kernel(q_r),
            walk=self._tree_read_kernel,
            build=self._tree_read_kernel,
            walk_points=False,
            conditions=conditions,
        )

    def _forward_tokens(
        self,
        boundary: TensorDict,
        g_unit: torch.Tensor,
        g_scalars: torch.Tensor | None,
        gauge: torch.Tensor,
        out_rows: list[tuple[int, int]] | None = None,
    ) -> TensorDict:
        r"""The batched network on the surface task's token inputs.

        ``boundary`` holds the boundary tokens over ``(B, N)``: ``points`` and
        ``normals`` ``(B, N, d)``, ``measure`` ``(B, N)`` and the conditions' inputs
        in the block layout, ``seed`` and
        the block ``vecs`` and ``tens`` (:meth:`_condition_tokens`);
        ``g_unit`` ``(B, K, d)`` and ``g_scalars`` ``(B, S)`` (or ``None``) the
        global inputs (:meth:`_global_fields`); ``gauge`` ``(B, 1, 1)`` the length
        unit (:meth:`_gauge`); ``out_rows`` the
        surface task's output tokens as ``(start, stop)`` ranges of the boundary
        tokens (``None``: every boundary token).
        Returns the declared outputs over the ``(B, N_out)`` tokens the head reads.

        The token state is one TensorDict over ``(B, N)``, the boundary tokens, then
        the padding tokens that complete ``N`` to a multiple of
        :data:`_TOKEN_ALIGNMENT` (:func:`_pad_tokens`), with fields ``h``, ``r``, ``n_hat``,
        ``g_hat``, ``log_w`` and the block-level ``vecs`` and ``tens``; the slice
        blocks take its tensors (module docstring, *Data layout*). The interior task
        runs :class:`QueryChunks` instead.
        """
        tokens, gauge, geo_kernel = self._boundary_tokens(boundary, g_unit, gauge)
        log_shift = self._log_shift(boundary, tokens["log_w"].dtype)
        ### The blocks' and the head frame's log weights take the shift.
        if log_shift is not None:
            tokens.set("log_w", tokens["log_w"] + log_shift[..., None])
        padded = _pad_tokens(tokens, _TOKEN_ALIGNMENT)
        ### The end of the real tokens when padding follows them, None when nothing was
        ### padded: then every read below is that of the unpadded token set.
        n_real = None if padded is tokens else tokens.batch_size[1]
        tokens = padded
        _hint_token_axis(tokens.batch_size[1], self._token_widths)

        h, log_w, r = tokens["h"], tokens["log_w"], tokens["r"]
        vecs, tens = tokens.get("vecs", None), tokens.get("tens", None)
        for block in self._block_callables():
            h = block(h, log_w, r, tokens["n_hat"], tokens["g_hat"], _EPS, vecs, tens, geo_kernel=geo_kernel)
        tokens.set("h", h)

        ### The tokens the head reads: the prediction boundaries' cells, never the
        ### padding tokens. The head frame builds its slice anchors from ALL tokens (r, log_w and h).
        if out_rows is not None:
            out = torch.cat([tokens[:, a:b] for a, b in out_rows], dim=1)
        else:
            out = tokens if n_real is None else tokens[:, :n_real]
        ### The frame is built after the head's coefficients: autograd accumulates the
        ### gradient of h over its consumers in the order they were built.
        return self._head(out, (lambda: self._frame_anchors(h, log_w, r)) if self._needs_frame else None)

    def _boundary_tokens(self, boundary: TensorDict, g_unit: torch.Tensor, gauge: torch.Tensor):
        r"""The boundary tokens' state over ``(B, N)`` (:meth:`_forward_tokens`,
        :class:`QueryChunks`), the
        length unit and the geometry kernel the blocks are called with."""
        points = boundary["points"]
        b, n, _ = points.shape
        _hint_token_axis(n, self._token_widths)
        geo_kernel = self._resolve_geo_kernel(points)
        K, dim = self._n_global_vectors, self.n_spatial_dims
        g_hat = g_unit[:, None].expand(b, n, K, dim)

        ### Frame: no origin. Positions enter in units of gauge, the full-resolution
        ### boundary's measure-weighted RMS pairwise distance (_gauge, the same for
        ### every draw).
        normals = boundary["normals"]
        n_hat = normals / normals.norm(dim=-1, keepdim=True).clamp_min(_EPS)

        ### A cell of zero measure keeps a weight of _EPS times the case's total measure,
        ### a floor relative to the case's units, so a similarity leaves every log weight
        ### shifted by one constant, which no softmax over points reads. A condition
        ### whose total is below _EPS times the case's has every cell at the floor.
        measure = boundary["measure"].to(points.dtype)
        log_w = torch.log(measure.clamp_min(_EPS * measure.sum(1, keepdim=True)))[..., None]

        ### Boundary inputs: each condition's seed through its input layer, its
        ### vectors and tensors through its maps onto the shared channels.
        h, b_vecs, b_tens = self._boundary_channels(boundary)
        tokens = TensorDict(
            h=h,
            r=points / gauge,
            n_hat=n_hat,
            g_hat=g_hat,
            log_w=log_w,
            **self._block_extras(b_vecs, b_tens, b, n, points),
            batch_size=[b, n],
        )
        return tokens, gauge, geo_kernel

    def _query_tokens(self, query: TensorDict, surface, log_w, g_unit, g_scalars, gauge, like):
        r"""The prediction points' tokens over ``(B, Q)`` (:meth:`QueryChunks._fields`) and
        their built-in wall distance ``(B, Q)`` in frame units, their geometry measured
        from ``surface`` (:meth:`_query_geometry`); ``log_w`` and ``like`` are the
        boundary tokens' log weights and points."""
        b = like.shape[0]
        K, dim = self._n_global_vectors, self.n_spatial_dims
        query_points = query["points"]
        nq = query_points.shape[1]
        q_nhat, s, builtin = self._query_geometry(query, surface, gauge, like)
        q_g = g_unit[:, None].expand(b, nq, K, dim)
        q_seed = [self._dots(q_nhat, q_g)]
        if g_scalars is not None:
            q_seed.append(g_scalars[:, None, :].expand(b, nq, -1).to(like.dtype))
        h_q = self.embed(self.qt.seed(torch.cat([*q_seed, builtin], -1)))
        ### Passive queries: log-weight -inf leaves them out of every softmax
        ### over points (routing prototypes, anchors, slice states, head-frame
        ### anchors), so they read every block's slices and never write.
        q_logw = log_w.new_full((b, nq, 1), -math.inf)
        q_tokens = TensorDict(
            h=h_q,
            r=query_points / gauge,
            n_hat=q_nhat,
            g_hat=q_g,
            log_w=q_logw,
            **self._block_extras(None, None, b, nq, like),
            batch_size=[b, nq],
        )
        return q_tokens, s[..., 0].abs()

    def _query_geometry(self, query: TensorDict, surface, gauge, like):
        r"""The prediction points' direction ``(B, Q, d)``, signed distance ``s``
        ``(B, Q, 1)`` in frame units and built-in channels ``(B, Q, 2)``, from the
        boundary surface ``surface`` (``(points, cells)``, :class:`_CaseInputs`) of
        their one case."""
        ### The built-in channels s and asinh(s / 1e-6), and the token's "normal"
        ### sign(d) u with sign(0) = +1, the gradient of the signed distance: both
        ### continuous through the surface (module docstring, *Interior geometry*).
        q_points = query["points"][0]
        with torch.autocast(q_points.device.type, enabled=False):
            dist, u = exact_boundary_geometry(q_points, *surface, kernel=self._resolve_geometry_kernel(q_points))
        with torch.autocast(like.device.type, enabled=False):
            dist, u = dist[None].to(like.dtype), u[None].to(like.dtype)
            q_nhat = torch.where((dist >= 0)[..., None], u, -u)
            s = (dist / gauge.squeeze(-1))[..., None]  # (B, Q, 1), in frame units
            builtin = torch.cat([s, soft_log_length(s)], -1)
        return q_nhat, s, builtin

    def _frame_anchors(self, h, log_w, r):
        r"""The head frame's prototypes ``(B, S, hidden)`` and slice anchors ``(B, S, d)``
        from the tokens' states, log weights and positions."""
        ### The frame's prototypes are synthesized once from the tokens with their
        ### measures (the same quadrature that weights the anchors), so the frame is
        ### refinement invariant like the blocks; both routings read those prototypes.
        protos_f = self.frame_assign.prototypes(h, log_w)
        a_f = _softmax_over_points(self.frame_assign.logits(h, protos_f) + log_w)
        return protos_f, torch.einsum("bns,bnc->bsc", a_f, r)

    def _head(self, out: TensorDict, frame) -> TensorDict:
        r"""The declared outputs of the tokens ``out`` (their ``h``, ``r``, ``n_hat`` and
        ``g_hat``), with the head frame's ``(prototypes, anchors)``
        (:meth:`_frame_anchors`, or a callable returning them; ``None`` without outputs
        of rank 1 or more)."""
        K, dim = self._n_global_vectors, self.n_spatial_dims
        ### With declared scales or output_stats the head and the assembly run in the
        ### parameters' precision under autocast: bf16's relative step (4e-3) would be
        ### carried into the physical outputs by the redimensionalization.
        head_dtype = self.head.weight.dtype
        if self._fp32_head:
            head_ctx = lambda: torch.autocast(out["h"].device.type, enabled=False)  # noqa: E731
            h_out = out["h"].to(head_dtype)
        else:
            head_ctx, h_out = contextlib.nullcontext, out["h"]
        with head_ctx():
            coeffs = self._coeff_layout.unpack(self.head(self.norm_out(h_out)))
        if not self._needs_frame:
            ### Scalar-only outputs are the coefficients themselves; no token basis.
            return coeffs
        ### Head frame: the radial basis vector is the direction from the token's
        ### soft slice anchor (measure-weighted mean of token positions), not
        ### from a centroid; translation covariant.
        protos_f, z_f = frame() if callable(frame) else frame
        b_f = torch.softmax(self.frame_assign.logits(out["h"], protos_f), dim=-1)
        r_out = out["r"]
        u = r_out - torch.einsum("bns,bsc->bnc", b_f, z_f).to(r_out.dtype)
        r_hat = u / u.norm(dim=-1, keepdim=True).clamp_min(_EPS)

        ### The token basis. True vectors: the global directions, the normal, the
        ### radial direction and the polar complements of (rhat, n) and each
        ### (rhat, g_k) (the vector's component orthogonal to rhat, left
        ### unnormalized so it vanishes smoothly where the pair is parallel; any
        ### dimension). Pseudovectors: the Levi-Civita construction from the
        ### primaries {g_k, n, rhat}: in three dimensions their pairwise cross
        ### products (for K = 1: g x n, g x rhat, n x rhat), in two their
        ### perpendiculars; both flip sign under a reflection while their
        ### coefficients do not.
        n_out, g_out = out["n_hat"], out["g_hat"]
        gs = [g_out[:, :, k] for k in range(K)]
        true_basis = gs + [n_out, r_hat, _polar_complement(r_hat, n_out)]
        for g in gs:
            true_basis.append(_polar_complement(r_hat, g))
        primaries = gs + [n_out, r_hat]
        pseudo_basis = _pseudovectors(primaries, dim)
        true_vecs = torch.stack(true_basis, dim=-2)  # (B, N, m, d)
        pseudo_vecs = (
            torch.stack(pseudo_basis, dim=-2)
            if pseudo_basis
            else true_vecs.new_zeros(*out.batch_size, 0, dim)
        )
        if self._fp32_head:
            true_vecs, pseudo_vecs = true_vecs.to(head_dtype), pseudo_vecs.to(head_dtype)
        ### Each output of rank >= 1 from its coefficients, in place of them.
        with head_ctx():
            for i, (name, layout) in enumerate(self._output_layouts.items()):
                if layout.spec.rank:
                    key = FieldSchema.key(name)
                    index = getattr(self, f"_output_index_{i}")
                    coeffs.set(key, _assemble_output(coeffs.get(key), layout, true_vecs, pseudo_vecs, index))
        return coeffs


class QueryChunks:
    r"""An interior task's forward pass split into chunks of prediction points.

    The result of :meth:`ISLA.query_chunks`, which runs the interior task's boundary side
    once and holds what the prediction points read of it as leaves that accumulate
    gradients. Calling ``chunks(start, stop)`` runs the points' side for a range of
    the prediction points; after the chunks' backward passes,
    :meth:`backward` carries the leaves' gradients through the boundary side once.
    The constructor is internal: only :meth:`ISLA.query_chunks` builds one (and
    ``ISLA.forward``, whose interior path is the same computation in one graph,
    without leaves).

    Examples
    --------
    One training step over chunks of at most ``max_pairs`` walk pairs:

    >>> chunks = model.query_chunks(boundary_meshes=..., prediction_points=points)  # doctest: +SKIP
    >>> for start, stop in chunks.spans(max_pairs):  # doctest: +SKIP
    ...     loss(chunks(start, stop), targets[start:stop]).backward()
    >>> chunks.backward()  # doctest: +SKIP
    """

    def __init__(self, model: ISLA, inputs: "_CaseInputs", detach: bool = True) -> None:
        boundary, query, g_unit = inputs.boundary, inputs.query, inputs.g_unit
        ### detach=False (forward): what the points read stays in the boundary side's
        ### graph, so one backward pass reaches the parameters.
        self._detach = detach
        self._model, self._query, self._g_unit, self._g_scalars = model, query, g_unit, inputs.g_scalars
        ### Follow-up: each chunk's geometry (_query_geometry) rebuilds the boundary side
        ### of the search (the weld, the zero-area filter, the cell tree, its slabs and
        ### the winding tree) inside the search's custom ops. Building it once needs a
        ### setup op that returns those records and a query op that reads them, with
        ### the slab choice made independent of each chunk's point count.
        self._surface = inputs.surface
        self._scales, self._out_points, self._global_td = inputs.scales, inputs.out_points, inputs.global_td
        self._roots, self._leaves = [], []
        tokens, self._gauge, self._geo_kernel = model._boundary_tokens(boundary, g_unit, inputs.gauge)
        log_shift = model._log_shift(boundary, tokens["log_w"].dtype)
        self._like, self._log_w = boundary["points"], tokens["log_w"]
        ### The forest now; each chunk walks its own points with their geometry, so the
        ### pair lists and geometry held are one chunk's, and the fast walk's array
        ### limit applies per chunk.
        self._walk = model._tree_read_walk(
            tokens,
            query["points"] / self._gauge,
            g_unit,
            boundary.get("normal_tol", None),
            conditions=boundary.get("condition", None),
        )
        ### Every other boundary-side tensor the chunks read is a leaf too, so a chunk's
        ### backward pass stops there and backward() carries the inputs' gradients (the
        ### boundary points', the global data's) through the boundary side once. The
        ### prediction points stay in the caller's graph, so a chunk's outputs can be
        ### differentiated in space.
        if detach:
            self._gauge, self._like, self._g_unit = map(self._leaf, (self._gauge, self._like, g_unit))
            if self._g_scalars is not None:
                self._g_scalars = self._leaf(self._g_scalars)
            self._surface = (self._leaf(self._surface[0]), self._surface[1])
            if self._scales is not None:
                self._scales = {k: self._leaf(v) for k, v in self._scales.items()}
            ### The read's sources too: their pair invariants take a gradient when the geometry requires one.
            sources = ("src_r", "src_n", "src_vecs", "g_unit")
            self._walk.geometry = [
                {k: self._leaf(v) if k in sources and v is not None else v for k, v in g.items()}
                for g in self._walk.geometry
            ]
        ### The blocks' log weights take the shift (the forest took the raw ones); each
        ### chunk's read takes the cells' and the conditions' shifts as leaves.
        self._log_shift = self._condition_shift = None
        if log_shift is not None:
            tokens.set("log_w", tokens["log_w"] + log_shift[..., None])
            self._log_shift = self._leaf(log_shift)
            self._condition_shift = self._leaf(model._condition_shift(boundary, log_shift.dtype))
        ### The boundary side's token axis completed as forward completes its own
        ### (_pad_tokens): at an unaligned count cuBLAS reduces some token-axis bf16
        ### products, the slice anchors among them, with split-K partial sums rounded to bf16.
        n_cells = tokens.batch_size[1]
        tokens = _pad_tokens(tokens, _TOKEN_ALIGNMENT)
        h, log_w, r = tokens["h"], tokens["log_w"], tokens["r"]
        vecs, tens = tokens.get("vecs", None), tokens.get("tens", None)
        self._slices, self._sources = [], []
        for i, block in enumerate(model._block_callables()):
            h, slices = block(
                h, log_w, r, tokens["n_hat"], tokens["g_hat"], _EPS, vecs, tens, geo_kernel=self._geo_kernel, slices=True
            )
            self._slices.append(tuple(self._leaf(t) for t in slices))
            h_b = h[:, :n_cells].reshape(-1, h.shape[-1])
            kv = model.kernel_reads[i].sources(h_b, self._walk, model._reads_compiled())
            self._sources.append(self._leaf(kv))
        self._frame = (
            tuple(self._leaf(t) for t in model._frame_anchors(h, log_w, r)) if model._needs_frame else None
        )

    @property
    def n_points(self) -> int:
        r"""The number of prediction points, :math:`Q`.

        Returns
        -------
        int
            :math:`Q`.
        """
        return int(self._query.shape[1])

    @functools.cached_property
    def pair_counts(self) -> Int[torch.Tensor, " n_points"]:
        r"""The number of (prediction point, boundary source) pairs of the boundary
        read that each prediction point reads, counted without making the pair
        lists.

        Returns
        -------
        torch.Tensor
            Integer tensor of shape :math:`(Q,)`.
        """
        return walk_pair_counts(self._walk)

    def spans(self, max_pairs: int) -> list[tuple[int, int]]:
        r"""Split the prediction points into ranges of bounded pair counts.

        Parameters
        ----------
        max_pairs : int
            The most walk pairs (:attr:`pair_counts`) a range may hold.

        Returns
        -------
        list[tuple[int, int]]
            Consecutive ranges ``(start, stop)`` that cover ``0`` to
            :attr:`n_points`, the fewest, filled from the start, whose pair counts
            are each at most ``max_pairs``. A point with more pairs than
            ``max_pairs`` is alone in its range.
        """
        n = self.n_points
        cum = self.pair_counts.cumsum(0).cpu() if n else None
        if n == 0 or int(cum[-1]) <= max_pairs:
            return [(0, n)]
        spans, start = [], 0
        while start < n:
            base = int(cum[start - 1]) if start else 0
            stop = max(int(torch.searchsorted(cum, base + max_pairs, right=True)), start + 1)
            spans.append((start, stop))
            start = stop
        return spans

    def _leaf(self, t: torch.Tensor) -> torch.Tensor:
        if not self._detach:
            return t
        leaf = t.detach().requires_grad_(t.requires_grad)
        self._roots.append(t)
        self._leaves.append(leaf)
        return leaf

    def __call__(self, start: int, stop: int) -> Mesh:
        r"""The output of the prediction points ``start`` to ``stop``.

        Parameters
        ----------
        start, stop : int
            The range of prediction points, as in ``prediction_points[start:stop]``.

        Returns
        -------
        Mesh
            The point-cloud ``Mesh`` that ``ISLA.forward`` returns, restricted to
            those points: its rows equal ``ISLA.forward``'s up to summation
            order, differentiable in the parameters and the leaves.
        """
        points = self._out_points[start:stop]
        return Mesh(points=points, point_data=self._fields(start, stop)[0].to(points.device), global_data=self._global_td)

    def _fields(self, start: int, stop: int) -> TensorDict:
        r"""The declared output fields of the prediction points ``start`` to ``stop``.

        Parameters
        ----------
        start, stop : int
            The range of prediction points, as in ``prediction_points[start:stop]``.

        Returns
        -------
        TensorDict
            The output fields, keyed as the ``point_data`` of :meth:`__call__`'s
            ``Mesh``, with batch size ``(1, stop - start)``.
        """
        m = self._model
        q_tokens, q_wall = m._query_tokens(
            self._query[:, start:stop], self._surface, self._log_w, self._g_unit, self._g_scalars, self._gauge, self._like
        )
        pairs = walk_range_pairs(
            m.kernel_basis,
            walk_points_range(self._walk, start, stop, q_tokens["n_hat"], q_wall, q_tokens["r"]),
            log_shift=self._log_shift,
            condition_shift=self._condition_shift,
        )
        h = q_tokens["h"]
        vecs, tens = q_tokens.get("vecs", None), q_tokens.get("tens", None)
        for i, read in enumerate(m._block_read_callables()):
            h = read(
                h, self._slices[i], q_tokens["r"], q_tokens["n_hat"], q_tokens["g_hat"], _EPS, vecs, tens,
                geo_kernel=self._geo_kernel,
            )  # fmt: skip
            delta = m.kernel_reads[i].read(self._sources[i], h.reshape(-1, h.shape[-1]), pairs, m._reads_compiled())
            h = h + delta.view_as(h)
        q_tokens.set("h", h)
        return m._redimensionalize(m._head(q_tokens, self._frame), self._scales)

    def backward(self) -> None:
        r"""Run the boundary side's backward pass.

        Carries the gradients that the chunks' backward passes accumulated in the
        leaves through the boundary side once, so the parameters' gradients are
        those of the unchunked step whose loss is the sum of the chunks'. Releases
        the boundary side's graph; call it once, after every chunk's backward pass.

        Returns
        -------
        None
            The gradients accumulate in the parameters' ``.grad``.
        """
        roots = [(t, leaf.grad) for t, leaf in zip(self._roots, self._leaves) if leaf.grad is not None]
        self._roots, self._leaves = [], []
        if roots:
            torch.autograd.backward([t for t, _ in roots], [g for _, g in roots])
