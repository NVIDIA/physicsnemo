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

r"""Pointwise multitask variational Gaussian Process head for field regression.

Provides :class:`FieldVariationalGPHead`, a module that can be attached to any
backbone which exposes *per-point* features to produce per-point predictive
uncertainty over a multi-channel field (e.g. surface pressure +
wall-shear-stress).

This is the *field* member of a two-head family.  Both are variational GPs with
inducing points, a Matern-5/2 ARD kernel and a variational ELBO; they differ in
what a "data point" is:

========================  ======================  ==========================
Head                      Input                   Output
========================  ======================  ==========================
`VariationalGPHead`       one pooled embedding    one scalar per geometry
                          ``(B, D)``              ``(B,)``
`FieldVariationalGPHead`  per-point features      ``num_tasks`` channels per
                          ``(..., D)``            point ``(..., num_tasks)``
========================  ======================  ==========================

``B`` is the number of geometries in the batch, ``D`` the feature width (the
head's *input_dim*), and ``num_tasks`` the number of output channels.  The
leading ``...`` stands for any batch/point dimensions, so ``(B, N, D)`` for
``N`` points per geometry, or ``(N, D)`` for a single unbatched point cloud.

The posterior mean is the field prediction; the posterior variance is the
per-point uncertainty, which grows as a point's feature moves away from the
learned inducing points (a distance-aware, single-pass UQ signal).

Attaching to a backbone
-----------------------
The head is model-agnostic: it consumes a feature tensor and
nothing else.  There is no dependency on any particular backbone, no assumption
about mesh topology, and coordinates are not required as a separate input
(any positional information the backbone encodes simply arrives inside the
features).  The only contract is:

1. the backbone emits per-point features whose last dimension is
   ``input_dim`` — any leading batch/point dims are flattened internally, so
   ``(B, N, D)``, ``(N, D)`` and ``(B, T, N, D)`` all work;
2. targets are supplied as ``(..., num_tasks)`` matching those leading dims.

So any point-wise encoder works — GeoTransolver, DoMINO, MeshGraphNet — by
exposing whatever it already computes before its final projection::

    head = FieldVariationalGPHead(input_dim=feat_dim, num_tasks=4,
                                  n_train=n_points_per_epoch)

    feats = backbone.encode(batch)        # (B, N, feat_dim)
    mean, neg_elbo = head.forward_and_loss(feats, targets)
    loss = neg_elbo + lambda_mse * mse(mean, targets)

At inference, :meth:`FieldVariationalGPHead.predict` returns the mean plus the
epistemic/total variance split in a single forward pass.

What a working recipe needs
---------------------------
The head owns its objective, but that objective alone does not take a randomly
initialized backbone to a useful field surrogate.  These are properties of the
recipe rather than of the architecture, which is why they live in the training
script; a caller that skips them should expect a collapsed variance or a
diverged noise scale rather than a bad-but-working model:

* **Inducing points seeded from real features.** The default random-normal
  inducing points sit nowhere near the backbone's feature distribution.  Push a
  few batches through the backbone and pass the features to
  :meth:`FieldVariationalGPHead.set_inducing_points`.
* **An auxiliary MSE on the posterior mean.** The ELBO can buy likelihood by
  inflating the variance instead of improving the mean; anchoring the mean
  removes that shortcut while the backbone is still learning the field.
* **A ramp on the KL term (and optionally on the whole ELBO).** Start with the
  data-fit term dominant so the mean is accurate before the KL pulls the
  variational posterior toward the prior.  ``beta`` in
  :meth:`FieldVariationalGPHead.forward_and_loss` is that weight.
* **A noise floor plus gradient clipping, whenever the noise MLP is on.** The
  heteroscedastic ELBO weights each point by :math:`1/\sigma^2(x)`, so one
  point whose noise collapses dominates the step; *noise_std_range* bounds the
  collapse and clipping absorbs the overcorrection that follows it.
* **Optionally, a penalty computed in kernel space.** Auxiliary terms on the
  GP-input geometry go through :meth:`transform_features`, whose output is fed
  back with ``pretransformed=True`` so the transform runs once per step.

The reference recipe under
``examples/cfd/external_aerodynamics/transformer_models`` implements all of
these: ``src/conf/geotransolver_surface_field_gp.yaml`` holds the settled
values and the README's "Field Variational GP Head" section explains them.

Key design choices
------------------
* **Independent multitask GP** — Each of the ``num_tasks`` output channels has
  its own variational GP (shared inducing-point structure, independent kernels
  and variational parameters), built with GPyTorch's
  ``IndependentMultitaskVariationalStrategy``.
* **Float64 GP internals (default)** — Short lengthscales on the inducing-point
  covariance make ``K_uu`` ill-conditioned in float32.  GP internals run in
  float64 by default; inputs are upcast on entry and outputs downcast on exit so
  gradients flow through the backbone seamlessly.  Controlled by *use_double*.
* **Optional DKL feature extractor** — A small pointwise MLP can be inserted
  between the backbone features and the GP kernel (Deep Kernel Learning),
  reducing a wide feature vector to a compact, well-conditioned kernel input.
* **Matern ARD kernel, smoothness 5/2 by default** — The Matern order
  :math:`\nu` sets how many times the sample paths are differentiable, so it
  encodes how smooth the field is assumed to be: 1/2 gives the non-differentiable
  Ornstein-Uhlenbeck limit, 5/2 gives twice-differentiable paths, and
  :math:`\nu \to \infty` recovers the RBF kernel.  5/2 is the usual choice for
  smooth physical fields — differentiable enough for a pressure or shear field,
  short of the RBF limit whose sample paths are analytic and which tends to
  over-smooth.  Set *matern_nu* to change it; GPyTorch implements the three
  half-integer orders 1/2, 3/2 and 5/2.  ARD (Automatic Relevance Determination)
  gives every kernel input dimension its own lengthscale, so unhelpful DKL
  features can be switched off by growing theirs.
* **Optional heteroscedastic noise** — The observation noise can be made a
  function of the features rather than one learned scalar per channel; see
  :meth:`FieldVariationalGPHead._hetero_neg_elbo`.

Requires ``gpytorch`` — install via ``pip install gpytorch`` or use the
``uq-extras`` optional dependency group.
"""

from __future__ import annotations

import importlib
import math
from typing import Literal, NamedTuple

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core.version_check import check_version_spec

_GPYTORCH_AVAILABLE = check_version_spec("gpytorch", hard_fail=False)

if _GPYTORCH_AVAILABLE:
    gpytorch = importlib.import_module("gpytorch")
    _ApproximateGP = gpytorch.models.ApproximateGP
    CholeskyVariationalDistribution = (
        gpytorch.variational.CholeskyVariationalDistribution
    )
    VariationalStrategy = gpytorch.variational.VariationalStrategy
    IndependentMultitaskVariationalStrategy = (
        gpytorch.variational.IndependentMultitaskVariationalStrategy
    )
    VariationalELBO = gpytorch.mlls.VariationalELBO
else:
    _ApproximateGP = nn.Module


def _require_gpytorch() -> None:
    if not _GPYTORCH_AVAILABLE:
        raise ImportError(
            "physicsnemo.experimental.uq.FieldVariationalGPHead requires gpytorch. "
            "Install it with: pip install gpytorch  "
            "(or: pip install nvidia-physicsnemo[uq-extras])"
        )


class _MultitaskVariationalGPLayer(_ApproximateGP):
    """Low-level independent multitask variational GP with Matern-5/2 ARD kernels.

    This is an internal building block used by :class:`FieldVariationalGPHead`.  Users
    should not need to instantiate it directly.

    Parameters
    ----------
    inducing_points : Float[torch.Tensor, "tasks n_inducing gp_dim"]
        Initial inducing point locations.
    input_dim : int
        Dimensionality of each input (must match last dim of *inducing_points*).
    num_tasks : int
        Number of output channels / independent GPs.
    lengthscale_range : tuple[float, float]
        Hard interval constraint on per-dimension lengthscales.
    lengthscale_prior : tuple[float, float] | None
        ``(concentration, rate)`` for a Gamma prior on lengthscales.
    outputscale_prior : tuple[float, float] | None
        ``(concentration, rate)`` for a Gamma prior on the output scale.
    matern_nu : float
        Matern smoothness order; one of ``0.5``, ``1.5``, ``2.5``.
    """

    def __init__(
        self,
        inducing_points: Float[torch.Tensor, "tasks n_inducing gp_dim"],
        input_dim: int = 16,
        num_tasks: int = 4,
        lengthscale_range: tuple[float, float] = (0.01, 10.0),
        lengthscale_prior: tuple[float, float] | None = None,
        outputscale_prior: tuple[float, float] | None = None,
        matern_nu: float = 2.5,
    ) -> None:
        _require_gpytorch()
        batch_shape = torch.Size([num_tasks])

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2),
            batch_shape=batch_shape,
        )
        base_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        variational_strategy = IndependentMultitaskVariationalStrategy(
            base_strategy,
            num_tasks=num_tasks,
        )
        super().__init__(variational_strategy)

        self.num_tasks = num_tasks
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)

        ls_constraint = gpytorch.constraints.Interval(*lengthscale_range)
        ls_prior_obj = None
        if lengthscale_prior is not None:
            ls_prior_obj = gpytorch.priors.GammaPrior(*lengthscale_prior)

        base_kernel = gpytorch.kernels.MaternKernel(
            nu=matern_nu,
            ard_num_dims=input_dim,
            batch_shape=batch_shape,
            lengthscale_constraint=ls_constraint,
            lengthscale_prior=ls_prior_obj,
        )

        os_prior_obj = None
        if outputscale_prior is not None:
            os_prior_obj = gpytorch.priors.GammaPrior(*outputscale_prior)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel,
            batch_shape=batch_shape,
            outputscale_prior=os_prior_obj,
        )

    def forward(
        self, x: Float[torch.Tensor, "points gp_dim"]
    ) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class FieldVariationalGPPrediction(NamedTuple):
    """Structured output of :meth:`FieldVariationalGPHead.predict`.

    Every field has shape ``(..., num_tasks)``: the leading dimensions of the
    features passed to :meth:`FieldVariationalGPHead.predict`, followed by the
    channel dimension.

    Attributes
    ----------
    mean : Float[torch.Tensor, "... tasks"]
        Predictive mean.
    variance : Float[torch.Tensor, "... tasks"]
        Total predictive variance (epistemic + input-dependent observation
        noise).
    lower : Float[torch.Tensor, "... tasks"]
        Lower bound of the confidence interval.
    upper : Float[torch.Tensor, "... tasks"]
        Upper bound of the confidence interval.
    epistemic_variance : Float[torch.Tensor, "... tasks"]
        Latent GP function variance *only* (the reducible / model uncertainty,
        excluding the constant likelihood noise floor).  This is the signal to
        use for active learning and for "where is the model uncertain?" maps —
        it has far more spatial contrast than the noise-dominated total
        ``variance``.
    """

    mean: Float[torch.Tensor, "... tasks"]
    variance: Float[torch.Tensor, "... tasks"]
    lower: Float[torch.Tensor, "... tasks"]
    upper: Float[torch.Tensor, "... tasks"]
    epistemic_variance: Float[torch.Tensor, "... tasks"]


class FieldVariationalGPHead(nn.Module):
    r"""Pointwise independent multitask variational GP head for field UQ.

    Attach this module to any backbone that produces per-point features to
    obtain per-point predictive uncertainty over a multi-channel field.  The
    posterior mean is the field prediction; the posterior variance is the
    per-point uncertainty.  That variance is distance-aware by construction but
    not calibrated by construction: whether its scale matches observed error
    depends on the recipe and on the train/test shift, so validate it on
    held-out geometries before reading it as an error bar.

    Inputs of shape ``(..., D)`` are accepted (e.g. ``(B, N, D)`` or
    ``(N, D)``); all leading dimensions are flattened into the point dimension,
    the GP is evaluated, and outputs are reshaped back to ``(..., num_tasks)``.

    Every parameter after *input_dim* is keyword-only, since the head is
    normally built from a config block rather than positionally.

    Parameters
    ----------
    input_dim : int
        Dimension of each per-point feature vector from the backbone.
    n_train : int
        Total number of *training points* (across all geometries) — used for
        the ELBO normalization constant so the data term and KL term are
        balanced when minibatching at the point level.  Count points, not
        geometries: 10 geometries of ``N`` points each is ``10 * N``, not 10.
        If a geometry is subsampled during training, count the points actually
        fed to the head per pass, i.e. ``n_geometries * points_per_geometry``.
        Required: the constant sets the balance between the data-fit and KL
        terms, and no default can stand in for the size of the caller's dataset.
    num_tasks : int, optional
        Number of output channels (independent GPs). Default is 4.
    n_inducing : int, optional
        Number of inducing points per task. Default is 256.
    inducing_points : Float[torch.Tensor, "*tasks n_inducing gp_dim"] | None, optional
        Initial inducing locations, either ``(M, gp_dim)`` (shared init,
        broadcast across tasks) or ``(num_tasks, M, gp_dim)``.  If *None*,
        random normal points are used, which is rarely what you want for
        training — see :meth:`set_inducing_points`. Default is ``None``.
    lengthscale_range : tuple[float, float], optional
        Hard interval constraint ``[lo, hi]`` on per-dimension ARD
        lengthscales. Default is ``(0.01, 10.0)``.
    lengthscale_prior : tuple[float, float] | None, optional
        ``(concentration, rate)`` for a Gamma prior on lengthscales.
        Default is ``None``.
    outputscale_prior : tuple[float, float] | None, optional
        ``(concentration, rate)`` for a Gamma prior on the output scale.
        Default is ``None``.

        .. note::
           Both priors reach the objective through GPyTorch's
           :class:`~gpytorch.mlls.VariationalELBO`, which sums the registered
           prior log-probabilities into the loss.  Setting *noise_mlp_hidden*
           switches training to :meth:`_hetero_neg_elbo`, which builds its
           likelihood term by hand and so carries no prior term: the priors are
           then inert and only *lengthscale_range*, a hard constraint rather
           than a prior, still binds.
    matern_nu : float, optional
        Smoothness order :math:`\nu` of the Matern kernel: one of ``0.5``,
        ``1.5`` or ``2.5``, the half-integer orders GPyTorch implements.  It
        controls how many times the GP's sample paths are differentiable, so it
        is an assumption about the field: ``2.5`` (default) gives
        twice-differentiable paths, appropriate for a smooth surface field,
        while ``0.5`` is the rough Ornstein-Uhlenbeck limit.  Default is
        ``2.5``.
    mlp_hidden : list[int] | None, optional
        Hidden layer sizes for an optional pointwise DKL feature extractor MLP
        inserted before the GP kernel.  ``None`` feeds the features directly to
        the GP. Default is ``None``.
    feature_norm : {"none", "l2_radial"}, optional
        Normalization applied to the GP-input features.  ``"none"`` passes them
        through.  ``"l2_radial"`` splits each feature into its unit direction
        plus its (batch-standardized) magnitude, appended as one extra ARD
        dimension — so ``gp_input_dim`` becomes ``mlp_hidden[-1] + 1``.  This
        pins the feature scale, preventing the DKL map from shrinking distances
        to circumvent the lengthscale constraint, while keeping the radial
        out-of-distribution cue that a pure unit-sphere projection discards.
        Default is ``"none"``.
    use_double : bool, optional
        If ``True``, GP internals run in float64 for numerical stability of the
        Cholesky decomposition on ``K_uu``. Default is ``True``.
    jitter : tuple[float, float], optional
        ``(float_value, double_value)`` passed to
        ``gpytorch.settings.cholesky_jitter``. Default is ``(1e-3, 1e-4)``.
    confidence_z : float, optional
        Z-score multiplier for the confidence interval returned by
        :meth:`predict`.  Default is ``1.96`` (95 % interval).
    noise_mlp_hidden : list[int] | None, optional
        Hidden sizes of an observation-noise MLP over the GP-input features.
        ``None`` (default) keeps the standard homoscedastic
        ``MultitaskGaussianLikelihood`` (one noise scalar per channel).  When
        set, the observation noise becomes input-dependent, which makes the
        *total* predictive std informative for per-point error ranking — with a
        constant noise floor the total std ranks points identically to the
        epistemic std, so all ranking signal comes from the (typically <1 %)
        epistemic share of the variance.  The split between the epistemic and
        observation-noise terms is retained; only the latter gains spatial
        structure.
    noise_std_range : tuple[float, float], optional
        Hard clamp ``(lo, hi)`` on the per-point noise std, as a safety net
        against a degenerate zero-noise solution.  Default ``(1e-3, 10.0)``.

    Attributes
    ----------
    gp_layer : _MultitaskVariationalGPLayer
        The independent multitask variational GP.
    likelihood : gpytorch.likelihoods.MultitaskGaussianLikelihood
        Homoscedastic observation-noise model.  Retained even when
        *noise_mlp_hidden* is set (the heteroscedastic path computes its own
        noise), because it holds the learned per-channel noise floor.
    mll : gpytorch.mlls.VariationalELBO
        Marginal log-likelihood objective (its ``beta`` can be annealed).
    feature_extractor : nn.Sequential | None
        Optional DKL MLP.
    gp_input_dim : int
        Width of the kernel input, after the DKL MLP and *feature_norm*.

    See Also
    --------
    physicsnemo.experimental.uq.VariationalGPHead
        The scalar counterpart: pools a geometry to one embedding and predicts a
        single value per geometry rather than a field.

    Examples
    --------
    >>> head = FieldVariationalGPHead(
    ...     input_dim=448, num_tasks=4, n_inducing=256,
    ...     n_train=51200 * 100, mlp_hidden=[128],
    ... )
    >>> feats = torch.randn(1, 4096, 448)
    >>> pred = head.predict(feats)
    >>> pred.mean.shape
    torch.Size([1, 4096, 4])
    """

    def __init__(
        self,
        input_dim: int,
        *,
        n_train: int,
        num_tasks: int = 4,
        n_inducing: int = 256,
        inducing_points: Float[torch.Tensor, "*tasks n_inducing gp_dim"] | None = None,
        lengthscale_range: tuple[float, float] = (0.01, 10.0),
        lengthscale_prior: tuple[float, float] | None = None,
        outputscale_prior: tuple[float, float] | None = None,
        matern_nu: float = 2.5,
        mlp_hidden: list[int] | None = None,
        feature_norm: Literal["none", "l2_radial"] = "none",
        use_double: bool = True,
        jitter: tuple[float, float] = (1e-3, 1e-4),
        confidence_z: float = 1.96,
        noise_mlp_hidden: list[int] | None = None,
        noise_std_range: tuple[float, float] = (1e-3, 10.0),
    ) -> None:
        super().__init__()
        _require_gpytorch()

        # Config-driven instantiation (hydra, YAML) delivers plain strings, which
        # the Literal annotation cannot police at runtime.
        if feature_norm not in ("none", "l2_radial"):
            raise ValueError(
                f"feature_norm must be 'none' or 'l2_radial', got {feature_norm!r}"
            )

        self.num_tasks = num_tasks
        self._use_double = use_double
        self._jitter = jitter
        self._confidence_z = confidence_z
        self._feature_norm = feature_norm

        if mlp_hidden:
            layers: list[nn.Module] = []
            in_dim = input_dim
            for h in mlp_hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            self.feature_extractor = nn.Sequential(*layers)
            gp_input_dim = mlp_hidden[-1]
        else:
            self.feature_extractor = None
            gp_input_dim = input_dim

        # 'l2_radial' keeps the L2-normalized direction AND appends the pre-norm
        # feature magnitude as one extra ARD dimension.  Projecting onto the unit
        # sphere alone makes out-of-distribution geometries indistinguishable in
        # norm, collapsing the OOD/in-distribution std ratio toward 1.0.  The
        # magnitude is standardized by a (non-affine) BatchNorm tracking the
        # training distribution, so OOD magnitudes land in the tails -> larger
        # kernel distance -> higher posterior variance.
        if feature_norm == "l2_radial":
            self._radial_bn = nn.BatchNorm1d(1, affine=False)
            gp_input_dim += 1
        else:
            self._radial_bn = None
        self.gp_input_dim = gp_input_dim

        inducing_points = self._init_inducing(
            inducing_points, n_inducing, gp_input_dim, num_tasks
        )

        gp_layer = _MultitaskVariationalGPLayer(
            inducing_points,
            gp_input_dim,
            num_tasks=num_tasks,
            lengthscale_range=lengthscale_range,
            lengthscale_prior=lengthscale_prior,
            outputscale_prior=outputscale_prior,
            matern_nu=matern_nu,
        )
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_tasks
        )

        if use_double:
            gp_layer = gp_layer.double()
            likelihood = likelihood.double()

        self.gp_layer = gp_layer
        # Kept even in the heteroscedastic case: it is the homoscedastic noise
        # model (unused then), and eval scripts call ``head.likelihood.eval()``.
        self.likelihood = likelihood
        self.mll = VariationalELBO(self.likelihood, self.gp_layer, num_data=n_train)
        self._num_data = int(n_train)

        # ---- Optional input-dependent (heteroscedastic) observation noise ----
        # The default MultitaskGaussianLikelihood learns one noise scalar per
        # channel. That constant can dominate the total predictive variance, and
        # being constant it adds no information to the per-point ranking of
        # |error| — the total std then ranks points exactly like the epistemic
        # std. Making the noise a function of the GP-input features lets that
        # share of the variance carry ranking signal instead. On a deterministic
        # target there is no observational scatter to learn, so the term absorbs
        # mean-model discrepancy: it marks where the surrogate is structurally
        # wrong. A second variational GP on the log-noise is the classical
        # alternative; here it is an amortized MLP, which is far cheaper — see
        # :meth:`_hetero_neg_elbo` for that trade-off and the caveats.
        self._noise_range = (float(noise_std_range[0]), float(noise_std_range[1]))
        if noise_mlp_hidden:
            layers: list[nn.Module] = []
            in_dim = gp_input_dim
            for h in noise_mlp_hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            layers.append(nn.Linear(in_dim, num_tasks))
            noise_head = nn.Sequential(*layers)
            # Zero-init the output layer so the modulation starts at exactly 1x
            # and training begins from the homoscedastic solution rather than
            # from a random (possibly tiny or huge) per-point noise.
            nn.init.zeros_(noise_head[-1].weight)
            nn.init.zeros_(noise_head[-1].bias)
            base = torch.zeros(num_tasks)
            if use_double:
                noise_head = noise_head.double()
                base = base.double()
            self.noise_head = noise_head
            # Per-task log base std; the MLP only supplies a bounded
            # multiplicative deviation around it.
            self.log_base_noise = nn.Parameter(base)
        else:
            self.noise_head = None
            self.log_base_noise = None

    @staticmethod
    def _init_inducing(
        inducing_points: Float[torch.Tensor, "*tasks n_inducing gp_dim"] | None,
        n_inducing: int,
        gp_input_dim: int,
        num_tasks: int,
    ) -> Float[torch.Tensor, "tasks n_inducing gp_dim"]:
        """Return inducing points of shape ``(num_tasks, M, gp_input_dim)``."""
        if inducing_points is None:
            return torch.randn(num_tasks, n_inducing, gp_input_dim)
        if inducing_points.dim() == 2:
            return inducing_points.unsqueeze(0).expand(num_tasks, -1, -1).contiguous()
        if inducing_points.dim() == 3:
            return inducing_points
        raise ValueError(
            "inducing_points must be (M, D) or (num_tasks, M, D), "
            f"got shape {tuple(inducing_points.shape)}"
        )

    def _gp_context(self):
        """Safety-net jitter for near-singular covariance matrices."""
        return gpytorch.settings.cholesky_jitter(
            float_value=self._jitter[0], double_value=self._jitter[1]
        )

    @property
    def heteroscedastic(self) -> bool:
        """Whether an input-dependent observation-noise head is active."""
        return self.noise_head is not None

    def _pointwise_noise_var(
        self, gp_in: Float[torch.Tensor, "points gp_dim"]
    ) -> Float[torch.Tensor, "points tasks"]:
        """Per-point, per-task observation-noise variance.

        ``sigma_t(x) = exp(log_base_noise_t + clamp(g_t(x), -3, 3))`` — a learned
        per-task base scale times a bounded multiplicative modulation from the
        noise MLP.

        The clamp bounds the *modulation*, not the noise: it limits how far one
        point may sit from its channel's learned base scale, to about 20x either
        way (``exp(3) = 20.1``).  The absolute floor is ``noise_std_range[0]``,
        applied below, so the low end of the clamp does not need to be small —
        its job is to stop a single step from driving a point's noise toward 0,
        where the ``1/sigma^2(x)`` weighting in :meth:`_hetero_neg_elbo` would
        let that point dominate the gradient.  Widening the range weakens that
        guard; the symmetric +/-3 leaves the base scale free to move across
        channels while keeping any one point within a factor of 20 of it.
        """
        log_mod = self.noise_head(gp_in).clamp(-3.0, 3.0)
        std = torch.exp(self.log_base_noise.to(log_mod.dtype) + log_mod)
        std = std.clamp(self._noise_range[0], self._noise_range[1])
        return std.square()

    def _hetero_neg_elbo(
        self,
        dist: "gpytorch.distributions.MultitaskMultivariateNormal",
        gp_target: Float[torch.Tensor, "points tasks"],
        gp_in: Float[torch.Tensor, "points gp_dim"],
        beta: float,
    ) -> Float[torch.Tensor, ""]:
        r"""Negative ELBO with diagonal, input-dependent Gaussian noise.

        The expected log-likelihood under a heteroscedastic Gaussian is the
        variational analogue of the attenuated regression loss of Kendall & Gal
        (*What Uncertainties Do We Need in Bayesian Deep Learning for Computer
        Vision?*, NeurIPS 2017),

        .. math::
            -\log p(y \mid x) \;\simeq\;
            \frac{\lVert y - \mu(x) \rVert^2 + \operatorname{Var}[f(x)]}
                 {2\,\sigma^2(x)}
            + \tfrac{1}{2}\log \sigma^2(x),

        differing only in that the GP contributes the extra
        :math:`\operatorname{Var}[f(x)]` term (the latent posterior variance),
        which their deterministic network does not have.  Each point is weighted
        by :math:`1/\sigma^2(x)`, so the noise head learns to down-weight
        genuinely noisy regions instead of forcing the mean to fit them.

        Mirrors ``gpytorch.mlls.VariationalELBO``'s normalization exactly — the
        expected log-likelihood is summed over points and tasks then divided by
        the number of points, and the KL is divided by ``num_data / beta`` — so
        the loss is on the same scale as the homoscedastic path and the existing
        learning rates and beta/NLL warmup schedules carry over unchanged.  That
        normalization is the stochastic variational bound of Hensman, Fusi &
        Lawrence (*Gaussian Processes for Big Data*, UAI 2013); ``num_data`` is
        its :math:`N`.

        Note on interpretation: :math:`\sigma^2(x)` is the variance of the
        Gaussian likelihood, i.e. input-dependent observation noise. For example,
        on a deterministic target it is not measurement noise — it
        absorbs mean-model discrepancy, so treat it as a learned discrepancy
        variance rather than as physical variability.  Unlike the classical
        variational heteroscedastic GP (Lázaro-Gredilla & Titsias, ICML 2011),
        which puts a second GP on the log-noise and so carries a second KL term,
        this amortized network has no prior on :math:`\sigma`; ``noise_std_range``
        and gradient clipping are the only things preventing it from collapsing.
        The network is chosen for cost: a second variational process would bring
        its own inducing set, Cholesky factor and KL term, roughly doubling the
        variational state and the per-step :math:`O(M^3)` work, whereas the noise
        MLP adds only a few small layers.

        This term is assembled directly rather than through
        :class:`~gpytorch.mlls.VariationalELBO`, so unlike the homoscedastic
        path it carries no registered-prior or added-loss contribution: any
        ``lengthscale_prior`` / ``outputscale_prior`` is inert here, and the
        kernel is shaped by ``lengthscale_range`` alone.
        """
        mu = dist.mean
        latent_var = dist.variance
        noise_var = self._pointwise_noise_var(gp_in)
        # E_q[log N(y | f, sigma^2)] for a diagonal Gaussian, where the
        # E_q[(y - f)^2] term contributes the latent variance.
        ll = -0.5 * (
            math.log(2.0 * math.pi)
            + noise_var.log()
            + ((gp_target - mu).square() + latent_var) / noise_var
        )
        log_lik = ll.sum() / gp_target.shape[0]
        kl = self.gp_layer.variational_strategy.kl_divergence().sum()
        kl = kl / (self._num_data / max(float(beta), 1e-8))
        return -(log_lik - kl)

    def _transform_features(
        self, features: Float[torch.Tensor, "... dim"]
    ) -> Float[torch.Tensor, "... gp_dim"]:
        """Run optional DKL extractor then the (scale-fixing) feature norm.

        The normalization pins the GP-input feature scale so the kernel
        lengthscale must do the smoothing work; without it the DKL map can
        shrink feature distances to defeat a lengthscale constraint.
        ``l2_radial`` keeps the L2-normalized direction but appends the
        (standardized) pre-norm magnitude as an extra dimension, retaining the
        radial out-of-distribution cue that a pure unit-sphere projection
        discards.
        """
        if self.feature_extractor is not None:
            features = self.feature_extractor(features)
        if self._feature_norm == "l2_radial":
            # Split into direction (unit sphere) + standardized magnitude.
            magnitude = features.norm(dim=-1, keepdim=True)
            direction = features / magnitude.clamp_min(1e-12)
            # BatchNorm1d expects (N, C); flatten all leading dims into N.
            lead_shape = magnitude.shape[:-1]
            mag_std = self._radial_bn(magnitude.reshape(-1, 1)).reshape(*lead_shape, 1)
            features = torch.cat([direction, mag_std], dim=-1)
        return features

    def _apply_fe(
        self, features: Float[torch.Tensor, "... dim"]
    ) -> Float[torch.Tensor, "... gp_dim"]:
        """Run feature transform (DKL + norm), then cast to GP precision.

        The cast is what keeps the Cholesky factorization of ``K_uu`` viable:
        with short lengthscales the inducing points are strongly correlated and
        the matrix is ill-conditioned, which in float32 shows up as a Cholesky
        failure or a negative predictive variance.  Only the GP internals run in
        float64 — outputs are cast back in :meth:`forward_and_loss` and
        :meth:`predict`, so the backbone keeps training in its own precision.
        """
        features = self._transform_features(features)
        if self._use_double:
            return features.double()
        return features

    def transform_features(
        self, features: Float[torch.Tensor, "... dim"]
    ) -> Float[torch.Tensor, "... gp_dim"]:
        """Return the features the kernel sees, in GP-input space.

        Exposed for auxiliary losses that need kernel-space geometry — a
        penalty on distances between points, for instance — which would
        otherwise have no way to reach this space: the transform is applied
        inside :meth:`forward_and_loss`, and re-deriving it in the caller would
        run the DKL MLP twice per step and update the ``l2_radial`` BatchNorm's
        running statistics twice.  Feed the result back through
        :meth:`forward_and_loss` with ``pretransformed=True`` so the step
        transforms once.

        The output is in the same space as the inducing points and carries
        gradient to the backbone, and unlike the internal path it is *not* cast
        to the GP's working precision.
        """
        return self._transform_features(features)

    @staticmethod
    def _flatten_points(
        features: Float[torch.Tensor, "... dim"],
    ) -> tuple[Float[torch.Tensor, "points dim"], torch.Size]:
        """Flatten all leading dims into a single point dimension."""
        lead = features.shape[:-1]
        return features.reshape(-1, features.shape[-1]), lead

    @torch.no_grad()
    def set_inducing_points(
        self, points: Float[torch.Tensor, "*tasks n_inducing input_dim"]
    ) -> None:
        """Re-seed inducing locations from collected features.

        Accepts ``(M, D)`` (shared across tasks) or ``(num_tasks, M, D)`` in
        the *raw feature* space (the DKL extractor, if any, is applied here).
        The variational mean is zeroed and the variational covariance reset to
        a small identity so GP-side optimization restarts cleanly.
        """
        base = self.gp_layer.variational_strategy.base_variational_strategy
        device = base.inducing_points.device

        # Apply the same DKL + feature-norm transform used at inference so the
        # inducing points live in the same (normalized) GP-input space.
        if self.feature_extractor is not None:
            fe_device = next(self.feature_extractor.parameters()).device
            points = points.to(fe_device)
        points = self._transform_features(points)
        points = self._init_inducing(
            points, points.shape[-2], self.gp_input_dim, self.num_tasks
        )
        if self._use_double:
            points = points.double()
        points = points.to(device)

        base.inducing_points.data.copy_(points)
        vd = base._variational_distribution
        vd.variational_mean.data.zero_()
        m = points.shape[-2]
        eye = torch.eye(m, device=device, dtype=vd.chol_variational_covar.dtype)
        vd.chol_variational_covar.data.copy_(
            (eye * 0.01).expand_as(vd.chol_variational_covar)
        )

    def forward(
        self, features: Float[torch.Tensor, "... dim"]
    ) -> gpytorch.distributions.MultitaskMultivariateNormal:
        r"""Forward pass returning the per-point multitask predictive distribution.

        Parameters
        ----------
        features : Float[torch.Tensor, "... dim"]
            Per-point features from the backbone; any leading dims are
            flattened into the point dimension.

        Returns
        -------
        gpytorch.distributions.MultitaskMultivariateNormal
            Predictive distribution over ``(P, num_tasks)`` in float (the GP's
            working precision); reshape via :meth:`predict` for original dtype.
        """
        flat, _ = self._flatten_points(features)
        with self._gp_context():
            return self.gp_layer(self._apply_fe(flat))

    def forward_and_loss(
        self,
        features: Float[torch.Tensor, "... dim"],
        target: Float[torch.Tensor, "... tasks"],
        beta: float = 1.0,
        pretransformed: bool = False,
        return_variance: bool = False,
    ) -> (
        tuple[Float[torch.Tensor, "... tasks"], Float[torch.Tensor, ""]]
        | tuple[
            Float[torch.Tensor, "... tasks"],
            Float[torch.Tensor, ""],
            Float[torch.Tensor, "... tasks"],
        ]
    ):
        r"""Forward pass returning the predictive mean and negative ELBO.

        Parameters
        ----------
        features : Float[torch.Tensor, "... dim"]
            Per-point features of shape ``(..., D)``.
        target : Float[torch.Tensor, "... tasks"]
            Per-point field targets of shape ``(..., num_tasks)``.
        beta : float, optional
            KL-term weight for the ELBO (for KL annealing). Default ``1.0``.
        pretransformed : bool, optional
            If ``True``, ``features`` are assumed to already be in the GP-input
            space (i.e. the output of :meth:`transform_features`) so the DKL +
            feature-norm transform is skipped and only the precision cast is
            applied. Default ``False``.
        return_variance : bool, optional
            If ``True``, also return the per-point *latent* (epistemic) variance
            of shape ``target`` (with gradient), for auxiliary losses such as a
            within-sample concordance penalty. Default ``False``.

        Returns
        -------
        tuple of Float[torch.Tensor, "... tasks"] and Float[torch.Tensor, ""]
            ``(mean, neg_elbo)`` — predictive mean reshaped to ``target``'s
            shape and the negative ELBO (scalar), both in the caller's dtype. If
            ``return_variance`` is ``True``, a third element is the per-point
            latent variance reshaped to ``target``'s shape.
        """
        orig_dtype = features.dtype
        flat, lead = self._flatten_points(features)
        flat_target = target.reshape(-1, self.num_tasks)
        gp_target = flat_target.double() if self._use_double else flat_target

        # GPyTorch's VariationalELBO scales the KL term by ``num_data / beta``,
        # so beta must stay strictly positive. Floor it to a tiny value so that
        # ``beta -> 0`` (KL annealing warmup) effectively removes the KL term
        # without a division-by-zero.
        self.mll.beta = max(float(beta), 1e-8)
        if pretransformed:
            gp_in = flat.double() if self._use_double else flat
        else:
            gp_in = self._apply_fe(flat)
        with self._gp_context():
            dist = self.gp_layer(gp_in)
            if self.heteroscedastic:
                neg_elbo = self._hetero_neg_elbo(dist, gp_target, gp_in, beta)
            else:
                neg_elbo = -self.mll(dist, gp_target)
        mean = dist.mean.to(orig_dtype).reshape(*lead, self.num_tasks)
        if return_variance:
            var = dist.variance.to(orig_dtype).reshape(*lead, self.num_tasks)
            return mean, neg_elbo.to(orig_dtype), var
        return mean, neg_elbo.to(orig_dtype)

    def loss(
        self,
        features: Float[torch.Tensor, "... dim"],
        target: Float[torch.Tensor, "... tasks"],
        beta: float = 1.0,
    ) -> Float[torch.Tensor, ""]:
        """Compute the (beta-weighted) negative ELBO loss."""
        _, neg_elbo = self.forward_and_loss(features, target, beta=beta)
        return neg_elbo

    @torch.no_grad()
    def predict(
        self, features: Float[torch.Tensor, "... dim"]
    ) -> FieldVariationalGPPrediction:
        r"""Produce per-point predictions with their predictive uncertainty.

        Parameters
        ----------
        features : Float[torch.Tensor, "... dim"]
            Per-point features of shape ``(..., D)``.

        Returns
        -------
        FieldVariationalGPPrediction
            Named tuple ``(mean, variance, lower, upper, epistemic_variance)`` —
            all of shape ``(..., num_tasks)`` in the caller's dtype.  The
            confidence interval is ``mean +/- confidence_z * sqrt(variance)``.
            ``variance`` is the total (epistemic + observation noise) predictive
            variance; ``epistemic_variance`` is the latent GP term alone, which
            is the signal to use for active learning and "where is the model
            uncertain?" maps.
        """
        orig_dtype = features.dtype
        flat, lead = self._flatten_points(features)
        was_training = self.training
        self.eval()
        self.likelihood.eval()
        try:
            with self._gp_context(), gpytorch.settings.fast_pred_var():
                gp_in = self._apply_fe(flat)
                dist = self.gp_layer(gp_in)
                # Latent (epistemic) variance, before the observation-noise
                # floor is added.
                epistemic_var = dist.variance
                if self.heteroscedastic:
                    # Same decomposition, but the observation-noise term now
                    # varies per point instead of being one scalar per channel.
                    mean = dist.mean
                    var = epistemic_var + self._pointwise_noise_var(gp_in)
                else:
                    pred = self.likelihood(dist)
                    mean = pred.mean
                    var = pred.variance
                z = self._confidence_z
                lower = mean - z * var.sqrt()
                upper = mean + z * var.sqrt()
            return FieldVariationalGPPrediction(
                mean=mean.to(orig_dtype).reshape(*lead, self.num_tasks),
                variance=var.to(orig_dtype).reshape(*lead, self.num_tasks),
                lower=lower.to(orig_dtype).reshape(*lead, self.num_tasks),
                upper=upper.to(orig_dtype).reshape(*lead, self.num_tasks),
                epistemic_variance=epistemic_var.to(orig_dtype).reshape(
                    *lead, self.num_tasks
                ),
            )
        finally:
            if was_training:
                self.train()
