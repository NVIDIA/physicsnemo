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

import os

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from physicsnemo.diffusion.noise_schedulers import LinearGaussianNoiseScheduler


class DiffusionDataset_topodiff(Dataset):
    def __init__(self, topologies, vfs_stress_strain, load_im):
        image_size = topologies.shape[1]

        self.topologies = topologies
        self.vfs_stress_strain = vfs_stress_strain
        self.image_size = image_size
        self.load_im = load_im

    def __len__(self):
        return self.topologies.shape[0]

    def __getitem__(self, idx):
        cons = np.zeros((5, self.image_size, self.image_size))

        cons[0] = self.vfs_stress_strain[idx][:, :, 0]
        cons[1] = self.vfs_stress_strain[idx][:, :, 1]
        cons[2] = self.vfs_stress_strain[idx][:, :, 2]
        cons[3] = self.load_im[idx][:, :, 0]
        cons[4] = self.load_im[idx][:, :, 1]

        return np.expand_dims(self.topologies[idx], 0) * 2 - 1, cons


def load_data_topodiff(
    topologies, vfs_stress_strain, load_im, batch_size, deterministic=False
):
    dataset = DiffusionDataset_topodiff(topologies, vfs_stress_strain, load_im)

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def load_data(root, prefix, file_format, num_file_start=0, num_file_end=30000):
    """
    root: path to the folder of training data
    prefix: file prefix to the ground truth topology, boundary condition and stress/strain
    file_format: .npy for the conditions; .png for the ground truth topologies
    """
    data_array = []

    for i in range(num_file_start, num_file_end):
        file = f"{root}{prefix}{i}{file_format}"
        if file_format == ".npy":
            data_array.append(np.load(file))
        elif file_format == ".png":
            data_array.append(np.array(Image.open(file)) / 255)
        else:
            raise NotImplementedError

    return np.array(data_array).astype(np.float64)


def load_data_regressor(root):
    file_list = os.listdir(root)
    idx_list = []
    for file in file_list:
        if file.startswith("gt_topo_"):
            idx = int(file.split(".")[0][8:])
            idx_list.append(idx)
    idx_list.sort()

    topology_array, load_array, pf_array = [], [], []
    for i in idx_list:
        topology_array.append(
            np.array(Image.open(root + "gt_topo_" + str(i) + ".png")) / 255
        )
        load_array.append(np.load(root + "cons_load_array_" + str(i) + ".npy"))
        pf_array.append(np.load(root + "cons_pf_array_" + str(i) + ".npy"))

    labels = np.load(root + "deflections_scaled_diff.npy")
    return (
        np.array(topology_array).astype(np.float64),
        np.array(load_array).astype(np.float64),
        np.array(pf_array).astype(np.float64),
        labels[idx_list],
    )


def load_data_classifier(root):
    """
    root: path to the folder of training data
    prefix: file prefix to the ground truth topology, boundary condition and stress/strain
    file_format: .npy for the conditions; .png for the ground truth topologies
    """
    file_list = os.listdir(root)
    labels = np.load(root + "labels.npy")
    image_list = []
    label_list = []
    for file in file_list:
        if file.startswith("img_"):
            idx = int(file.split(".")[0][4:])
            image = Image.open(root + file)
            image_list.append(np.array(image) / 255)
            label_list.append(labels[idx])

    return np.array(image_list).astype(np.float64), np.array(label_list).astype(
        np.float64
    )


# =============================================================================
# Diffusion Framework Components
# =============================================================================


class DDPMLinearNoiseScheduler(LinearGaussianNoiseScheduler):
    r"""Discrete DDPM noise scheduler with a linear beta schedule.

    Implements the classic DDPM forward process with
    :math:`\beta_t = \text{linspace}(\beta_{\min}, \beta_{\max}, T)` and
    :math:`\bar\alpha_t = \prod_{s=1}^{t}(1-\beta_s)`.

    Time values are discrete integer indices in :math:`[0, T)` represented as
    float tensors.

    Parameters
    ----------
    n_steps : int
        Number of diffusion steps :math:`T`.
    min_beta : float
        Minimum beta in the linear schedule.
    max_beta : float
        Maximum beta in the linear schedule.
    """

    def __init__(
        self,
        n_steps: int = 1000,
        min_beta: float = 1e-4,
        max_beta: float = 0.02,
    ) -> None:
        self.n_steps = n_steps
        self.min_beta = min_beta
        self.max_beta = max_beta

        betas = torch.linspace(min_beta, max_beta, n_steps)
        alphas_individual = 1.0 - betas
        alpha_bars = torch.cumprod(alphas_individual, dim=0)

        # Store precomputed tables (registered as plain tensors, not parameters)
        self._betas = betas
        self._alphas_individual = alphas_individual
        self._alpha_bars = alpha_bars
        self._sqrt_alpha_bars = alpha_bars.sqrt()
        self._sqrt_one_minus_alpha_bars = (1.0 - alpha_bars).sqrt()

    def _index(self, t: Tensor) -> Tensor:
        """Clamp and convert time to integer indices."""
        return t.long().clamp(0, self.n_steps - 1)

    def alpha(self, t: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Signal coefficient :math:`\sqrt{\bar\alpha_t}`."""
        return self._sqrt_alpha_bars.to(t.device)[self._index(t)]

    def alpha_dot(self, t: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Finite-difference approximation of :math:`\dot\alpha(t)`."""
        idx = self._index(t)
        table = self._sqrt_alpha_bars.to(t.device)
        cur = table[idx]
        prev = table[(idx - 1).clamp(min=0)]
        return cur - prev

    def sigma(self, t: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Noise level :math:`\sqrt{1-\bar\alpha_t}`."""
        return self._sqrt_one_minus_alpha_bars.to(t.device)[self._index(t)]

    def sigma_dot(self, t: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Finite-difference approximation of :math:`\dot\sigma(t)`."""
        idx = self._index(t)
        table = self._sqrt_one_minus_alpha_bars.to(t.device)
        cur = table[idx]
        prev = table[(idx - 1).clamp(min=0)]
        return cur - prev

    def sigma_inv(self, sigma: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Inverse mapping: find closest discrete timestep for a given sigma."""
        table = self._sqrt_one_minus_alpha_bars.to(sigma.device)
        # For each sigma value, find the closest index
        diffs = (table.unsqueeze(0) - sigma.reshape(-1, 1)).abs()
        indices = diffs.argmin(dim=1)
        return indices.to(sigma.dtype).reshape(sigma.shape)

    def timesteps(
        self,
        num_steps: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N+1"]:
        r"""Generate discrete DDPM timesteps from ``n_steps-1`` down to ``0``."""
        if num_steps >= self.n_steps:
            indices = torch.arange(self.n_steps - 1, -1, -1, device=device, dtype=dtype)
        else:
            step_indices = torch.arange(num_steps, device=device, dtype=dtype)
            scale = (self.n_steps - 1) / (num_steps - 1)
            indices = (scale * (num_steps - 1 - step_indices)).round()
            if dtype is not None:
                indices = indices.to(dtype)
        zero = torch.zeros(1, device=device, dtype=dtype)
        return torch.cat([indices, zero])

    def sample_time(
        self,
        N: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N"]:
        r"""Sample N random discrete timesteps uniformly in :math:`[0, T)`."""
        t = torch.randint(0, self.n_steps, (N,), device=device)
        if dtype is not None:
            t = t.to(dtype)
        else:
            t = t.float()
        return t

    def loss_weight(self, t: Float[Tensor, " N"]) -> Float[Tensor, " N"]:
        r"""Loss weight: :math:`\alpha(t)^2/\sigma(t)^2`.

        This ensures that ``loss_weight * ||x0_hat - x0||^2`` is equivalent to
        the uniform-weighted epsilon-MSE loss ``||eps_hat - eps||^2``.
        """
        a = self.alpha(t)
        s = self.sigma(t)
        return (a / s) ** 2

    def betas_at(self, t: Tensor) -> Tensor:
        """Return individual beta values at discrete timestep ``t``."""
        return self._betas.to(t.device)[self._index(t)]

    def alphas_individual_at(self, t: Tensor) -> Tensor:
        """Return individual alpha values (1 - beta) at timestep ``t``."""
        return self._alphas_individual.to(t.device)[self._index(t)]


class DDPMSolver:
    r"""Discrete DDPM reverse-step solver.

    Implements the DDPM posterior mean update:

    .. math::
        \mu_\theta(\mathbf{x}_t, t)
        = \frac{1}{\sqrt{\alpha_t}}
        \left(\mathbf{x}_t
        - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\hat\epsilon_\theta
        \right)

    with optional posterior noise:

    .. math::
        \mathbf{x}_{t-1} = \mu_\theta + \sqrt{\tilde\beta_t}\,\mathbf{z}

    Parameters
    ----------
    score_predictor : callable
        A score predictor ``(x, t) -> score``. The solver converts score to
        epsilon internally via ``scheduler.score_to_epsilon``.
    noise_scheduler : DDPMLinearNoiseScheduler
        Scheduler providing alpha/sigma/beta values.
    stochastic : bool
        If True, inject posterior noise at each step (except the final one).
    """

    def __init__(self, score_predictor, noise_scheduler, stochastic=True):
        self.score_predictor = score_predictor
        self.scheduler = noise_scheduler
        self.stochastic = stochastic

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        """Perform one discrete DDPM reverse step from ``t_cur`` to ``t_next``."""
        score = self.score_predictor(x, t_cur)
        eps = self.scheduler.score_to_epsilon(score, t_cur)

        # Individual (non-cumulative) alpha and beta at t_cur
        alpha_ind = self.scheduler.alphas_individual_at(t_cur)
        beta = self.scheduler.betas_at(t_cur)
        sigma_t = self.scheduler.sigma(t_cur)

        # Reshape for broadcasting
        def _bc(v):
            return v.reshape(-1, *([1] * (x.ndim - 1)))

        # DDPM posterior mean
        mu = _bc(1.0 / alpha_ind.sqrt()) * (x - _bc(beta / sigma_t) * eps)

        # Posterior noise (skip at t=0)
        if self.stochastic:
            noise_mask = (t_next > 0).float()
            if noise_mask.any():
                mu = mu + _bc(noise_mask) * _bc(beta.sqrt()) * torch.randn_like(x)

        return mu


class ClassifierGuidance:
    r"""DPS-compatible classifier guidance on noisy samples.

    Computes :math:`\gamma\,\nabla_{\mathbf{x}}\log p(y|\mathbf{x}_t, t)` by
    running a time-aware classifier on the noisy sample and differentiating.

    Implements the :class:`~physicsnemo.diffusion.guidance.DPSGuidance` protocol.

    Parameters
    ----------
    classifier : callable
        Classifier ``(x, time_steps=t) -> logits``.
    labels : Tensor
        Target class labels of shape ``(B,)``.
    scale : float
        Guidance scale :math:`\gamma`.
    """

    def __init__(self, classifier, labels, scale=1.0):
        self.classifier = classifier
        self.labels = labels
        self.scale = scale

    def __call__(
        self,
        x: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
        x_0: Float[Tensor, " B *dims"],
    ) -> Float[Tensor, " B *dims"]:
        with torch.enable_grad():
            x_grad = x.detach().requires_grad_(True)
            logits = self.classifier(x_grad, time_steps=t.long())
            loss = F.cross_entropy(logits, self.labels[: x.shape[0]])
            grad = torch.autograd.grad(loss, x_grad)[0]
        # Negate: grad(CE) = -grad(log p(y|x)), and we want +grad(log p(y|x))
        return -self.scale * grad
