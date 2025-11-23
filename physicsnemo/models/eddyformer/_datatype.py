from typing import Tuple
from torch import Tensor

import torch

from dataclasses import dataclass, replace
from functools import cached_property

from ._basis import Basis, Legendre

def interp1d(value: Tensor, xs: Tensor, method: str) -> Tensor:
  """
    Interpolate from 1D regular grid to a target points.

    Args:
      value: Values on a uniform grid along the first axis.
      xs: Resolution or an array normalized by the domain size.
      method: Interpolation method. One of "fft", "linear", or
              f"lag{n}" for n-point Lagrangian interpolation.
  """
  if method == "fft":
    coef = torch.fft.rfft(value, dim=0, norm="forward")

    k = 2 * torch.pi * torch.arange(len(coef), device=xs.device)
    f = torch.exp(1j * k * xs[..., None]); f[..., 1:-1] *= 2
    return torch.tensordot(f.real, coef.real, dims=1) \
         - torch.tensordot(f.imag, coef.imag, dims=1)

  if method.startswith("lag"):
    n_points = int(method[3:])

    assert n_points % 2 == 0
    r = n_points // 2 - 1

    n = len(value)

    i = (xs * (N := n - 1)).int()
    i = torch.clip(i, r, n - n_points + r)

    # 1. pad the input grid

    v_pad = value, value[:r+2]

    if r > 0: v_pad = (value[-r:], ) + v_pad
    value = torch.concatenate(v_pad, dim=0)

    # 2. construct polynomials

    out = 0

    for j in range(n_points):
        lag = value[i + j]

        for k in range(n_points):
            if j == k: continue
            fac = xs - (i + k - r) / N
            while fac.ndim < lag.ndim:
               fac = fac.unsqueeze(-1)
            lag *= fac * N / (j - k)

        out += lag
    return out

  raise ValueError(f"invalid interpolation {method=}")

# ---------------------------------------------------------------------------- #
#                               SPECTRAL ELEMENT                               #
# ---------------------------------------------------------------------------- #

@dataclass
class SEM:

    """
    Spectral element expansion. The sub-domain partition is
    given by the `mesh` attribute. The spectral coefficients
    of each element is stored in the first channel dimension, 
    whose size must equal to the number of elements.
    """

    T_: str

    # Mesh

    size: Tensor
    mesh: Tuple[int]

    # Data

    mode_: Tuple[int] = None
    nodal: Tensor = None

    @property
    def ndim(self) -> int:
        return len(self.mesh)

    @property
    def mode(self) -> Tuple[int]:
        if self.mode_: return self.mode_
        return self.nodal.shape[:self.ndim]

    @property
    def use_elem(self) -> bool:
        return self.T_.endswith("elem")

    @staticmethod
    def basis(T_: str) -> Basis:
        if T_.startswith("leg"): return Legendre
        raise ValueError(f"invalid basis {T_=}")

    @cached_property
    def T(self) -> Tuple[Basis]:
        """
        Basis on each dimension.
        """
        T = self.basis(self.T_)
        return tuple(map(T, self.mode))

    def to(self, mode: Tuple[int]) -> "SEM":
        """
        Resample to another mode.

        Args:
            mode: Number of modes.
        """
        out = SEM(self.T_, self.size, self.mesh, mode)

        value = self.nodal
        for n in range(self.ndim):
            coef = self.T[n].modal(value)
            if (pad:=out.mode[n] - self.mode[n]) <= 0: coef = coef[:mode[n]]
            else: coef = torch.concat([coef, torch.zeros(pad, *coef.shape[1:], device=coef.device)], dim=0)
            value = out.T[n].nodal(coef).movedim(0, self.ndim - 1)

        return out.new(value)

    def at(self, *xs: Tensor, uniform: bool = False) -> Tensor:
        """
        Evaluate on rectilinear grids.

        Args:
            xs: Coordinate of each dimension.
            uniform: Whether `xs` are uniformly spaced.
        """
        value = self.nodal
        for n in range(self.ndim):
            x = xs[n] / self.size[n]
            coef = self.T[n].modal(value)

            # indices of each global coordinate `x`
            idx = torch.floor(x * float(self.mesh[n])).int()
            idx = torch.minimum(idx, torch.tensor(self.mesh[n] - 1, device=idx.device))

            # global coordinate to local coordinate
            ys = x * float(self.mesh[n]) - torch.arange(self.mesh[n], device=x.device)[idx]

            if not uniform:

                # coefficients where each `x` belongs
                coef = coef.movedim(self.ndim, 0)[idx]

                # evaluate at each coordinate and move the output axis to the last dimension
                # after `ndim` iterations, the axes are automatically rolled to the correct order
                value = torch.vmap(self.T[n].at, out_dims=self.ndim - 1)(coef, ys)

            else:

                # coordinates within each element
                ys = ys.reshape(self.mesh[n], -1)

                # batched evaluation of all coordinates
                value = torch.vmap(self.T[n].at, (self.ndim, 0))(coef, ys)
                value = torch.movedim(value.flatten(end_dim=1), 0, self.ndim - 1)

        return value

# ---------------------------------- COORDS ---------------------------------- #

    @cached_property
    def grid(self) -> Tensor:
        axes = [self.T[n].grid.to(self.size.device) for n in range(self.ndim)]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)

    @cached_property
    def coords(self) -> Tensor:
        local = self.grid
        for _ in range(self.ndim):
            local = local.unsqueeze(self.ndim)
        return self.origins + local * self.lengths

    @cached_property
    def origins(self) -> Tensor:
        left = [torch.arange(m, device=self.size.device) / m for m in self.mesh]
        return torch.stack(torch.meshgrid(*left, indexing="ij"), dim=-1) * self.size

    @cached_property
    def lengths(self) -> Tensor:
        ns = torch.tensor(self.mesh, device=self.size.device)
        return self.size / ns.float()

# --------------------------------- DATA TYPE -------------------------------- #

    def new(self, nodal: Tensor) -> "SEM":
        assert nodal.shape[:self.ndim] == self.mode
        return replace(self, mode_=None, nodal=nodal)

    def eval(self, resolution: Tuple[int]) -> Tensor:
        xs = [torch.linspace(0, s, n, device=self.size.device) for n, s in zip(resolution, self.size)]
        return self.at(*xs, uniform=False)

    def from_grid(self, value: Tensor, method: str) -> "SEM":
        """
        Interpolate grid values to a target datatype.

        Args:
            value: Input tensor (include boundary points).
            method: Interpolation method along each axis.
                    See `interp1d::method` for details.
        """
        xs = self.coords / self.size
        for n in range(self.ndim):

            # interpolate at each collocation points. `idx` is the
            # index of the elements along the `n`'th dimension.
            idx = [slice(None) if i == n else 0 for i in range(self.ndim)]
            value = interp1d(value, xs[tuple(idx * 2 + [n])], method)

            # roll the output. The interpolated values have shape `(mode, mesh)`,
            # which are moved to the middle (`ndim - 1`) and the end (`ndim + n`) of
            # the dimensions. After `ndim` iterations, all axes are ordered correctly.
            value = torch.moveaxis(value, (0, 1), (self.ndim - 1, self.ndim + n))

        return self.new(value)
