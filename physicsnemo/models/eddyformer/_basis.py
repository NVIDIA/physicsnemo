from typing import Protocol
from torch import Tensor

import torch
import torch.nn as nn

import numpy as np

class Basis(Protocol):

    grid: Tensor
    quad: Tensor

    m: int
    f: Tensor

    def fn(self, xs: Tensor) -> Tensor:
        """
        Evaluate basis functions at given points.
        """

    def at(self, coef: Tensor, xs: Tensor) -> Tensor:
        """
        Evaluate basis expansion at given points.
        """
        mat = self.fn(xs)[..., :len(coef)]
        return torch.tensordot(mat, coef, dims=1)

    def modal(self, vals: Tensor) -> Tensor:
        """
        Convert nodal values to modal coefficients.
        """

    def nodal(self, coef: Tensor) -> Tensor:
        """
        Convert modal coefficients to nodal values.
        """

# ---------------------------------------------------------------------------- #
#                                   LEGENDRE                                   #
# ---------------------------------------------------------------------------- #

from numpy.polynomial import legendre

class Legendre(nn.Module, Basis):

    """
    Shifted Legendre polynomials:
    - `(1 - x^2) Pn''(x) - 2 x Pn(x) + n (n + 1) Pn(x) = 0`
    - `Pn^~(x) = Pn(2 x - 1)`
    """

    def extra_repr(self) -> str:
        return f"m={self.m}"

    def __init__(self, m: int, endpoint: bool = False):
        """
        """
        super().__init__()
        self.m = m

        if endpoint: m -= 1
        c = (0, ) * m + (1, )
        dc = legendre.legder(c)

        x = legendre.legroots(dc if endpoint else c)
        y = legendre.legval(x, c if endpoint else dc)

        if endpoint:
            x = np.concatenate([[-1], x, [1]])
            y = np.concatenate([[1], y, [1]])

        w = 1 / y ** 2
        if endpoint: w /= m * (m + 1)
        else: w /= 1 - x ** 2

        self.register_buffer("grid", torch.tensor((1 + x) / 2, dtype=torch.float))
        self.register_buffer("quad", torch.tensor(w, dtype=torch.float))

        self.register_buffer("f", self.fn(self.grid))

    def fn(self, xs: Tensor) -> Tensor:
        """
        """
        P = torch.ones_like(xs), 2 * xs - 1

        for i in range(2, self.m):
            a, b = (i * 2 - 1) / i, (i - 1) / i
            P += a * P[-1] * P[1] - b * P[-2],

        return torch.stack(P, dim=-1)

# --------------------------------- TRANSFORM -------------------------------- #

    def modal(self, vals: Tensor) -> Tensor:
        norm = 2 * torch.arange(self.m, device=vals.device) + 1
        coef = self.f * norm * self.quad[:, None]
        return torch.tensordot(coef.to(vals.dtype).T, vals, dims=1)

    def nodal(self, coef: Tensor) -> Tensor:
        return Legendre.at(self, coef, self.grid.to(coef.dtype))

class LegendreSEM(Legendre):

    def __init__(self, m: int):
        super().__init__(m, True)

        xs = self.grid[:, torch.newaxis]
        mat = super().modal(self.f[:, :-2] * xs * (1 - xs))
        self.register_buffer("inv", torch.linalg.inv(mat[:-2]))

    def at(self, coef: Tensor, xs: Tensor) -> Tensor:
        vals = super().at(coef[2:], xs)
        while xs.ndim < vals.ndim:
            xs = xs.unsqueeze(-1)
        return coef[0] * (1 - xs) + coef[1] * xs + vals * xs * (1 - xs)

    def modal(self, vals: Tensor) -> Tensor:
        """
        """
        xs = self.grid.to(vals.dtype)
        for _ in range(vals.ndim - 1):
            xs = xs.unsqueeze(-1)

        coef = super().modal(vals - (1 - xs) * vals[0] - xs * vals[-1])[:-2]
        return torch.concat([vals[[0, -1], ...], torch.tensordot(self.inv.to(coef.dtype), coef, 1)], axis=0)

    def nodal(self, coef: Tensor) -> Tensor:
        """
        """
        xs = self.grid.to(coef.dtype)
        for _ in range(coef.ndim - 1):
            xs = xs.unsqueeze(-1)

        vals = xs * (1 - xs) * super().nodal(coef[2:])
        return coef[0] * (1 - xs) + coef[1] * xs + vals