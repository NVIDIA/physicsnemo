from typing import Tuple, Union
from torch import Tensor

import torch
import torch.nn as nn
import numpy as np

from functools import partial, cache
from scipy import integrate
from tqdm import tqdm

from ..module import Module
from ._basis import Basis
from ._datatype import SEM

class SEMConv(Module):

    odim: Tuple[int]
    kernel: nn.ParameterList

    def __init__(self,
                 idim: int,
                 odim: Union[int, Tuple[int]],
                 T: Tuple[Basis],
                 kernel_mode: Tuple[int],
                 kernel_size: Tuple[int],
                 kernel_init_std: float = 1e-7):
        """
        """
        super().__init__()
        self.T = nn.ModuleList(T)

        if isinstance(odim, int):
            self.odim = (odim, )
        else:
            self.odim = odim
            odim = np.prod(odim)

        self.kernel = nn.ParameterList()
        for n, (m, s) in enumerate(zip(kernel_mode, kernel_size)):
            self.kernel.append(nn.Parameter(coef:=torch.empty(s * m, idim, odim)))

            torch.nn.init.normal_(coef, std=kernel_init_std)
            self.register_buffer(f"ws_{n}", weight(T[n], s))

    def factor(self, ϕ: SEM, dim: int) -> SEM:
        """
        Factorized SEM convolution.

        Args:
            ϕ: Input SEM feature field.
            dim: Dimension to convolve over.
        """
        coef, ws = self.kernel[dim], getattr(self, f"ws_{dim}")
        out = sem_conv(ϕ.nodal, coef, ws, T=ϕ.T[dim], ndim=ϕ.ndim, dim=dim)
        return ϕ.new(out.reshape(out.shape[:-1] + self.odim))

    def __call__(self, ϕ: SEM) -> SEM:
        return ϕ.new(sum(self.factor(ϕ, n).nodal for n in range(ϕ.ndim)))

# ---------------------------------------------------------------------------- #
#                                  CONVOLUTION                                 #
# ---------------------------------------------------------------------------- #

def kernel(coef: Tensor, xs: Tensor) -> Tensor:
    """
    Evaluate the Fourier kernel.

    Args:
        coef: Fourier coefficients.
        xs: Query coordinates.
    """
    r, i_ = torch.split(coef, (m:=(n:=len(coef)) // 2 + 1, n - m))
    i = torch.zeros_like(r); i[1:n-m+1] = torch.flip(i_, dims=[0])

    k = 2 * torch.pi * torch.arange(m, device=xs.device)
    f = torch.exp(1j * k * xs[..., None]); f[..., 1:-1] *= 2

    return torch.tensordot(f.real, r, 1) \
         - torch.tensordot(f.imag, i, 1)

@cache
def weight(T: Basis, s: int, use_mp: bool = True) -> Tensor:
    """
    """
    print(f"Pre-computing weights for `{T=}` and `{s=}`...")

    eps = torch.finfo(torch.float).eps
    ab = T.grid[..., None] + torch.tensor([-s/2, s/2])

    map_ = map
    if use_mp:
        from concurrent.futures import ThreadPoolExecutor
        map_ = (pool := ThreadPoolExecutor()).map

    def quad(T: Basis, m: int, a: float, b: float) -> Tensor:
        f = lambda x: T.fn(torch.tensor(x))[m]
        y, e = integrate.quad(f, a, b)
        return y

    ws = []
    for i in range(-s//2, s//2 + 1):
        ws.append(w:=[])

        for a, b in tqdm(ab, f"{i=}"):
            a = torch.clip(a - i, -eps, 1 + eps)
            b = torch.clip(b - i, -eps, 1 + eps)

            q = torch.tensor(list(map_(partial(quad, T, a=a, b=b), range(T.m))))
            w.append(torch.linalg.solve(T.fn(T.grid).T, q).tolist())

    if use_mp: pool.shutdown()
    return torch.tensor(ws)

def sem_conv(nodal: Tensor, coef: Tensor, ws: Tensor, *, T: Basis, ndim: int, dim: int):
    """
    Args:
        w: An (s + 1, m, n) array where s is the window size, m is the number
           of quadrature nodes, and n is the number of output nodes.
    """
    n = ndim + dim # mesh dim

    ns = "".join(map(chr, range(120, 120 + ndim)))
    ms = ns.replace(i:=ns[dim], o:=ns[dim].upper())

    pad_r = nodal.index_select(n, torch.arange(0, r:=len(ws)//2, device=nodal.device))
    pad_l = nodal.index_select(n, torch.arange(nodal.shape[n]-r, nodal.shape[n], device=nodal.device))

    f = torch.concatenate([pad_l, nodal, pad_r], dim=n)
    out = []

    for k, w in enumerate(ws):

        x = T.grid + k - r
        xy = T.grid[:, None] - x

        fx = torch.narrow(f, n, k, nodal.shape[n])
        gxy = kernel(coef, xy / (len(ws) - 1))

        eqn = f"{ns}...i, {o}{i}io, {o}{i} -> {ms}...o"
        out.append(torch.einsum(eqn, fx, gxy, w))

    return sum(out)
