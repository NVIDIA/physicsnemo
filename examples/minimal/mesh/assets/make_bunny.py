from pathlib import Path

import pyvista as pv
import torch

from physicsnemo.mesh.io.io_pyvista import from_pyvista
from physicsnemo.mesh.remeshing._remeshing import remesh

mesh = from_pyvista(pv.examples.download_bunny_coarse())
mesh = remesh(mesh.clean().subdivide(levels=3, filter="linear"), 400)
mesh = mesh.rotate(axis="x", angle=torch.pi / 2).rotate(axis="z", angle=torch.pi / 2)

torch.save(mesh, Path(__file__).parent / "bunny.pt")
