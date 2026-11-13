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

"""Tests for the recipe-local global-data mesh reader and its terminal step."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import apply_model_config, build_dataset, load_dataset_config
from merge_global_data import MeshReaderWithGlobalData, SurfaceMeshToDomainMesh
from omegaconf import OmegaConf
from tensordict import TensorDict

from physicsnemo.datapipes.protocols import DatasetBase
from physicsnemo.mesh import DomainMesh, Mesh

_RECIPE_ROOT = Path(__file__).resolve().parents[1]

#: The tunnel boundaries DrivAerML's case files store beside the vehicle.
DOMAIN_BOUNDARIES = ("inlet", "no_slip", "outlet", "slip")


def tunnel_boundaries() -> dict[str, Mesh]:
    """Four two-triangle squares standing for the tunnel's boundaries, apart."""
    square = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]])
    return {
        name: Mesh(
            points=square * (i + 2) + torch.tensor([-5.0, -4.0, -1.0 - i]), cells=cells
        )
        for i, name in enumerate(DOMAIN_BOUNDARIES)
    }


def write_surface_case(root: Path, n_cells: int = 30) -> DomainMesh:
    """One DrivAerML-shaped case file under ``root``: a vehicle with targets, the tunnel."""
    g = torch.Generator().manual_seed(0)
    vehicle = Mesh(
        points=torch.randn(3 * n_cells, 3, generator=g),
        cells=torch.arange(3 * n_cells).reshape(n_cells, 3),
        cell_data={
            "pMeanTrim": torch.randn(n_cells, generator=g),
            "wallShearStressMeanTrim": torch.randn(n_cells, 3, generator=g),
        },
        global_data={"TimeValue": torch.tensor(0.0)},
    )
    source = DomainMesh(
        interior=Mesh(points=torch.zeros(0, 3)),
        boundaries={"vehicle": vehicle, **tunnel_boundaries()},
        global_data={
            "U_inf": torch.tensor([10.0, 0.0, 0.0]),
            "rho_inf": torch.tensor(2.0),
            "p_inf": torch.tensor(0.0),
            "L_ref": torch.tensor(5.0),
        },
    )
    path = root / "run_001" / "sample.pdmsh"
    path.parent.mkdir()
    source.save(path)
    return source


def test_merge_global_data_preserves_geometry_caches(tmp_path):
    """Merging case metadata leaves reusable geometry caches populated."""
    mesh = Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        cells=torch.tensor([[0, 1, 2]]),
        global_data={"existing": torch.tensor(1.0)},
    )
    expected_centroids = mesh.cell_centroids.clone()
    mesh.save(tmp_path / "sample.pmsh")
    TensorDict(
        {"external": torch.tensor(2.0)},
        batch_size=[],
    ).save(tmp_path / "global_data")

    reader = MeshReaderWithGlobalData(
        tmp_path,
        pattern="*.pmsh",
        merge_global_data_from="../global_data",
    )
    result = reader._load_sample(0)

    assert set(result.global_data.keys()) == {"existing", "external"}
    assert ("cell", "centroids") in result._cache.keys(
        include_nested=True,
        leaves_only=True,
    )
    torch.testing.assert_close(result.cell_centroids, expected_centroids)


def test_reader_loads_the_domain_boundaries_whole_beside_the_subsampled_surface(
    tmp_path,
):
    """With domain_boundaries the sample is a DomainMesh: the same surface draw as
    its interior, the named siblings whole, the case's global data at its level."""
    source = write_surface_case(tmp_path)
    kw = dict(
        pattern="run_*/*.pdmsh/_tensordict/boundaries/vehicle",
        subsample_n_cells=10,
        merge_global_data_from="../../global_data",
    )
    bare = MeshReaderWithGlobalData(tmp_path, **kw)
    whole = MeshReaderWithGlobalData(
        tmp_path, domain_boundaries=list(DOMAIN_BOUNDARIES), **kw
    )
    for r in (bare, whole):
        r.set_generator(torch.Generator().manual_seed(7))
    mesh, _ = bare[0]
    assert isinstance(mesh, Mesh)  # without the argument, the sample is as before
    domain, meta = whole[0]
    assert isinstance(domain, DomainMesh) and domain.interior.n_cells == 10
    assert torch.equal(domain.interior.points, mesh.points)
    assert torch.equal(domain.interior.cells, mesh.cells)
    assert set(domain.boundary_names) == set(DOMAIN_BOUNDARIES)
    assert set(meta["boundary_names"]) == set(DOMAIN_BOUNDARIES)
    for name in DOMAIN_BOUNDARIES:
        assert torch.equal(
            domain.boundaries[name].points, source.boundaries[name].points
        )
        assert torch.equal(domain.boundaries[name].cells, source.boundaries[name].cells)
    assert set(domain.global_data.keys()) == set(mesh.global_data.keys())
    with pytest.raises(FileNotFoundError, match="domain boundary 'roof'"):
        MeshReaderWithGlobalData(tmp_path, domain_boundaries=["roof"], **kw)[0]
    with pytest.raises(ValueError, match="twice"):
        MeshReaderWithGlobalData(tmp_path, domain_boundaries=["inlet", "inlet"], **kw)
    ### Domain parallelism gives a sharded sample, which the boundaries cannot join.
    sharded = dict(domain_parallel={}, device_mesh=SimpleNamespace(ndim=1))
    MeshReaderWithGlobalData(tmp_path, **kw, **sharded)  # allowed without them
    with pytest.raises(ValueError, match="domain parallelism"):
        MeshReaderWithGlobalData(
            tmp_path, domain_boundaries=list(DOMAIN_BOUNDARIES), **kw, **sharded
        )


def test_surface_mesh_to_domain_mesh_keeps_the_other_boundaries():
    """On a Mesh it is MeshToDomainMesh. On a DomainMesh whose interior is the
    surface, it converts the surface and keeps the other boundaries."""
    g = torch.Generator().manual_seed(0)
    surface = Mesh(
        points=torch.randn(6, 3, generator=g),
        cells=torch.tensor([[0, 1, 2], [3, 4, 5]]),
        cell_data={"pressure": torch.tensor([10.0, 20.0])},
    )
    walls = tunnel_boundaries()
    to_domain = SurfaceMeshToDomainMesh(cell_data_targets=["pressure"])
    assert set(to_domain(surface).boundary_names) == {"vehicle"}
    sample = DomainMesh(
        interior=surface, boundaries=walls, global_data={"U_inf": torch.ones(3)}
    )
    out = to_domain.apply_to_domain(sample)
    assert set(out.boundary_names) == {"vehicle", *DOMAIN_BOUNDARIES}
    for name in DOMAIN_BOUNDARIES:
        assert torch.equal(out.boundaries[name].points, walls[name].points)
    torch.testing.assert_close(out.interior.points, surface.cell_centroids)
    torch.testing.assert_close(
        out.interior.point_data["pressure"], surface.cell_data["pressure"]
    )
    assert torch.equal(out.global_data["U_inf"], torch.ones(3))
    clash = DomainMesh(interior=surface, boundaries={"vehicle": walls["inlet"]})
    with pytest.raises(ValueError, match="already has a boundary 'vehicle'"):
        to_domain.apply_to_domain(clash)


def _surface_sample(
    ds_cfg, root: Path, seed: int = 0, sampling_resolution: int = 10
) -> DomainMesh:
    """The first augmented sample of ``drivaer_ml_surface.yaml``-shaped ``ds_cfg``."""
    ds_cfg = OmegaConf.merge(
        ds_cfg,
        {
            "dataset_paths": {"drivaer_ml": str(root)},
            "sampling_resolution": sampling_resolution,
        },
    )
    dataset = build_dataset(ds_cfg, augment=True, device=None, num_workers=1)
    try:
        dataset.set_generator(torch.Generator().manual_seed(seed))
        domain, _ = dataset[0]
    finally:
        DatasetBase.close(dataset)
    return domain


def test_surface_dataset_samples_are_unchanged_by_the_terminal_step(tmp_path):
    """drivaer_ml_surface.yaml ends with SurfaceMeshToDomainMesh: for a model that
    reads no domain boundaries, its samples equal MeshToDomainMesh's, bit for bit."""
    write_surface_case(tmp_path)
    ours = load_dataset_config(_RECIPE_ROOT / "datasets" / "drivaer_ml_surface.yaml")
    assert ours.pipeline.transforms[-1]._target_.endswith(".SurfaceMeshToDomainMesh")
    theirs = OmegaConf.merge(ours, {})
    theirs.pipeline.transforms[-1]._target_ = "${dp:MeshToDomainMesh}"
    a, b = _surface_sample(ours, tmp_path), _surface_sample(theirs, tmp_path)
    assert a.boundary_names == b.boundary_names == ["vehicle"]
    for name in ("interior", "global_data", "boundaries"):
        left, right = getattr(a, name), getattr(b, name)
        keys = sorted(map(str, left.keys(include_nested=True, leaves_only=True)))
        assert keys == sorted(
            map(str, right.keys(include_nested=True, leaves_only=True))
        )
    torch.testing.assert_close(a.interior.points, b.interior.points, rtol=0, atol=0)
    for key in a.interior.point_data.keys(include_nested=True, leaves_only=True):
        assert torch.equal(a.interior.point_data[key], b.interior.point_data[key])
    va, vb = a.boundaries["vehicle"], b.boundaries["vehicle"]
    assert torch.equal(va.points, vb.points) and torch.equal(va.cells, vb.cells)
    for key in va.cell_data.keys(include_nested=True, leaves_only=True):
        assert torch.equal(va.cell_data[key], vb.cell_data[key])
    for key in a.global_data.keys(include_nested=True, leaves_only=True):
        assert torch.equal(a.global_data[key], b.global_data[key])


def test_model_config_adds_the_flipped_tunnel_to_the_surface_samples(tmp_path):
    """isla_surface.yaml's dataset keys add the tunnel, moved with the vehicle by
    the dataset's transforms, reverse the vehicle and declare its new orientation."""
    source = write_surface_case(tmp_path)
    plain = load_dataset_config(_RECIPE_ROOT / "datasets" / "drivaer_ml_surface.yaml")
    model = OmegaConf.load(_RECIPE_ROOT / "conf" / "model" / "isla_surface.yaml")
    adapted = apply_model_config(plain, model)
    assert (plain.surface_normals_point, adapted.surface_normals_point) == (
        "inward",
        "outward",
    )
    ### Above the vehicle's 30 cells, so it keeps its stored points in order.
    theirs, ours = (
        _surface_sample(cfg, tmp_path, sampling_resolution=100)
        for cfg in (plain, adapted)
    )
    assert set(ours.boundary_names) == {"vehicle", *DOMAIN_BOUNDARIES}
    vehicle, stored = ours.boundaries["vehicle"], theirs.boundaries["vehicle"]
    assert torch.equal(vehicle.points, stored.points)
    assert torch.equal(vehicle.cells, stored.cells[:, [0, 2, 1]])
    assert torch.equal(vehicle.cell_data["normals"], -stored.cell_data["normals"])
    assert torch.equal(ours.interior.points, theirs.interior.points)
    for key, value in stored.global_data.items(include_nested=True, leaves_only=True):
        assert torch.equal(vehicle.global_data[key], value)
    ### The tunnel moved with the vehicle: the same rigid motion and scaling, so
    ### every distance between their points shrank by 1 / L_ref.
    raw = torch.cat(
        [source.boundaries[name].points for name in ("vehicle", *DOMAIN_BOUNDARIES)]
    )
    moved = torch.cat(
        [ours.boundaries[name].points for name in ("vehicle", *DOMAIN_BOUNDARIES)]
    )
    exact = "donot_use_mm_for_euclid_dist"
    torch.testing.assert_close(
        torch.cdist(moved, moved, compute_mode=exact),
        torch.cdist(raw, raw, compute_mode=exact) / 5.0,
    )
