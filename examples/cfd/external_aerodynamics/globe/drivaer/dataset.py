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

"""Dataset loading and preprocessing for the GLOBE DrivAerML 3D case study.

Reads DrivAerML simulation outputs (VTP car body surfaces, geometry CSVs,
force coefficient CSVs), triangulates and assembles nondimensionalized
prediction targets.  The GLOBE boundary mesh is created at load time by
randomly subsampling cells from the cached surface mesh.
"""

import csv
import os
from pathlib import Path
from typing import Literal, Self, Sequence

import pyvista as pv
import torch
from jaxtyping import Float
from tensordict import TensorClass, TensorDict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.experimental.utils import CachedPreprocessingDataset
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.primitives.planar import structured_grid
from physicsnemo.mesh.projections import embed
from physicsnemo.mesh.remeshing import partition_cells
from physicsnemo.utils.logging import PythonLogger

logger = PythonLogger("globe.drivaer.dataset")

### Reference conditions (constant across all DrivAerML runs)
# Confirmed by domino_nim_finetuning/src/openfoam_datapipe.py and cross-checked
# against the ratio pMeanTrim / CpMeanTrim at stagnation points.
U_INF = 38.89  # m/s  (140 km/h freestream velocity)
Q_INF = 0.5 * U_INF**2  # ~756 m²/s²  (kinematic dynamic pressure)
NU = 1.5e-5  # m²/s  (kinematic viscosity of air)

### DrivAerML CFD domain geometry (constant across all 500 runs)
# The ground boundary layer starts at X_BL in the CFD setup (see Ashton et al.,
# "DrivAerML", arXiv:2408.11969v2, Sect. 3.3 / Fig. 11).  Upstream of X_BL the
# floor is inviscid ("slip"); downstream it is a viscous wall ("no-slip").
X_BL = -2.339  # m  (slip-to-noslip floor transition)

Split = Literal["train", "validation"]


def create_domain_boundaries(
    ground_z: float,
    *,
    x_bl: float = X_BL,
    x_upstream: float = -10,
    x_downstream: float = 10,
    y_half_width: float = 11,
    n_x: int = 20,
    n_y: int = 20,
) -> dict[str, Mesh[2, 3]]:
    """Create triangulated ground-plane boundary meshes for the CFD domain.

    Returns two horizontal Mesh[2, 3] patches at ``z = ground_z``:

    - ``"slip_floor"``: inviscid upstream ground, from *x_upstream* to *x_bl*.
    - ``"no_slip_floor"``: viscous ground wall, from *x_bl* to *x_downstream*.

    These encode the CFD domain floor boundary conditions so that GLOBE's
    boundary-to-boundary communication can learn ground-proximity effects
    (underbody flow, front lift, diffuser behavior).

    Args:
        ground_z: z-coordinate of the ground plane (tire contact level).
        x_bl: Streamwise location of the slip-to-noslip transition.
        x_upstream: Upstream extent of the slip floor.
        x_downstream: Downstream extent of the no-slip floor.
        y_half_width: Half-width of each ground patch in the y direction.
        n_x: Number of grid points in the x direction per patch.
        n_y: Number of grid points in the y direction per patch.

    Returns:
        ``{"slip_floor": Mesh, "no_slip_floor": Mesh}``
    """

    def _ground_patch(x_min: float, x_max: float) -> Mesh[2, 3]:
        mesh = embed(
            structured_grid.load(
                x_min=x_min,
                x_max=x_max,
                y_min=-y_half_width,
                y_max=y_half_width,
                n_x=n_x,
                n_y=n_y,
            ),
            target_n_spatial_dims=3,
        ).translate([0.0, 0.0, ground_z])
        # Flip cell winding so normals point upward (+z, into the fluid domain)
        return Mesh(points=mesh.points, cells=mesh.cells[:, [0, 2, 1]])

    return {
        "slip_floor": _ground_patch(x_upstream, x_bl),
        "no_slip_floor": _ground_patch(x_bl, x_downstream),
    }


class DrivAerMLSample(TensorClass):
    """Single preprocessed DrivAerML sample for GLOBE training / inference.

    Attributes:
        prediction_mesh: Mesh of points where GLOBE predicts output fields.
            Contains ``point_data`` with nondimensional targets (``C_p``,
            ``C_f``) and cell connectivity for visualization and force
            integration.
        boundary_meshes: Boundary condition meshes keyed by BC type:

            - ``"vehicle"``: car body surface (randomly subsampled at load
              time by :meth:`DrivAerMLDataSet.__getitem__`).
            - ``"no_slip_floor"``: viscous ground plane downstream of X_BL.
            - ``"slip_floor"``: inviscid ground plane upstream of X_BL.

            The domain floor meshes are created during preprocessing via
            :func:`create_domain_boundaries` and cached alongside the sample.
        reference_lengths: Per-sample reference lengths (``L_ref``,
            ``delta_turb``) used for GLOBE multiscale kernel construction.
        dimensional_constants: ``U_inf``, ``q_inf`` for re-dimensionalization.
        aero_coefficients: Ground-truth ``Cd``, ``Cl``, ``Cs`` from the
            simulation, for evaluation of integrated force predictions.
    """

    prediction_mesh: Mesh
    boundary_meshes: TensorDict[str, Mesh]
    reference_lengths: TensorDict[str, Float[torch.Tensor, ""]]
    dimensional_constants: TensorDict
    aero_coefficients: TensorDict

    @property
    def model_input_kwargs(self) -> dict:
        """Keyword arguments for :meth:`GLOBE.forward`."""
        return {
            "prediction_points": self.prediction_mesh.points,
            "boundary_meshes": self.boundary_meshes,
            "reference_lengths": self.reference_lengths,
            "global_data": None,
        }

    def prepare(
        self,
        n_prediction_points: int,
        device: torch.device | str,
        *,
        randomize_vehicle: bool = False,
    ) -> Self:
        """Subsample prediction points, precompute boundary geometry, transfer to device.

        This consolidates the per-sample preparation that runs between cache
        loading and model forward.  Designed to be called from a prefetch
        worker thread (CPU-bound) so preparation of sample N+1 overlaps with
        GPU processing of sample N.

        Args:
            n_prediction_points: Maximum number of prediction points to keep.
                If the mesh has fewer points, all are kept.
            device: Target device for the returned sample.
            randomize_vehicle: If True, sample random points inside each
                vehicle boundary cell instead of using centroids. Typically
                enabled during training for data augmentation.

        Returns:
            This sample, modified in-place and moved to *device*.
        """
        n_points = min(n_prediction_points, self.prediction_mesh.n_points)
        mask = torch.randint(self.prediction_mesh.n_points, (n_points,))
        self.prediction_mesh = self.prediction_mesh.to_point_cloud().slice_points(mask)

        for bc_type, mesh in self.boundary_meshes.items():
            if bc_type == "vehicle" and randomize_vehicle:
                mesh._cache["cell", "centroids"] = mesh.sample_random_points_on_cells()
            else:
                _ = mesh.cell_centroids
            _ = mesh.cell_areas
            _ = mesh.cell_normals

        return self.to(device)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DrivAerMLDataSet(CachedPreprocessingDataset):
    """Disk-cached preprocessing dataset for DrivAerML + GLOBE.

    The cached ``.pt`` files store hyperparameter-invariant data (full-
    resolution surface mesh, reference lengths, aero coefficients).  The
    GLOBE boundary mesh is created on every load by randomly subsampling
    ``n_faces_per_boundary`` cells from the surface mesh, so changing the
    target face count takes effect immediately without invalidating caches.
    """

    def __init__(
        self,
        sample_paths: Sequence[Path | str],
        cache_dir: Path | str | None = None,
        *,
        n_faces_per_boundary: int = 20_000,
    ):
        super().__init__(sample_paths=sample_paths, cache_dir=cache_dir)
        self.n_faces_per_boundary = n_faces_per_boundary

    def __getitem__(self, index) -> DrivAerMLSample:  # ty: ignore[invalid-method-override]
        sample: DrivAerMLSample = super().__getitem__(index)
        sample.boundary_meshes["vehicle"] = self.subsample_mesh(
            sample.prediction_mesh, self.n_faces_per_boundary
        )
        return sample

    @staticmethod
    def subsample_mesh(
        mesh: Mesh[2, 3],
        n_cells: int,
        *,
        geometry_only: bool = True,
        voronoi: bool = False,
    ) -> Mesh[2, 3]:
        """Randomly subsample cells with area correction.

        Selects ``n_cells`` random cells, compacts away unreferenced vertices,
        and corrects cell areas so that the subsampled mesh better represents
        the original surface.

        Two area-correction strategies are available:

        * **Voronoi** (``voronoi=True``): uses :func:`partition_cells` to
          assign every original cell to its nearest subsampled centroid and
          accumulate areas and normals.  This gives locally-correct effective
          areas that approximate the surface Voronoi diagram of the subsampled
          centroids, at the cost of an O(N) nearest-neighbour pass over the
          full mesh each time a sample is loaded.
        * **Uniform** (``voronoi=False``): rescales all subsampled cell areas
          by a single global factor so that total area is conserved.  Much
          faster, but every subsampled cell gets the same scale factor
          regardless of local density.

        Args:
            mesh: Source Mesh to subsample from.
            n_cells: Number of cells to select.
            geometry_only: If ``True`` (default), strip all field data and
                return a geometry-only Mesh (used for GLOBE boundary input).
                If ``False``, preserve ``point_data`` and ``cell_data``
                (used for visualization).
            voronoi: If ``True``, use Voronoi-corrected areas and normals
                via :func:`partition_cells`.  If ``False`` (default), use
                uniform area rescaling (faster, but less accurate).

        Returns:
            Mesh with ``n_cells`` cells and corrected area cache.
        """
        if n_cells <= 0:
            raise ValueError(f"{n_cells=!r} must be positive.")
        indices = torch.randperm(mesh.n_cells)[:n_cells]
        boundary = mesh.slice_cells(indices).clean(
            merge_points=False,
            remove_duplicate_cells=False,
            remove_unused_points=True,
        )

        if geometry_only:
            boundary = Mesh(points=boundary.points, cells=boundary.cells)

        if voronoi:
            partition = partition_cells(mesh, seeds=boundary.cell_centroids)
            boundary._cache["cell", "areas"] = partition.cluster_areas
            boundary._cache["cell", "normals"] = partition.cluster_normals
        else:
            total_area = mesh.cell_areas.sum()
            raw_areas = boundary.cell_areas
            boundary._cache["cell", "areas"] = raw_areas * (
                total_area / raw_areas.sum()
            )

        return boundary

    @classmethod
    def get_split_paths(
        cls,
        data_dir: Path,
        split: Split,
    ) -> list[Path]:
        """Read a split CSV and return sample paths for the requested split.

        Each split is defined by a CSV file in ``splits/{split}.csv``
        (relative to this module) with at least a ``run_idx`` column.

        Args:
            data_dir: Root directory containing ``run_N/`` subdirectories.
            split: ``"train"`` or ``"validation"``.

        Returns:
            Sorted list of absolute paths to individual run directories.
        """
        splits_csv = Path(__file__).parent / "splits" / f"{split}.csv"
        with open(splits_csv) as f:
            run_indices = [row["run_idx"] for row in csv.DictReader(f)]
        return sorted(data_dir / f"run_{idx}" for idx in run_indices)

    @classmethod
    def make_dataloader(
        cls,
        sample_paths: Sequence[Path],
        cache_dir: Path,
        *,
        world_size: int = 1,
        rank: int = 0,
        num_workers: int | None = None,
        n_faces_per_boundary: int = 20_000,
    ) -> DataLoader:
        """Create a distributed DataLoader yielding one sample per iteration.

        Args:
            sample_paths: Paths to individual ``run_N/`` directories.
            cache_dir: Directory for disk-cached preprocessed ``.pt`` files.
            world_size: Total distributed ranks.
            rank: This process's rank.
            num_workers: DataLoader worker processes per rank.  When ``None``
                (the default), auto-computed from this rank's CPU affinity
                and the per-node rank count detected from launcher env vars
                (``LOCAL_WORLD_SIZE`` / ``OMPI_COMM_WORLD_LOCAL_SIZE`` /
                ``SLURM_NTASKS_PER_NODE``), reserving one core per GPU the
                rank drives for kernel-launch orchestration and OS work.
            n_faces_per_boundary: Number of cells randomly subsampled from
                the surface mesh to form the GLOBE boundary mesh.

        Returns:
            DataLoader with :class:`DistributedSampler`.
        """
        if num_workers is None:
            n_cpus = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else os.cpu_count() or 1
            )
            ### Per-node rank count, in launcher-precedence order.  torchrun
            ### sets ``LOCAL_WORLD_SIZE``; OpenMPI sets ``OMPI_COMM_WORLD_LOCAL_SIZE``;
            ### pure SLURM srun sets ``SLURM_NTASKS_PER_NODE``.  Default to 1
            ### when running standalone.
            local_world_size = 1
            for env_var in (
                "LOCAL_WORLD_SIZE",
                "OMPI_COMM_WORLD_LOCAL_SIZE",
                "SLURM_NTASKS_PER_NODE",
            ):
                if (raw := os.environ.get(env_var)) is not None:
                    try:
                        local_world_size = max(1, int(raw))
                        break
                    except ValueError:
                        continue
            ### Pigeonhole: if my affinity slice times the number of
            ### co-located ranks exceeds total CPUs, the slice must be
            ### shared with siblings (no per-rank pinning); divide to get
            ### my fair share.  When the launcher pinned each rank
            ### exclusively, this branch is skipped.
            if n_cpus * local_world_size > (os.cpu_count() or n_cpus):
                n_cpus //= local_world_size
            ### One orchestration core per GPU this rank actually drives.
            ### In DDP each rank drives one (its ``local_rank`` GPU)
            ### regardless of how many are visible; single-process
            ### multi-GPU drives every visible GPU.
            n_visible_gpus = max(1, torch.cuda.device_count())
            orchestration = 1 if world_size > 1 else n_visible_gpus
            num_workers = max(0, n_cpus - orchestration)
            if rank == 0:
                logger.info(
                    f"Auto-set DataLoader num_workers={num_workers} "
                    f"(n_cpus={n_cpus}, local_world_size={local_world_size}, "
                    f"world_size={world_size}, n_visible_gpus={n_visible_gpus})"
                )

        dataset = cls(
            sample_paths=sample_paths,
            cache_dir=cache_dir,
            n_faces_per_boundary=n_faces_per_boundary,
        )
        return DataLoader(
            dataset,
            sampler=DistributedSampler(
                dataset=dataset,
                num_replicas=world_size,
                rank=rank,
            ),
            batch_size=None,
            collate_fn=lambda x: x,
            num_workers=num_workers,
            prefetch_factor=1 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess(sample_path: Path) -> DrivAerMLSample:
        """Preprocess a single DrivAerML run into a GLOBE-ready sample.

        Performs the expensive, hyperparameter-invariant work that is cached
        to disk by :class:`CachedPreprocessingDataset`:

            1. Load the VTP car body surface and triangulate.
            2. Compute nondimensional surface fields (C_p, C_f).
            3. Interpolate cell-centered data to mesh vertices.
            4. Parse geometry reference CSV for per-sample reference lengths.
            5. Parse force/moment CSV for ground-truth aero coefficients.
            6. Create CFD domain floor boundaries via
               :func:`create_domain_boundaries`.

        The vehicle body boundary mesh (random cell subsampling, which depends
        on ``n_faces_per_boundary``) is NOT performed here; it runs
        post-cache-load in :meth:`DrivAerMLDataSet.__getitem__`.

        Args:
            sample_path: Path to a ``run_N/`` directory.

        Returns:
            :class:`DrivAerMLSample` with domain floor boundaries cached and
            ``"vehicle"`` boundary populated later by ``__getitem__``.
        """
        sample_dir = Path(sample_path)
        run_idx = sample_dir.name.removeprefix("run_")

        ### Load VTP car body surface
        vtp_path = sample_dir / f"boundary_{run_idx}.vtp"
        if not vtp_path.exists():
            raise FileNotFoundError(f"Missing VTP boundary file: {vtp_path}")
        pv_surface: pv.PolyData = pv.read(vtp_path)

        ### Triangulate and ensure consistent face winding
        if not pv_surface.is_all_triangles:
            pv_surface = pv_surface.triangulate()
        pv_surface.compute_normals(
            cell_normals=True,
            point_normals=False,
            consistent_normals=True,
            auto_orient_normals=False,
            inplace=True,
        )
        logger.info(f"run_{run_idx}: {pv_surface.n_cells:,} surface cells")

        ### Compute nondimensional surface fields (cell-centered)
        pv_surface.cell_data["C_p"] = pv_surface.cell_data["CpMeanTrim"].copy()
        pv_surface.cell_data["C_f"] = (
            pv_surface.cell_data["wallShearStressMeanTrim"] / Q_INF
        )

        for name in (
            "CpMeanTrim",
            "pMeanTrim",
            "pPrime2MeanTrim",
            "wallShearStressMeanTrim",
            "Normals",
        ):
            if name in pv_surface.cell_data:
                del pv_surface.cell_data[name]

        ### Interpolate cell data to vertices for GLOBE prediction targets
        pv_surface_pt = pv_surface.cell_data_to_point_data()
        del pv_surface

        ### Build the prediction Mesh (geometry + prediction targets)
        prediction_mesh = from_pyvista(pv_surface_pt)
        prediction_mesh = Mesh(
            points=prediction_mesh.points,
            cells=prediction_mesh.cells,
            point_data=prediction_mesh.point_data.select("C_p", "C_f").apply(
                torch.Tensor.float
            ),
        )

        ### Parse geometry reference CSV
        geo_ref_path = sample_dir / f"geo_ref_{run_idx}.csv"
        geo_ref = _read_single_row_csv(geo_ref_path)
        l_ref = float(geo_ref["lRef"])
        a_ref = float(geo_ref["aRef"])

        ### Turbulent boundary layer thickness at x = L_ref (flat-plate estimate)
        # δ_turb = 0.37 * L * Re_L^(-1/5), where Re_L = U_inf * L / ν.
        # This physics-derived small scale gives the kernel an O(1)
        # nondimensionalization for near-wall and near-ground interactions.
        re_l = U_INF * l_ref / NU
        delta_turb = 0.37 * l_ref * re_l ** (-0.2)

        ### Parse force/moment CSV
        force_mom_path = sample_dir / f"force_mom_{run_idx}.csv"
        force_mom = _read_single_row_csv(force_mom_path)

        ### Create CFD domain floor boundaries
        ground_z = float(prediction_mesh.points[:, 2].min())
        domain_boundaries = create_domain_boundaries(ground_z)

        return DrivAerMLSample(
            prediction_mesh=prediction_mesh,
            boundary_meshes=TensorDict(domain_boundaries),  # ty: ignore[invalid-argument-type]
            reference_lengths=TensorDict(
                {
                    "L_ref": torch.as_tensor(l_ref),
                    "delta_turb": torch.as_tensor(delta_turb),
                },
            ),
            dimensional_constants=TensorDict(
                {
                    "U_inf": torch.as_tensor(U_INF),
                    "q_inf": torch.as_tensor(Q_INF),
                    "A_ref": torch.as_tensor(a_ref),
                },
            ),
            aero_coefficients=TensorDict(
                {
                    "Cd": torch.as_tensor(float(force_mom["Cd"])),
                    "Cl": torch.as_tensor(float(force_mom["Cl"])),
                    "Cs": torch.as_tensor(float(force_mom["Cs"])),
                },
            ),
        )

    @classmethod
    def load_single_sample(
        cls,
        sample_path: Path,
        *,
        n_faces_per_boundary: int = 20_000,
        device: torch.device | str = "cpu",
    ) -> DrivAerMLSample:
        """Load, preprocess, and fully populate a single sample (no cache).

        Convenience method for inference and visualization scripts that need
        a complete sample without going through the DataLoader.  Equivalent
        to calling ``preprocess`` + adding the vehicle boundary + device
        transfer.

        Args:
            sample_path: Path to a ``run_N/`` directory.
            n_faces_per_boundary: Number of vehicle surface cells to randomly
                subsample.
            device: Target device.

        Returns:
            Fully populated :class:`DrivAerMLSample` on *device*.
        """
        sample = cls.preprocess(sample_path)
        sample.boundary_meshes["vehicle"] = cls.subsample_mesh(
            sample.prediction_mesh, n_faces_per_boundary
        )
        return sample.to(device)


# ---------------------------------------------------------------------------
# Postprocessing / visualization
# ---------------------------------------------------------------------------


def postprocess(
    pred_mesh: Mesh[0, 3],
    sample: DrivAerMLSample,
    *,
    fields: Sequence[str] | None = None,
) -> Mesh[2, 3]:
    """Build a combined pred/true/error Mesh with integrated force coefficients.

    Assembles a single Mesh whose ``point_data`` contains nested
    ``"true"``, ``"pred"``, and ``"error"`` TensorDicts for the selected
    fields, and whose ``global_data`` contains integrated surface force
    coefficients for the prediction and CSV ground-truth coefficients.

    Args:
        pred_mesh: Point-cloud Mesh with predicted field values in
            ``point_data``.
        sample: The preprocessed sample.  ``sample.prediction_mesh`` provides
            the ground-truth fields and cell connectivity;
            ``sample.dimensional_constants["A_ref"]`` is used for
            normalization; ``sample.aero_coefficients`` provides the
            authoritative CSV ground-truth force coefficients.
        fields: Which field names to compare.  If ``None``, uses the sorted
            intersection of pred and true ``point_data`` keys.

    Returns:
        Combined Mesh with ``point_data["true"]``, ``point_data["pred"]``,
        ``point_data["error"]`` for the selected fields, and
        ``global_data["pred"]`` (integrated from predictions) /
        ``global_data["true"]`` (CSV ground truth) each containing
        scalar force coefficient tensors (Cd, Cl, Cs).

    Raises:
        ValueError: If pred_mesh and the sample surface mesh have
            different numbers of points.
    """
    true_mesh = sample.prediction_mesh

    if pred_mesh.n_points != true_mesh.n_points:
        raise ValueError(
            f"Point count mismatch: {pred_mesh.n_points=} vs {true_mesh.n_points=}"
        )

    if fields is None:
        fields = sorted(
            set(pred_mesh.point_data.keys(include_nested=True, leaves_only=True))
            & set(true_mesh.point_data.keys(include_nested=True, leaves_only=True))
        )

    ### Build combined point_data
    pred_selected = pred_mesh.point_data.select(*fields)
    true_selected = true_mesh.point_data.select(*fields)
    error_data: TensorDict = pred_selected.apply(  # ty: ignore[invalid-assignment]
        lambda p, t: p - t, true_selected
    )

    ### Compute integrated force coefficients on predictions
    # pred_mesh is a point cloud (no cells), so we construct a surface
    # mesh with true_mesh's cell connectivity for integration.
    a_ref = float(sample.dimensional_constants["A_ref"])
    pred_surface = Mesh(
        points=true_mesh.points,
        cells=true_mesh.cells,
        point_data=pred_mesh.point_data,
    )

    return Mesh(
        points=true_mesh.points,
        cells=true_mesh.cells,
        point_data=TensorDict(
            {
                "true": true_selected,
                "pred": pred_selected,
                "error": error_data,
            },
            batch_size=[true_mesh.n_points],
        ),
        global_data=TensorDict(
            {
                "pred": compute_surface_force_coefficients(
                    surface_mesh=pred_surface, a_ref=a_ref
                ),
                "true": sample.aero_coefficients,
            }
        ),
    )


def visualize_comparison(
    combined: Mesh[2, 3],
    *,
    save_path: Path | None = None,
    show: bool = False,
    backend: Literal["matplotlib", "pyvista"] = "pyvista",
) -> None:
    """Render a 3D comparison of predicted vs. true surface fields.

    Takes the combined Mesh returned by :func:`postprocess` and draws
    truth / prediction / error rows for each field.

    Args:
        combined: Mesh returned by :func:`postprocess`, with
            ``point_data["true"]``, ``point_data["pred"]``, and
            ``point_data["error"]``.
        save_path: File path for the rendered screenshot.  Defaults to
            ``drivaer_comparison.png`` in the current directory.
        show: Whether to display interactively (requires a display).
        backend: Rendering backend.  ``"pyvista"`` gives high-quality
            GPU-accelerated rendering but needs EGL or OSMesa on
            headless nodes.  ``"matplotlib"`` works everywhere via
            ``mpl_toolkits.mplot3d`` (lower fidelity, but no GPU
            rendering dependency).
    """
    if save_path is None:
        save_path = Path("drivaer_comparison.png")

    ### Flatten nested keys to dot-separated strings for display
    kind_data: dict[str, TensorDict] = {
        key: combined.point_data[key].flatten_keys(".")  # ty: ignore[unresolved-attribute]
        for key in ("true", "pred", "error")
    }
    fields = sorted(kind_data["true"].keys())
    kinds = {"true": "Truth", "pred": "Prediction", "error": "Error"}

    from visualization import visualize_matplotlib, visualize_pyvista

    if backend == "pyvista":
        visualize_pyvista(combined, kind_data, kinds, fields, save_path, show)
    elif backend == "matplotlib":
        visualize_matplotlib(combined, kind_data, kinds, fields, save_path, show)
    else:
        raise ValueError(
            f"Unsupported {backend=!r}. Must be 'matplotlib' or 'pyvista'."
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _read_single_row_csv(path: Path) -> dict[str, str]:
    """Read a single-row CSV file and return the row as a dict."""
    with open(path) as f:
        reader = csv.DictReader(f)
        return next(reader)


def compute_surface_force_coefficients(
    surface_mesh: Mesh[2, 3],
    a_ref: float,
) -> TensorDict:
    """Integrate predicted surface fields to obtain force coefficients.

    Computes drag, lift, and side-force coefficients by area-weighted
    integration of pressure and skin-friction contributions over the car
    body surface.

    The pressure force on the body is ``-C_p * n`` (outward normal convention)
    and the friction force is ``C_f`` (tangential).  Normal orientation is
    determined at integration time using the divergence theorem (matching
    the AirFRANS pattern), so cell winding in the input mesh does not
    need to be pre-corrected.

    Args:
        surface_mesh: Car surface Mesh with ``point_data["C_p"]`` and
            ``point_data["C_f"]``, plus cell connectivity for area
            computation.
        a_ref: Reference frontal area for normalization.

    Returns:
        TensorDict with scalar-tensor entries ``"Cd"``, ``"Cl"``, ``"Cs"``.
    """
    areas = surface_mesh.cell_areas  # (n_cells,)
    raw_normals = surface_mesh.cell_normals  # (n_cells, 3)

    ### Orient normals outward using the divergence theorem
    # For a closed surface, integral(x . n dA) = 3V > 0 when normals point
    # outward.  For an open surface like the car body, we use the mesh
    # centroid to make this robust to arbitrary mesh positioning.
    mesh_centroid = surface_mesh.points.mean(dim=0)
    outward_sign = torch.sign(
        ((surface_mesh.cell_centroids - mesh_centroid) * raw_normals).sum(dim=-1)
        @ areas
    )
    normals = outward_sign * raw_normals

    ### Interpolate vertex-centered data to cell centers
    cell_data = surface_mesh.point_data_to_cell_data().cell_data
    cp_cells: torch.Tensor = cell_data["C_p"]  # ty: ignore[invalid-assignment]  # (n_cells,)
    cf_cells: torch.Tensor = cell_data["C_f"]  # ty: ignore[invalid-assignment]  # (n_cells, 3)

    ### Force coefficient per cell: (-C_p * n + C_f) * A
    f_pressure = -cp_cells.unsqueeze(-1) * normals  # (n_cells, 3)
    f_friction = cf_cells  # (n_cells, 3)
    f_total = (f_pressure + f_friction) * areas.unsqueeze(-1)  # (n_cells, 3)

    ### Integrate and normalize
    f_integrated = f_total.sum(dim=0) / a_ref  # (3,)

    return TensorDict(
        {
            "Cd": f_integrated[0],
            "Cl": f_integrated[2],
            "Cs": f_integrated[1],
        }
    )


if __name__ == "__main__":
    import os

    if not (_data_env := os.environ.get("DRIVAER_DATA_DIR")):
        raise ValueError("DRIVAER_DATA_DIR environment variable is not set.")
    data_dir = Path(_data_env)

    sample_paths = DrivAerMLDataSet.get_split_paths(data_dir, "train")

    sample = DrivAerMLDataSet.preprocess(sample_paths[0])
    logger.info(f"Sample path: {sample_paths[0]}")
    logger.info(f"Prediction mesh points: {sample.prediction_mesh.points.shape}")
    logger.info(f"Prediction mesh cells:  {sample.prediction_mesh.cells.shape}")
    logger.info(f"Output keys: {list(sample.prediction_mesh.point_data.keys())}")
    logger.info(f"Reference lengths: {sample.reference_lengths.to_dict()}")
    logger.info(f"Aero coefficients: {sample.aero_coefficients.to_dict()}")
    for bc_name, bc_mesh in sample.boundary_meshes.items():
        logger.info(
            f"Boundary '{bc_name}': {bc_mesh.n_points} pts, {bc_mesh.n_cells} cells"
        )
