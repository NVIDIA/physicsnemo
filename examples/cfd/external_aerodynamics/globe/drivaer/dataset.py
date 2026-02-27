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

Reads DrivAerML simulation outputs (VTP boundary meshes, geometry CSVs, force
coefficient CSVs), extracts the car body surface, decimates it for GLOBE
boundary input, and assembles nondimensionalized prediction targets.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, Sequence

import numpy as np
import pyvista as pv
import torch
from jaxtyping import Float, Int
from tensordict import TensorDict, tensorclass
from torch.distributed import ReduceOp, all_reduce, is_initialized
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from physicsnemo.experimental.models.globe.utilities.cached_dataset import (
    CachedPreprocessingDataset,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import from_pyvista
from physicsnemo.utils.logging import PythonLogger

logger = PythonLogger("globe.drivaer.dataset")

### Reference conditions (constant across all DrivAerML runs)
# Confirmed by domino_nim_finetuning/src/openfoam_datapipe.py and cross-checked
# against the ratio pMeanTrim / CpMeanTrim at stagnation points.
U_INF = 38.89  # m/s  (140 km/h freestream velocity)
Q_INF = 0.5 * U_INF**2  # ~756 m²/s²  (kinematic dynamic pressure)
NU = 1.5e-5  # m²/s  (kinematic viscosity of air)

### Preprocessing parameters
DEFAULT_BOUNDARY_N_FACES = 20_000
DOMAIN_BOUNDARY_TOL = 0.05  # meters; cells within this distance of the domain
#   boundary are classified as tunnel walls
GROUND_NORMAL_Z_THRESHOLD = 0.85  # cells at z_min with n_z above this threshold
#   are classified as ground plane (not car underbody)

Split = Literal["train", "validation"]


@tensorclass
class DrivAerMLSample:
    """Single preprocessed DrivAerML sample for GLOBE training / inference.

    Attributes:
        surface_mesh: Full-resolution car body surface with ``point_data``
            containing nondimensional prediction targets (``C_p``, ``C_f``).
            Has cell connectivity for visualization.
        boundary_meshes: ``{"car_body": Mesh}`` - decimated car body surface
            used as GLOBE boundary input (geometry only, no field data).
        reference_lengths: Per-sample reference lengths (``L_ref``,
            ``sqrt_A_ref``) used for GLOBE multiscale kernel construction.
        dimensional_constants: ``U_inf``, ``q_inf`` for re-dimensionalization.
        aero_coefficients: Ground-truth ``Cd``, ``Cl``, ``Cs`` from the
            simulation, for evaluation of integrated force predictions.
    """

    surface_mesh: Mesh
    boundary_meshes: TensorDict[str, Mesh]
    reference_lengths: TensorDict[str, Float[torch.Tensor, ""]]
    dimensional_constants: TensorDict
    aero_coefficients: TensorDict

    @property
    def model_input_kwargs(self) -> dict:
        """Keyword arguments for :meth:`GLOBE.forward`."""
        return {
            "prediction_points": self.surface_mesh.points,
            "boundary_meshes": self.boundary_meshes,
            "reference_lengths": self.reference_lengths,
            "global_data": None,
        }

    if TYPE_CHECKING:

        def to(self, *args: Any, **kwargs: Any) -> Self: ...


# ---------------------------------------------------------------------------
# Car body extraction
# ---------------------------------------------------------------------------


def _identify_car_body_cells(
    pv_mesh: pv.PolyData,
    *,
    tol: float = DOMAIN_BOUNDARY_TOL,
    ground_nz: float = GROUND_NORMAL_Z_THRESHOLD,
) -> np.ndarray:
    """Return a boolean mask identifying car body cells in a DrivAerML VTP.

    The VTP boundary mesh contains all domain boundaries merged into a single
    mesh with no patch identifiers. This function classifies each cell as
    either *tunnel wall* or *car body* using coordinate and surface-normal
    heuristics.

    Tunnel walls are identified as cells whose centers lie within *tol* of
    the domain bounding box on any face (inlet, outlet, sides, ceiling), or
    cells at z_min whose outward normal points predominantly upward (ground
    plane, not car underbody).

    Args:
        pv_mesh: Full boundary PolyData from the VTP file.
        tol: Distance threshold (meters) for identifying domain-boundary cells.
        ground_nz: Minimum z-component of the outward normal for a cell at
            z_min to be classified as ground rather than car underbody.

    Returns:
        Boolean array of shape ``(n_cells,)``. True = car body.
    """
    cell_centers: np.ndarray = pv_mesh.cell_centers().points  # (n_cells, 3)
    normals: np.ndarray = pv_mesh.cell_normals  # (n_cells, 3)
    x_min, x_max, y_min, y_max, z_min, z_max = pv_mesh.bounds

    is_inlet = cell_centers[:, 0] < x_min + tol
    is_outlet = cell_centers[:, 0] > x_max - tol
    is_left = cell_centers[:, 1] < y_min + tol
    is_right = cell_centers[:, 1] > y_max - tol
    is_ceiling = cell_centers[:, 2] > z_max - tol
    is_near_floor = cell_centers[:, 2] < z_min + tol
    is_ground = is_near_floor & (normals[:, 2] > ground_nz)

    is_tunnel = is_inlet | is_outlet | is_left | is_right | is_ceiling | is_ground
    return ~is_tunnel


def _extract_car_body(pv_mesh: pv.PolyData, **kwargs: Any) -> pv.PolyData:
    """Extract the car body sub-mesh from a full DrivAerML boundary VTP.

    The full VTP boundary mesh forms a closed surface (tunnel + car body),
    so VTK's normal-consistency algorithm can reliably orient all face
    normals outward.  This orientation step is performed on the FULL mesh
    first, then car body cells are extracted with their corrected winding
    intact.

    Args:
        pv_mesh: Full boundary PolyData from the VTP file.
        **kwargs: Forwarded to :func:`_identify_car_body_cells`.

    Returns:
        Triangulated PolyData of the car body with outward-facing normals
        and cell data preserved.
    """
    if not pv_mesh.is_all_triangles:
        pv_mesh = pv_mesh.triangulate()

    # Orient normals on the FULL (closed) boundary mesh. The full tunnel
    # envelope is closed, so auto_orient reliably determines outward.
    pv_mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=True,
    )

    car_mask = _identify_car_body_cells(pv_mesh, **kwargs)
    all_faces: np.ndarray = pv_mesh.regular_faces  # (n_cells, 3)
    car_faces = all_faces[car_mask]

    # Compact sub-mesh (renumber vertices, remove orphaned points)
    unique_pts, inverse = np.unique(car_faces.ravel(), return_inverse=True)
    new_faces = inverse.reshape(-1, 3)
    new_points = pv_mesh.points[unique_pts]

    faces_padded = np.column_stack(
        [np.full(len(new_faces), 3, dtype=np.int64), new_faces]
    ).ravel()
    car_body = pv.PolyData(new_points, faces_padded)

    for name in pv_mesh.cell_data:
        if name == "Normals":
            continue  # Skip VTK-generated normals array
        car_body.cell_data[name] = pv_mesh.cell_data[name][car_mask]

    return car_body


def _ensure_outward_normals(mesh: Mesh) -> None:
    """Flip cell winding if normals point inward (toward the mesh centroid).

    Uses a majority-vote heuristic: for each cell, checks whether the normal
    points away from the overall mesh centroid.  If more than half point
    inward, reverses the winding of ALL cells (swapping vertex indices 1
    and 2).  Modifies *mesh* in place.

    Args:
        mesh: A 2-manifold (surface) Mesh.
    """
    centroid = mesh.points.mean(dim=0)
    outward = mesh.cell_centroids - centroid  # (n_cells, 3)
    dot = (outward * mesh.cell_normals).sum(dim=-1)  # (n_cells,)
    if dot.mean() < 0:
        mesh.cells[:, 1], mesh.cells[:, 2] = (
            mesh.cells[:, 2].clone(),
            mesh.cells[:, 1].clone(),
        )
        mesh._cache.clear()
        logger.info("  flipped cell winding to ensure outward normals")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DrivAerMLDataSet(CachedPreprocessingDataset):
    """Disk-cached preprocessing dataset for DrivAerML + GLOBE."""

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
        num_workers: int = 4,
    ) -> DataLoader:
        """Create a distributed DataLoader yielding one sample per iteration.

        Args:
            sample_paths: Paths to individual ``run_N/`` directories.
            cache_dir: Directory for disk-cached preprocessed ``.pt`` files.
            world_size: Total distributed ranks.
            rank: This process's rank.
            num_workers: DataLoader worker processes.

        Returns:
            DataLoader with :class:`DistributedSampler`.
        """
        dataset = cls(sample_paths=sample_paths, cache_dir=cache_dir)
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
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess(
        sample_path: Path,
        boundary_n_faces: int = DEFAULT_BOUNDARY_N_FACES,
    ) -> DrivAerMLSample:
        """Preprocess a single DrivAerML run into a GLOBE-ready sample.

        Pipeline:
            1. Load the VTP boundary mesh (~600 MB) and extract the car body.
            2. Compute nondimensional surface fields (C_p, C_f).
            3. Interpolate cell-centered data to mesh vertices.
            4. Decimate the car body to create a compact GLOBE boundary mesh.
            5. Parse geometry reference CSV for per-sample reference lengths.
            6. Parse force/moment CSV for ground-truth aero coefficients.

        Args:
            sample_path: Path to a ``run_N/`` directory.
            boundary_n_faces: Target face count for the decimated GLOBE
                boundary mesh.  The actual count may differ slightly due to
                decimation algorithm constraints.

        Returns:
            Fully preprocessed :class:`DrivAerMLSample`.
        """
        sample_dir = Path(sample_path)
        run_idx = sample_dir.name.removeprefix("run_")

        ### Load VTP boundary mesh
        vtp_path = sample_dir / f"boundary_{run_idx}.vtp"
        if not vtp_path.exists():
            raise FileNotFoundError(f"Missing VTP boundary file: {vtp_path}")
        pv_boundary: pv.PolyData = pv.read(vtp_path)

        ### Extract car body sub-mesh
        car_body = _extract_car_body(pv_boundary)
        logger.info(
            f"run_{run_idx}: extracted {car_body.n_cells} car body cells "
            f"from {pv_boundary.n_cells} total boundary cells"
        )
        del pv_boundary  # free ~600 MB

        ### Compute nondimensional surface fields (cell-centered)
        car_body.cell_data["C_p"] = car_body.cell_data["CpMeanTrim"].copy()
        car_body.cell_data["C_f"] = (
            car_body.cell_data["wallShearStressMeanTrim"] / Q_INF
        )

        # Remove raw fields to avoid carrying unnecessary data
        for name in ("CpMeanTrim", "pMeanTrim", "pPrime2MeanTrim",
                     "wallShearStressMeanTrim", "Normals"):
            if name in car_body.cell_data:
                del car_body.cell_data[name]

        ### Interpolate cell data to vertices for GLOBE prediction targets
        car_body_pt = car_body.cell_data_to_point_data()

        # Keep only the nondimensional fields in point_data
        point_data = TensorDict(
            {
                "C_p": torch.from_numpy(
                    np.ascontiguousarray(car_body_pt.point_data["C_p"])
                ).float(),
                "C_f": torch.from_numpy(
                    np.ascontiguousarray(car_body_pt.point_data["C_f"])
                ).float(),
            },
            batch_size=[car_body_pt.n_points],
        )

        # Build the full-resolution surface Mesh (with cell connectivity)
        surface_mesh = Mesh(
            points=torch.from_numpy(
                np.ascontiguousarray(car_body_pt.points)
            ).float(),
            cells=torch.from_numpy(
                np.ascontiguousarray(car_body_pt.regular_faces)
            ).long(),
            point_data=point_data,
        )

        ### Decimate car body for GLOBE boundary input (geometry only)
        # Multi-stage decimation: large meshes (~7M faces) are reduced in
        # stages because a single extreme-ratio decimation can be very slow.
        # Each stage reduces by at most ~95%, which keeps per-stage runtime
        # manageable.  vtkQuadricDecimation (decimate) is used rather than
        # vtkDecimatePro for better performance on large meshes.
        car_body_coarse = car_body.copy()
        n_original = car_body_coarse.n_cells
        while car_body_coarse.n_cells > boundary_n_faces * 1.1:
            current = car_body_coarse.n_cells
            target = max(boundary_n_faces, int(current * 0.05))
            reduction = 1.0 - target / current
            car_body_coarse = car_body_coarse.decimate(reduction)
            logger.info(
                f"  decimation stage: {current} -> {car_body_coarse.n_cells} faces"
            )

        if not car_body_coarse.is_all_triangles:
            car_body_coarse = car_body_coarse.triangulate()

        boundary_mesh = Mesh(
            points=torch.from_numpy(
                np.ascontiguousarray(car_body_coarse.points)
            ).float(),
            cells=torch.from_numpy(
                np.ascontiguousarray(car_body_coarse.regular_faces)
            ).long(),
        )
        logger.info(
            f"run_{run_idx}: boundary mesh decimated from {n_original} "
            f"to {boundary_mesh.n_cells} faces"
        )

        ### Verify outward normal orientation (centroid heuristic)
        # For car-like meshes the majority of surface normals should point
        # away from the geometric centroid. If the mean dot product
        # (cell_center - centroid) . normal is negative, flip all cells.
        for mesh_to_check in (surface_mesh, boundary_mesh):
            _ensure_outward_normals(mesh_to_check)

        ### Parse geometry reference CSV
        geo_ref_path = sample_dir / f"geo_ref_{run_idx}.csv"
        geo_ref = _read_single_row_csv(geo_ref_path)
        l_ref = float(geo_ref["lRef"])
        a_ref = float(geo_ref["aRef"])

        ### Parse force/moment CSV
        force_mom_path = sample_dir / f"force_mom_{run_idx}.csv"
        force_mom = _read_single_row_csv(force_mom_path)

        return DrivAerMLSample(
            surface_mesh=surface_mesh,
            boundary_meshes=TensorDict({"car_body": boundary_mesh}),
            reference_lengths=TensorDict(
                {
                    "L_ref": torch.as_tensor(l_ref),
                    "sqrt_A_ref": torch.as_tensor(a_ref**0.5),
                },
            ),
            dimensional_constants=TensorDict(
                {
                    "U_inf": torch.as_tensor(U_INF),
                    "q_inf": torch.as_tensor(Q_INF),
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

    # ------------------------------------------------------------------
    # Postprocessing / visualization
    # ------------------------------------------------------------------

    @staticmethod
    def postprocess(
        pred_mesh: Mesh,
        sample: DrivAerMLSample,
        *,
        fields: Sequence[str] | None = None,
    ) -> Mesh:
        """Build a combined pred/true/error Mesh with integrated force coefficients.

        Assembles a single Mesh whose ``point_data`` contains nested
        ``"true"``, ``"pred"``, and ``"error"`` TensorDicts for the selected
        fields, and whose ``global_data`` contains integrated surface force
        coefficients for the prediction and CSV ground-truth coefficients.

        This method performs no visualization.  Pass the returned Mesh to
        :meth:`visualize_comparison` to render a subplot grid.

        Args:
            pred_mesh: Point-cloud Mesh with predicted field values in
                ``point_data``.
            sample: The preprocessed sample.  ``sample.surface_mesh`` provides
                the ground-truth fields and cell connectivity;
                ``sample.reference_lengths["sqrt_A_ref"]`` is used for
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
        true_mesh = sample.surface_mesh

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
        a_ref = float(sample.reference_lengths["sqrt_A_ref"]) ** 2
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

    @staticmethod
    def visualize_comparison(
        combined: Mesh,
        *,
        save_path: Path | None = None,
        show: bool = False,
    ) -> None:
        """Render a 3D comparison of predicted vs. true surface fields.

        Takes the combined Mesh returned by :meth:`postprocess` and draws
        truth / prediction / error rows for each field using PyVista.

        Args:
            combined: Mesh returned by :meth:`postprocess`, with
                ``point_data["true"]``, ``point_data["pred"]``, and
                ``point_data["error"]``.
            save_path: File path for the rendered screenshot.  Defaults to
                ``drivaer_comparison.png`` in the current directory.
            show: Whether to display interactively (requires a display).
        """
        from physicsnemo.mesh.io import to_pyvista

        if save_path is None:
            save_path = Path("drivaer_comparison.png")

        ### Flatten nested keys to dot-separated strings for display
        true_flat = combined.point_data["true"].flatten_keys(".")  # ty: ignore[unresolved-attribute]
        pred_flat = combined.point_data["pred"].flatten_keys(".")  # ty: ignore[unresolved-attribute]
        error_flat = combined.point_data["error"].flatten_keys(".")  # ty: ignore[unresolved-attribute]

        fields = sorted(true_flat.keys())
        kind_data = {"true": true_flat, "pred": pred_flat, "error": error_flat}
        kinds = {"true": "Truth", "pred": "Prediction", "error": "Error"}

        n_cols = len(fields)
        n_rows = len(kinds)

        plotter = pv.Plotter(
            shape=(n_rows, n_cols),
            off_screen=not show,
            window_size=(600 * n_cols, 500 * n_rows),
        )

        combined_pv = to_pyvista(combined.to("cpu"))

        for col, field_name in enumerate(fields):
            true_vals = true_flat[field_name].cpu().numpy()
            pred_vals = pred_flat[field_name].cpu().numpy()

            is_vector = true_vals.ndim > 1 and true_vals.shape[-1] > 1
            if is_vector:
                true_scalars = np.linalg.norm(true_vals, axis=-1)
                pred_scalars = np.linalg.norm(pred_vals, axis=-1)
                label = f"|{field_name}|"
            else:
                true_scalars = true_vals.ravel()
                pred_scalars = pred_vals.ravel()
                label = field_name

            ### Shared color limits across truth and prediction
            finite_all = np.concatenate([
                true_scalars[np.isfinite(true_scalars)],
                pred_scalars[np.isfinite(pred_scalars)],
            ])
            shared_clim = [float(finite_all.min()), float(finite_all.max())]

            for row, (key, title) in enumerate(kinds.items()):
                plotter.subplot(row, col)
                vals: torch.Tensor = kind_data[key][field_name]  # ty: ignore[invalid-assignment]

                if is_vector:
                    scalars = np.linalg.norm(vals.cpu().numpy(), axis=-1)
                else:
                    scalars = vals.cpu().numpy().ravel()

                if key == "error":
                    emax = float(
                        np.abs(scalars[np.isfinite(scalars)]).max()
                    )
                    if is_vector:
                        cmap, clim = "Reds", [0.0, emax]
                    else:
                        cmap, clim = "RdBu_r", [-emax, emax]
                else:
                    cmap, clim = "turbo", shared_clim

                plotter.add_mesh(
                    combined_pv.copy(),
                    scalars=scalars,
                    cmap=cmap,
                    clim=clim,
                    show_edges=False,
                    scalar_bar_args={"title": label if row == 0 else ""},
                )
                plotter.add_text(
                    f"{title}\n{label}" if row == 0 else title,
                    font_size=10,
                )
                plotter.camera_position = "xy"

        plotter.screenshot(str(save_path), scale=2)
        logger.info(f"Saved comparison to {save_path}")
        if show:
            plotter.show()
        plotter.close()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _read_single_row_csv(path: Path) -> dict[str, str]:
    """Read a single-row CSV file and return the row as a dict."""
    with open(path) as f:
        reader = csv.DictReader(f)
        return next(reader)


def compute_max_mesh_sizes(
    dataloader: DataLoader,
    device: torch.device,
    *,
    face_downsampling_ratio: float = 1.0,
    rank: int = 0,
) -> TensorDict[str, TensorDict[Literal["n_points", "n_cells"], Int[torch.Tensor, ""]]]:
    """Compute the maximum n_points and n_cells per boundary-condition type.

    Scans all samples in *dataloader*, tracking the largest boundary mesh
    dimensions for each BC type.  Uses distributed all-reduce to find the
    global maximum across all ranks.  Results are used to pad meshes to
    uniform sizes for ``torch.compile`` with static shapes.

    Args:
        dataloader: DataLoader yielding :class:`DrivAerMLSample` objects.
        device: Device for the all-reduce tensors.
        face_downsampling_ratio: Scale factor applied to cell counts (< 1.0
            for training, 1.0 for validation).
        rank: Distributed rank (progress bar shown only on rank 0).

    Returns:
        ``TensorDict`` mapping BC types to ``{"n_points": ..., "n_cells": ...}``.
    """
    raw_maxes: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_points": 0, "n_cells": 0}
    )

    for sample in tqdm(
        dataloader,
        desc=f"Computing max mesh sizes (rank {rank})",
        disable=rank != 0,
    ):
        for bc_type, mesh in sample.boundary_meshes.items():
            raw_maxes[bc_type]["n_points"] = max(
                raw_maxes[bc_type]["n_points"], mesh.n_points
            )
            n_cells = (
                int(mesh.n_cells * face_downsampling_ratio)
                if face_downsampling_ratio != 1.0
                else mesh.n_cells
            )
            raw_maxes[bc_type]["n_cells"] = max(
                raw_maxes[bc_type]["n_cells"], n_cells
            )

    result = TensorDict(
        {
            bc_type: TensorDict(
                {
                    "n_points": torch.tensor(sizes["n_points"], device=device),
                    "n_cells": torch.tensor(sizes["n_cells"], device=device),
                }
            )
            for bc_type, sizes in raw_maxes.items()
        },
    )

    if is_initialized():
        for bc_type in result.keys(include_nested=False):
            all_reduce(result[bc_type, "n_points"], op=ReduceOp.MAX)
            all_reduce(result[bc_type, "n_cells"], op=ReduceOp.MAX)

    if rank == 0:
        logger.info(f"Max mesh sizes: {result.to_dict()}")

    return result


def compute_surface_force_coefficients(
    surface_mesh: Mesh,
    a_ref: float,
) -> TensorDict:
    """Integrate predicted surface fields to obtain force coefficients.

    Computes drag, lift, and side-force coefficients by area-weighted
    integration of pressure and skin-friction contributions over the car
    body surface.

    The pressure force on the body is ``-C_p * n`` (outward normal convention)
    and the friction force is ``C_f`` (tangential).

    Note:
        The DrivAerML VTP boundary meshes contain non-manifold topology
        (merged patches), which prevents perfect outward-normal orientation
        for all cells.  Integrated force coefficients from this function are
        therefore approximate.  Ground-truth coefficients from the CSV files
        should be used for quantitative evaluation.

    Args:
        surface_mesh: Car surface Mesh with ``point_data["C_p"]`` and
            ``point_data["C_f"]``, plus cell connectivity for area
            computation.
        a_ref: Reference frontal area for normalization.

    Returns:
        TensorDict with scalar-tensor entries ``"Cd"``, ``"Cl"``, ``"Cs"``.
    """
    areas = surface_mesh.cell_areas  # (n_cells,)
    normals = surface_mesh.cell_normals  # (n_cells, 3)

    ### Interpolate vertex-centered data to cell centers (average of vertices)
    cells = surface_mesh.cells  # (n_cells, 3)
    cp_pts: torch.Tensor = surface_mesh.point_data["C_p"]  # (n_points,)
    cf_pts: torch.Tensor = surface_mesh.point_data["C_f"]  # (n_points, 3)

    cp_cells = cp_pts[cells].mean(dim=1)  # (n_cells,)
    cf_cells = cf_pts[cells].mean(dim=1)  # (n_cells, 3)

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
    logger.info(f"Surface mesh points: {sample.surface_mesh.points.shape}")
    logger.info(f"Surface mesh cells:  {sample.surface_mesh.cells.shape}")
    logger.info(
        f"Output keys: {list(sample.surface_mesh.point_data.keys())}"
    )
    logger.info(
        f"Boundary mesh: {sample.boundary_meshes['car_body'].n_cells} cells"
    )
    logger.info(f"Reference lengths: {sample.reference_lengths.to_dict()}")
    logger.info(f"Aero coefficients: {sample.aero_coefficients.to_dict()}")
