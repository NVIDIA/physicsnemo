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

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Literal, Sequence

import pyvista as pv
import torch
from jaxtyping import Bool, Float, Int
from tensordict import TensorClass, TensorDict
from torch.distributed import ReduceOp, all_reduce, is_initialized
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from physicsnemo.experimental.utils import CachedPreprocessingDataset
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus import compute_point_derivatives
from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.projections import project
from physicsnemo.nn.functional.neighbors import knn
from physicsnemo.utils.logging import PythonLogger

logger = PythonLogger("globe.airfrans.dataset")

RHO = 1  # kg/m^3
# NOTE: this RHO is correct; in some places, the AirFRANS authors incorrectly
# report their density as 1.204, but if you actually dig into the OpenFOAM case
# files, you can see that the density is actually 1. You can also confirm this
# from the data itself - observe that RHO=1 yields constant far-field total
# pressure (which is physically correct), but RHO=1.204 does not (which is
# physically incorrect).
NU = 1.56e-5  # m^2/s


class AirFRANSSample(TensorClass):
    prediction_mesh: Mesh  # Point cloud with nondimensional point_data and global_data
    boundary_meshes: TensorDict[str, Mesh]  # BC name -> Mesh
    reference_lengths: TensorDict[
        str, Float[torch.Tensor, ""]
    ]  # reference length names to scalar tensors
    dimensional_constants: (
        TensorDict  # U_inf, q_inf - only for postprocessing / redimensionalization
    )

    @property
    def model_input_kwargs(self) -> dict:
        """Kwargs for :meth:`GLOBE.forward`."""
        return {
            "prediction_points": self.prediction_mesh.points,
            "boundary_meshes": self.boundary_meshes,
            "reference_lengths": self.reference_lengths,
            "global_data": self.prediction_mesh.global_data,
        }


class AirFRANSDataSet(CachedPreprocessingDataset):
    @classmethod
    def get_split_paths(
        cls,
        data_dir: Path,
        task: Literal["full", "scarce", "reynolds", "aoa"],
        split: Literal["train", "test"],
    ) -> list[Path]:
        """Read ``manifest.json`` and return sample paths for a task/split.

        For the ``"scarce"`` task, the test split uses the ``"full"`` test set
        (``"scarce"`` only defines a reduced training set).

        Args:
            data_dir: Root directory containing ``manifest.json`` and sample
                subdirectories.
            task: AirFRANS task name (``"full"``, ``"scarce"``, ``"reynolds"``,
                ``"aoa"``).
            split: ``"train"`` or ``"test"``.

        Returns:
            List of absolute paths to individual sample directories.
        """
        manifest = json.loads((data_dir / "manifest.json").read_text())
        effective_task = "full" if (task == "scarce" and split == "test") else task
        return [data_dir / f for f in manifest[f"{effective_task}_{split}"]]

    @classmethod
    def make_dataloader(
        cls,
        sample_paths: Sequence[Path],
        cache_dir: Path,
        *,
        world_size: int = 1,
        rank: int = 0,
        num_workers: int | None = None,
    ) -> DataLoader:
        """Create a distributed DataLoader for this dataset.

        Each item is a single sample (``batch_size=None``) with identity
        collation, suitable for variable-size mesh data that cannot be
        stacked into uniform batches.

        Args:
            sample_paths: Paths to individual sample directories.
            cache_dir: Directory for disk caching of preprocessed samples.
            world_size: Total number of distributed ranks (across all nodes).
            rank: This process's distributed rank.
            num_workers: DataLoader worker processes per rank.  When ``None``
                (the default), auto-computed from this rank's CPU affinity
                and the per-node rank count detected from launcher env vars
                (``LOCAL_WORLD_SIZE`` / ``OMPI_COMM_WORLD_LOCAL_SIZE`` /
                ``SLURM_NTASKS_PER_NODE``), reserving one core per GPU the
                rank drives for kernel-launch orchestration and OS work.

        Returns:
            Configured DataLoader with distributed sampling.
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
            prefetch_factor=32 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )

    @staticmethod
    def preprocess(
        sample_path: Path,
        patch_out_nonphysical_values: bool = True,
        grad_c_p_clip_threshold: float = 20.0,
        c_pt_nonphysical_threshold: float = 1.02,
    ) -> AirFRANSSample:
        """Load an AirFRANS sample and compute nondimensional output fields.

        Reads the internal volume mesh, freestream boundary, and airfoil
        boundary from VTU/VTP files, then derives nondimensional fields
        (pressure coefficient, velocity ratio, turbulent viscosity, surface
        tractions, etc.) suitable for training and evaluation.

        Points with non-physical total-pressure values (numerical artifacts
        near stagnation points) are optionally patched to NaN so the loss
        function ignores them.

        Args:
            sample_path: Path to a sample directory containing
                ``{name}_internal.vtu``, ``{name}_aerofoil.vtp``, and
                ``{name}_freestream.vtp``.
            patch_out_nonphysical_values: Replace all output fields with NaN
                at points where the total-pressure coefficient exceeds
                ``c_pt_nonphysical_threshold``.
            grad_c_p_clip_threshold: Maximum allowed magnitude for the
                nondimensional pressure gradient ``|grad(C_p) * chord|``.
                Points exceeding this are set to NaN.
            c_pt_nonphysical_threshold: Total-pressure coefficient threshold
                above which points are considered non-physical (Bernoulli
                violation).  Default 1.02 allows small numerical overshoot.

        Returns:
            Preprocessed sample with nondimensional ``point_data`` on the
            interior mesh, the airfoil boundary mesh, reference lengths,
            and dimensional constants for redimensionalization.
        """
        ### Load meshes and convert to 2D Mesh objects
        sample_dir = Path(sample_path)
        base = sample_dir.name
        mesh_paths = {
            "freestream": sample_dir / f"{base}_freestream.vtp",
            "airfoil": sample_dir / f"{base}_aerofoil.vtp",
            "internal": sample_dir / f"{base}_internal.vtu",
        }
        for path in mesh_paths.values():
            if not path.exists():
                raise FileNotFoundError(f"Missing required file: {path}")

        freestream = project(
            from_pyvista(pv.read(mesh_paths["freestream"]), manifold_dim=1),
            keep_dims=[0, 1],
            transform_cell_data=True,
        )
        airfoil = project(
            from_pyvista(pv.read(mesh_paths["airfoil"]), manifold_dim=1),
            keep_dims=[0, 1],
        )
        internal = project(
            from_pyvista(pv.read(mesh_paths["internal"])),
            keep_dims=[0, 1],
            transform_point_data=True,
        )

        ### Reference quantities from freestream boundary
        U_inf: Float[torch.Tensor, "2"] = freestream.cell_data["U"].mean(dim=0)  # ty: ignore[invalid-assignment]
        U_inf_magnitude: Float[torch.Tensor, ""] = torch.norm(U_inf)
        q_inf: Float[torch.Tensor, ""] = 0.5 * RHO * U_inf_magnitude**2
        chord = 1.0

        ### Nondimensional volume fields (from raw simulation data on internal mesh)
        U: Float[torch.Tensor, "n_points 2"] = internal.point_data["U"]  # ty: ignore[invalid-assignment]
        p: Float[torch.Tensor, " n_points"] = internal.point_data["p"]  # ty: ignore[invalid-assignment]
        nut: Float[torch.Tensor, " n_points"] = internal.point_data["nut"]  # ty: ignore[invalid-assignment]

        U_over_U_inf: Float[torch.Tensor, "n_points 2"] = U / U_inf_magnitude
        C_p: Float[torch.Tensor, " n_points"] = p / q_inf
        C_pt: Float[torch.Tensor, " n_points"] = C_p + U_over_U_inf.square().sum(dim=-1)

        ### Gradient fields via Mesh calculus
        mesh_with_grads = compute_point_derivatives(mesh=internal, keys=["p", "U"])
        grad_C_p: Float[torch.Tensor, "n_points 2"] = mesh_with_grads.point_data[
            "p_gradient"
        ] * (chord / q_inf)
        # Clip nondimensional pressure-gradient values whose magnitude exceeds
        # the threshold.  Spurious spikes arise from the least-squares gradient
        # reconstruction near poorly-resolved regions (e.g. sharp trailing
        # edges or thin boundary layers).  These are replaced with NaN so that
        # they are masked out in the loss function.
        grad_C_p[grad_C_p.norm(dim=-1) > grad_c_p_clip_threshold] = torch.nan

        velocity_jacobian: Float[torch.Tensor, "n_points 2 2"] = (  # ty: ignore[invalid-assignment]
            mesh_with_grads.point_data["U_gradient"]
        )

        ### Surface force fields
        point_is_on_airfoil: Bool[torch.Tensor, " n_points"] = (  # ty: ignore[invalid-assignment]
            internal.point_data["implicit_distance"] == 0
        )

        # For each internal point, find the nearest airfoil surface point.
        # Uses O(n log m) KDTree lookup (auto-dispatched via physicsnemo)
        # instead of the O(n * m) brute-force distance matrix.

        nearest_airfoil_idx, _ = knn(
            points=airfoil.points, queries=internal.points, k=1
        )
        nearest_airfoil_idx: Int[torch.Tensor, " n_points"] = nearest_airfoil_idx[:, 0]

        # Orient normals outward (into the fluid) for the Cauchy traction formula.
        # Divergence theorem: integral(x . n) dl = 2 * area > 0 for outward normals.
        outward_sign = torch.sign(
            (airfoil.cell_centroids * airfoil.cell_normals).sum(dim=-1)
            @ airfoil.cell_areas
        )
        airfoil_normals: Float[torch.Tensor, "n_points 2"] = (
            outward_sign * airfoil.point_normals[nearest_airfoil_idx]
        )
        airfoil_normals[~point_is_on_airfoil] = torch.nan

        strain_rate: Float[torch.Tensor, "n_points 2 2"] = 0.5 * (
            velocity_jacobian + velocity_jacobian.transpose(1, 2)
        )
        wall_shear_stress: Float[torch.Tensor, "n_points 2 2"] = 2 * NU * strain_rate
        wall_shear_force: Float[torch.Tensor, "n_points 2"] = torch.einsum(
            "pij,pj->pi",
            wall_shear_stress,
            airfoil_normals,
        )
        pressure_force: Float[torch.Tensor, "n_points 2"] = (
            -p[:, None] * airfoil_normals
        )

        ### Assemble output fields
        output_fields = TensorDict(
            {
                "U/|U_inf|": U_over_U_inf,
                "ΔU/|U_inf|": (U - U_inf[None, :]) / U_inf_magnitude,
                "C_p": C_p,
                "C_pt": C_pt,
                "ln(1+nut/nu)": torch.log1p(nut / NU),
                "∇C_p*chord": grad_C_p,
                "C_F,shear": wall_shear_force / q_inf,
                "C_F,pressure": pressure_force / q_inf,
                "C_F": (wall_shear_force + pressure_force) / q_inf,
            },
            batch_size=[internal.n_points],
        )

        if patch_out_nonphysical_values:
            # In incompressible flow, total pressure is conserved along
            # streamlines (Bernoulli), so C_pt should not exceed 1.0.  Values
            # slightly above 1.0 arise from numerical artifacts in the CFD
            # solution (e.g. cell averaging near stagnation points).  Points
            # exceeding the threshold are replaced with NaN across ALL output
            # fields so that the loss function ignores them.
            non_physical_C_pt: Bool[torch.Tensor, " n_points"] = (
                C_pt > c_pt_nonphysical_threshold
            )
            if non_physical_C_pt.sum() / len(C_pt) > 0.0001:
                logger.warning(
                    f"In {sample_path.name}, {non_physical_C_pt.sum() / len(C_pt):.2%} of points had non-physical total pressures and were patched out."
                )
            output_fields[non_physical_C_pt] = torch.nan

        return AirFRANSSample(
            prediction_mesh=Mesh(
                points=internal.points,
                cells=internal.cells,
                point_data=output_fields,
                global_data=TensorDict(
                    {
                        "U_inf / U_inf_magnitude": U_inf / U_inf_magnitude,
                    }
                ),
            ),
            boundary_meshes=TensorDict(
                {"no_slip": Mesh(points=airfoil.points, cells=airfoil.cells)},  # ty: ignore[invalid-argument-type]
            ),
            reference_lengths=TensorDict(
                {
                    "chord": torch.as_tensor(chord),
                    "delta_FS": torch.as_tensor((NU / U_inf_magnitude * chord) ** 0.5),
                },
            ),
            dimensional_constants=TensorDict(
                {
                    "U_inf": U_inf,
                    "q_inf": q_inf,
                }
            ),
        )

    @staticmethod
    def postprocess(
        pred_mesh: Mesh[0, 2],
        sample: AirFRANSSample,
        *,
        fields: Sequence[str | tuple[str, ...]] | None = None,
    ) -> Mesh[2, 2]:
        """Build a combined pred/true/error Mesh with integrated force coefficients.

        Assembles a single Mesh whose ``point_data`` contains nested
        ``"true"``, ``"pred"``, and ``"error"`` TensorDicts for the selected
        fields, and whose ``global_data`` contains integrated surface force
        coefficients for both the prediction and the ground truth.

        This method performs no visualization.  Pass the returned Mesh to
        :meth:`visualize_comparison` to render a subplot grid.

        Args:
            pred_mesh: Point-cloud Mesh with predicted field values in
                ``point_data``.
            sample: The preprocessed sample.  ``sample.prediction_mesh`` provides
                the ground-truth fields and cell connectivity;
                ``sample.boundary_meshes["no_slip"]`` and
                ``sample.reference_lengths["chord"]`` are used for surface
                force integration.
            fields: Which field names to compare.  If ``None``, uses the sorted
                intersection of pred and true ``point_data`` keys.

        Returns:
            Combined Mesh with ``point_data["true"]``, ``point_data["pred"]``,
            ``point_data["error"]`` for the selected fields, and
            ``global_data["pred"]`` / ``global_data["true"]`` each containing
            scalar force coefficient tensors (Cd, Cl, and pressure/friction
            decompositions).

        Raises:
            ValueError: If pred_mesh and the sample interior mesh have
                different numbers of points.
        """
        true_mesh = sample.prediction_mesh

        if pred_mesh.n_points != true_mesh.n_points:
            raise ValueError(
                f"Point count mismatch: {pred_mesh.n_points=} != {true_mesh.n_points=}"
            )

        ### Determine fields to compare
        if fields is None:
            fields: list[str | tuple[str, ...]] = sorted(
                set(pred_mesh.point_data.keys(include_nested=True, leaves_only=True))
                & set(true_mesh.point_data.keys(include_nested=True, leaves_only=True))
            )

        ### Build combined point_data
        pred_selected = pred_mesh.point_data.select(*fields)
        true_selected = true_mesh.point_data.select(*fields)
        error_data: TensorDict = pred_selected.apply(  # ty: ignore[invalid-assignment]
            lambda p, t: p - t, true_selected
        )

        ### Compute integrated surface force coefficients
        airfoil_mesh: Mesh[1, 2] = sample.boundary_meshes["no_slip"]
        chord = float(sample.reference_lengths["chord"])

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
                        volume_mesh=pred_mesh, airfoil_mesh=airfoil_mesh, chord=chord
                    ),
                    "true": compute_surface_force_coefficients(
                        volume_mesh=true_mesh, airfoil_mesh=airfoil_mesh, chord=chord
                    ),
                }
            ),
        )

    @staticmethod
    def visualize_comparison(
        combined: Mesh[2, 2],
        *,
        show: bool = True,
        show_error: bool = True,
        xlim: tuple[float, float] = (-0.6, 1.6),
        ylim: tuple[float, float] = (-0.8, 0.8),
    ) -> None:
        """Render a subplot grid comparing predicted vs. true fields.

        Takes the combined Mesh returned by :meth:`postprocess` and draws
        truth / prediction / error rows for each field, using filled contour
        plots for volume fields and arrow annotations for surface-only
        fields.

        Args:
            combined: Mesh returned by :meth:`postprocess`, with
                ``point_data["true"]``, ``point_data["pred"]``, and
                ``point_data["error"]``.
            show: Whether to display the plot via ``plt.show()``.
            show_error: Whether to include an error row in the subplot grid.
            xlim: Horizontal axis limits for each subplot.
            ylim: Vertical axis limits for each subplot.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.tri as mpl_tri
        import numpy as np

        mpl.rcParams["contour.negative_linestyle"] = "solid"

        def _to_scalars(t: torch.Tensor, is_vec: bool) -> np.ndarray:
            """Reduce a possibly-vector tensor to a 1D float64 numpy array."""
            s = t.norm(dim=-1) if is_vec else t.reshape(-1)
            return s.float().cpu().numpy()

        ### Flatten nested keys to dot-separated strings for display
        true_flat = combined.point_data["true"].flatten_keys(".")  # ty: ignore[unresolved-attribute]
        pred_flat = combined.point_data["pred"].flatten_keys(".")  # ty: ignore[unresolved-attribute]
        error_flat = combined.point_data["error"].flatten_keys(".")  # ty: ignore[unresolved-attribute]

        fields = sorted(true_flat.keys())

        kind_data = {"true": true_flat, "pred": pred_flat, "error": error_flat}
        kinds: dict[str, str] = {"true": "Truth", "pred": "Prediction"}
        if show_error:
            kinds["error"] = "Error"
        n_rows, n_cols = len(kinds), len(fields)

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(4 * n_cols, 3.4 * n_rows),
            squeeze=False,
        )

        ### Build matplotlib Triangulation from the mesh topology
        pts = combined.points.cpu().numpy()
        tri = mpl_tri.Triangulation(
            pts[:, 0],
            pts[:, 1],
            triangles=combined.cells.cpu().numpy(),
        )

        for col, field_name in enumerate(fields):
            ### Extract truth/pred tensors and reduce vectors to magnitudes
            true_vals: torch.Tensor = true_flat[field_name]  # ty: ignore[invalid-assignment]
            pred_vals: torch.Tensor = pred_flat[field_name]  # ty: ignore[invalid-assignment]
            is_vector = true_vals.ndim > 1 and true_vals.shape[-1] > 1

            true_scalars_np = _to_scalars(true_vals, is_vector)
            pred_scalars_np = _to_scalars(pred_vals, is_vector)

            ### Shared color limits across truth and prediction
            all_finite = np.concatenate(
                [
                    true_scalars_np[np.isfinite(true_scalars_np)],
                    pred_scalars_np[np.isfinite(pred_scalars_np)],
                ]
            )
            shared_vmin = float(all_finite.min()) if len(all_finite) > 0 else 0.0
            shared_vmax = float(all_finite.max()) if len(all_finite) > 0 else 1.0
            if shared_vmin == shared_vmax:
                shared_vmin -= 1e-6
                shared_vmax += 1e-6

            ### Detect surface-only fields: every triangle has at least one
            # NaN in truth OR pred.  Fields like C_F exist only on the airfoil
            # surface; all interior points are NaN.
            is_surface_field = bool(
                np.all(
                    np.logical_or(
                        np.any(np.isnan(true_scalars_np[tri.triangles]), axis=1),
                        np.any(np.isnan(pred_scalars_np[tri.triangles]), axis=1),
                    )
                )
            )

            if is_surface_field:
                max_magnitude = float(
                    max(
                        np.nanmax(true_scalars_np),
                        np.nanmax(pred_scalars_np),
                    )
                )
                arrow_scale = 0.6 / max_magnitude if max_magnitude > 0 else 1.0
                surface_indices = np.argwhere(
                    ~np.logical_or(
                        np.isnan(true_scalars_np),
                        np.isnan(pred_scalars_np),
                    )
                )

            for row, (key, label) in enumerate(kinds.items()):
                ax = axes[row, col]
                vals: torch.Tensor = kind_data[key][field_name]  # ty: ignore[invalid-assignment]
                scalars_np = _to_scalars(vals, is_vector)

                ### Determine color limits and colormap
                if key == "error":
                    finite_err = scalars_np[np.isfinite(scalars_np)]
                    emax = (
                        float(np.abs(finite_err).max()) if len(finite_err) > 0 else 1.0
                    )
                    if is_vector:
                        cmap, vmin, vmax = "Reds", 0.0, emax
                    else:
                        cmap, vmin, vmax = "RdBu_r", -emax, emax
                else:
                    cmap, vmin, vmax = "turbo", shared_vmin, shared_vmax

                if vmin == vmax:
                    vmin -= 1e-6
                    vmax += 1e-6

                if is_surface_field:
                    ### Surface-only field: arrow visualization
                    vals_np = vals.float().cpu().numpy()
                    color_norm = mpl.colors.Normalize(
                        vmin=0,
                        vmax=max_magnitude,
                    )
                    colormap = plt.get_cmap(cmap)

                    for i in surface_indices[::8]:
                        color = colormap(color_norm(scalars_np[i]))
                        ax.annotate(
                            "",
                            xytext=(tri.x[i].item(), tri.y[i].item()),
                            xy=(
                                tri.x[i].item() + vals_np[i, 0].item() * arrow_scale,
                                tri.y[i].item() + vals_np[i, 1].item() * arrow_scale,
                            ),
                            arrowprops=dict(
                                arrowstyle="-",
                                color=color,
                                alpha=0.7,
                                shrinkA=0,
                                shrinkB=0,
                                lw=2,
                            ),
                            annotation_clip=False,
                        )
                    ax.plot(
                        tri.x[surface_indices],
                        tri.y[surface_indices],
                        "k.",
                        ms=0.2,
                        alpha=1,
                        zorder=1.5,
                    )
                    fig.colorbar(
                        mpl.cm.ScalarMappable(norm=color_norm, cmap=colormap),
                        ax=ax,
                        orientation="horizontal",
                        shrink=0.8,
                        fraction=0.03,
                        aspect=50,
                        pad=0.01,
                    )
                else:
                    ### Volume field: tricontourf + tricontour
                    tri.set_mask(np.any(np.isnan(scalars_np[tri.triangles]), axis=1))
                    extend_kwargs = {"vmin": vmin, "vmax": vmax, "extend": "both"}
                    contf = ax.tricontourf(
                        tri,
                        scalars_np,
                        levels=np.linspace(vmin, vmax, 101),
                        cmap=cmap,
                        zorder=-1,
                        **extend_kwargs,
                    )
                    ax.tricontour(
                        tri,
                        scalars_np,
                        levels=np.linspace(vmin, vmax, 26),
                        colors="k",
                        linewidths=0.2,
                        zorder=1,
                        **extend_kwargs,
                    )
                    ax.set_rasterization_zorder(0)
                    ax.set_facecolor("lightgray")
                    fig.colorbar(
                        mpl.cm.ScalarMappable(
                            norm=contf.norm,
                            cmap=contf.cmap,
                        ),
                        ax=ax,
                        orientation="horizontal",
                        extendrect="both",
                        shrink=0.8,
                        fraction=0.03,
                        aspect=50,
                        pad=0.01,
                    )

                ### Formatting
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_aspect("equal", adjustable="box")
                ax.tick_params(
                    axis="both",
                    which="both",
                    length=0,
                    bottom=False,
                    left=False,
                    labelbottom=False,
                    labelleft=False,
                )
                if row == 0:
                    ax.set_title(field_name, fontsize=12, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(label, fontsize=12, fontweight="bold")

        plt.tight_layout(h_pad=0.1, w_pad=0)
        if show:
            plt.show()

    @staticmethod
    def visualize_output_distributions(
        sample: "AirFRANSSample",
        show: bool = True,
    ) -> None:
        """Visualize distributions of output quantities with histograms.

        Creates a subplot grid showing the distribution of each output
        quantity, with special handling for vector fields (showing magnitude
        distributions).  Prints Polars summary statistics to the logger.

        Args:
            sample: Preprocessed AirFRANS sample whose
                ``prediction_mesh.point_data`` fields are plotted.
            show: Whether to display the plot with ``plt.show()``.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import polars as pl

        point_data = sample.prediction_mesh.point_data.flatten_keys(".")

        def _to_scalar_array(t: torch.Tensor) -> tuple[np.ndarray, bool]:
            """Reduce to 1D numpy; returns ``(array, is_vector)``."""
            is_vector = t.ndim > 1 and t.shape[-1] > 1
            if is_vector:
                t = torch.linalg.norm(t, dim=-1)
            return t.detach().float().cpu().numpy().flatten(), is_vector

        ### Create subplot grid
        field_keys = sorted(point_data.keys())
        n_cols = 3
        n_rows = (len(field_keys) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(15, 5 * n_rows), squeeze=False
        )
        axes_flat = axes.flatten()

        ### Plot distributions
        stats_data: dict[str, np.ndarray] = {}
        for idx, key in enumerate(field_keys):
            values, is_vector = _to_scalar_array(point_data[key])
            suffix = " (magnitude)" if is_vector else ""

            ax = axes_flat[idx]
            ax.hist(values[np.isfinite(values)], bins=50, alpha=0.7, edgecolor="black")
            mean = np.nanmean(values)
            ax.axvline(
                mean,
                color="red",
                linestyle="--",
                label=f"{mean = :.2f}",
                alpha=0.7,
            )
            ax.set_title(f"{key}{suffix} distribution")
            ax.set_xlabel("Magnitude" if is_vector else "Value")
            ax.set_yscale("log")
            ax.set_ylabel("Count (log scale)")
            ax.grid(True, alpha=0.3)
            ax.legend()

            stats_data[f"{key}{suffix}"] = values

        for idx in range(len(field_keys), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.tight_layout()
        if show:
            plt.show()

        ### Print summary statistics using Polars
        logger.info("\n### Summary Statistics ###")
        df = pl.DataFrame(stats_data).fill_nan(None)
        logger.info(f"\n{df.describe()}")


def compute_max_mesh_sizes(
    dataloader: DataLoader,
    device: torch.device,
    *,
    rank: int = 0,
) -> TensorDict[str, TensorDict[Literal["n_points", "n_cells"], Int[torch.Tensor, ""]]]:
    """Compute the maximum n_points and n_cells per boundary-condition type.

    Scans all samples in *dataloader*, tracking the largest boundary mesh
    dimensions for each BC type. Uses distributed all-reduce to find the
    global maximum across all ranks. The results are used to pad meshes to
    uniform sizes for ``torch.compile`` with static shapes.

    Args:
        dataloader: DataLoader yielding ``AirFRANSSample`` objects.
        device: Device for the all-reduce tensors.
        rank: Distributed rank (progress bar shown only on rank 0).

    Returns:
        TensorDict ``{bc_type: {"n_points": Tensor, "n_cells": Tensor}}``
        where each leaf is a scalar integer tensor on *device*.
    """
    ### Accumulate max sizes per BC type using plain ints (fast comparisons)
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
            raw_maxes[bc_type]["n_cells"] = max(
                raw_maxes[bc_type]["n_cells"], mesh.n_cells
            )

    ### Convert to TensorDict and all-reduce across ranks
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
    volume_mesh: Mesh[2, 2],
    airfoil_mesh: Mesh[1, 2],
    chord: float,
) -> TensorDict:
    """Integrate predicted surface fields to obtain section force coefficients.

    Maps volume-mesh predictions to the airfoil boundary mesh via
    nearest-neighbor lookup, then integrates the Cauchy traction
    (pressure + shear) over the airfoil contour to obtain section drag
    and lift coefficients.

    The pressure contribution is reconstructed as ``-C_p * n_outward``
    using the airfoil mesh's geometric normals (oriented outward via the
    divergence theorem).  The shear contribution uses the predicted
    ``C_F,shear`` field directly.

    Args:
        volume_mesh: Interior Mesh with ``point_data["C_p"]`` (scalar)
            and ``point_data["C_F,shear"]`` (2D vector).  Typically the
            model's prediction Mesh or the ground-truth interior Mesh.
        airfoil_mesh: Airfoil boundary Mesh with cell connectivity
            (1D manifold in 2D).  Typically
            ``sample.boundary_meshes["no_slip"]``.
        chord: Reference chord length for normalization.

    Returns:
        TensorDict with scalar-tensor entries ``"Cd"``, ``"Cl"`` (total
        section coefficients) and ``"Cd_pressure"``, ``"Cd_friction"``,
        ``"Cl_pressure"``, ``"Cl_friction"`` (decomposed contributions).

    Example:
        >>> sample = AirFRANSDataSet.preprocess(sample_path)
        >>> coeffs = compute_surface_force_coefficients(
        ...     volume_mesh=sample.prediction_mesh,
        ...     airfoil_mesh=sample.boundary_meshes["no_slip"],
        ...     chord=float(sample.reference_lengths["chord"]),
        ... )
        >>> print(f"Cd={coeffs['Cd']:.5f}, Cl={coeffs['Cl']:.5f}")
    """
    ### Map volume-mesh predictions to airfoil mesh points via KNN
    nearest_idx, _ = knn(points=volume_mesh.points, queries=airfoil_mesh.points, k=1)
    nearest_idx = nearest_idx[:, 0]  # (n_airfoil_points,)

    cp_surface = volume_mesh.point_data["C_p"][nearest_idx]
    cf_shear_surface = volume_mesh.point_data["C_F,shear"][nearest_idx]

    ### Construct a surface Mesh with the mapped predictions
    surface_mesh = Mesh(
        points=airfoil_mesh.points,
        cells=airfoil_mesh.cells,
        point_data=TensorDict(
            {"C_p": cp_surface, "C_F,shear": cf_shear_surface},
            batch_size=[airfoil_mesh.n_points],
        ),
    )

    ### Convert point data to cell-centered values
    surface_with_cells = surface_mesh.point_data_to_cell_data()
    cp_cells = surface_with_cells.cell_data["C_p"]  # (n_cells,)
    cf_shear_cells = surface_with_cells.cell_data["C_F,shear"]  # (n_cells, 2)

    ### Replace NaN values (from non-physical patching) with zero so they
    # contribute nothing to the integral rather than poisoning the sum.
    cp_cells = torch.nan_to_num(cp_cells, nan=0.0)  # ty: ignore[invalid-argument-type]
    cf_shear_cells = torch.nan_to_num(cf_shear_cells, nan=0.0)  # ty: ignore[invalid-argument-type]

    ### Cell geometry
    areas = surface_mesh.cell_areas  # (n_cells,) - edge lengths for 1D manifold
    raw_normals = surface_mesh.cell_normals  # (n_cells, 2)

    # Divergence theorem: integral(x . n) dl = 2 * area > 0 for outward normals.
    outward_sign = torch.sign(
        (surface_mesh.cell_centroids * raw_normals).sum(dim=-1) @ areas
    )
    normals = outward_sign * raw_normals  # (n_cells, 2), guaranteed outward

    ### Integrate surface forces (Cauchy traction on the body surface)
    # Pressure traction: -C_p * n_outward
    # Shear traction:    C_F,shear (already in body-force convention)
    f_pressure = -cp_cells[:, None] * normals * areas[:, None]  # (n_cells, 2)
    f_friction = cf_shear_cells * areas[:, None]  # (n_cells, 2)

    f_pressure_integrated = f_pressure.sum(dim=0) / chord  # (2,)
    f_friction_integrated = f_friction.sum(dim=0) / chord  # (2,)
    f_total_integrated = f_pressure_integrated + f_friction_integrated

    return TensorDict(
        {
            "Cd": f_total_integrated[0],
            "Cl": f_total_integrated[1],
            "Cd_pressure": f_pressure_integrated[0],
            "Cd_friction": f_friction_integrated[0],
            "Cl_pressure": f_pressure_integrated[1],
            "Cl_friction": f_friction_integrated[1],
        }
    )


if __name__ == "__main__":
    import os

    if not (_data_env := os.environ.get("AIRFRANS_DATA_DIR")):
        raise ValueError("AIRFRANS_DATA_DIR environment variable is not set.")
    data_dir = Path(_data_env)
    sample_paths = list(data_dir.iterdir())

    # Preprocess a sample
    sample = AirFRANSDataSet.preprocess(sample_paths[0])

    logger.info(f"Sample path: {sample_paths[0]}")
    logger.info(f"Interior mesh points: {sample.prediction_mesh.points.shape}")
    logger.info(f"Output keys: {list(sample.prediction_mesh.point_data.keys())}")
    logger.info(f"Boundary meshes: {list(sample.boundary_meshes.keys())}")

    ### Sanity-check: divergence theorem should confirm inward raw normals
    airfoil = sample.boundary_meshes["no_slip"]
    outward_sign = torch.sign(
        (airfoil.cell_centroids * airfoil.cell_normals).sum(dim=-1) @ airfoil.cell_areas
    )
    logger.info(f"Airfoil outward normal sign: {outward_sign.item()}")

    AirFRANSDataSet.visualize_output_distributions(sample, show=True)

    combined = AirFRANSDataSet.postprocess(
        pred_mesh=sample.prediction_mesh,
        sample=sample,
    )
    AirFRANSDataSet.visualize_comparison(combined)
