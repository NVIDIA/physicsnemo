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
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, Sequence

import pyvista as pv
import torch
from jaxtyping import Bool, Float, Int
from tensordict import TensorDict, tensorclass
from torch.distributed import ReduceOp, all_reduce, is_initialized
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from physicsnemo.experimental.models.globe.utilities.cached_dataset import (
    CachedPreprocessingDataset,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus import compute_point_derivatives
from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.projections import project
from physicsnemo.nn.functional.knn import knn
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


@tensorclass
class AirFRANSSample:
    interior_mesh: Mesh  # Point cloud with nondimensional point_data and global_data
    boundary_meshes: TensorDict[str, Mesh]  # BC name -> Mesh
    reference_lengths: TensorDict[
        str, Float[torch.Tensor, ""]
    ]  # reference length names to scalar tensors
    dimensional_constants: (
        TensorDict  # U_inf, q_inf - only for postprocessing / redimensionalization
    )

    @property
    def model_input_kwargs(self) -> dict:
        """Kwargs for :meth:`GLOBE.forward`, minus control-flow args like ``chunk_size``."""
        return {
            "prediction_points": self.interior_mesh.points,
            "boundary_meshes": self.boundary_meshes,
            "reference_lengths": self.reference_lengths,
            "global_data": self.interior_mesh.global_data,
        }

    if TYPE_CHECKING:

        def to(self, *args: Any, **kwargs: Any) -> Self:
            """Move sample and all nested data to the specified device/dtype.

            All tensors in ``interior_mesh``, ``boundary_meshes``,
            ``reference_lengths``, and ``dimensional_constants`` are moved
            together.

            Parameters
            ----------
            *args : Any
                Positional arguments forwarded to the underlying tensorclass
                ``to`` method.  Common usage: ``sample.to("cuda")`` or
                ``sample.to(torch.float32)``.
            **kwargs : Any
                Keyword arguments forwarded to the underlying tensorclass
                ``to`` method.

            Keyword Arguments
            -----------------
            device : torch.device, optional
                Target device.
            dtype : torch.dtype, optional
                Target floating-point or complex dtype.
            non_blocking : bool, optional
                Whether the transfer should be non-blocking.

            Returns
            -------
            AirFRANSSample
                A new sample on the target device/dtype.
            """
            ...


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
        num_workers: int = 8,
    ) -> DataLoader:
        """Create a distributed DataLoader for this dataset.

        Each item is a single sample (``batch_size=None``) with identity
        collation, suitable for variable-size mesh data that cannot be
        stacked into uniform batches.

        Args:
            sample_paths: Paths to individual sample directories.
            cache_dir: Directory for disk caching of preprocessed samples.
            world_size: Total number of distributed ranks.
            rank: This process's distributed rank.
            num_workers: Number of DataLoader worker processes.

        Returns:
            Configured DataLoader with distributed sampling.
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
        q: Float[torch.Tensor, " n_points"] = q_inf * U_over_U_inf.square().sum(dim=-1)
        C_pt: Float[torch.Tensor, " n_points"] = (p + q) / q_inf

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

        # The physicsnemo 1D-manifold normal is a 90-degree CCW rotation of the
        # edge tangent, whose sign depends on the vertex ordering in the VTP file.
        # For AirFRANS airfoil meshes the raw normals point INTO the body; the
        # -1 flip produces the body-outward normal (into the fluid), which is the
        # convention expected by the Cauchy traction formula:
        #   F_body = sigma · n_body = (-p I + 2 nu eps) · n_body
        airfoil_normals: Float[torch.Tensor, "n_points 2"] = (
            -1 * airfoil.point_normals[nearest_airfoil_idx]
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
            interior_mesh=Mesh(
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
        pred_mesh: Mesh,
        true_mesh: Mesh,
        *,
        fields: Sequence[str | tuple[str, ...]] | None = None,
        show: bool = True,
        show_error: bool = True,
    ) -> Mesh:
        """Visualize and compare predicted vs. true fields on a combined Mesh.

        Builds a combined Mesh whose ``point_data`` contains nested ``"true"``,
        ``"pred"``, and ``"error"`` TensorDicts, then renders a subplot grid
        using :meth:`Mesh.draw`.

        Args:
            pred_mesh: Point-cloud Mesh with predicted field values in
                ``point_data``.
            true_mesh: Mesh with ground-truth field values in ``point_data``.
                Should have cell connectivity (from preprocessing) for
                filled-polygon rendering.
            fields: Which field names to compare. If ``None``, uses the sorted
                intersection of pred and true ``point_data`` keys.
            show: Whether to display the plot via ``plt.show()``.
            show_error: Whether to include an error row in the subplot grid.

        Returns:
            Combined Mesh with ``point_data["true"]``, ``point_data["pred"]``,
            and ``point_data["error"]`` containing the selected fields.

        Raises:
            ValueError: If pred_mesh and true_mesh have different numbers of
                points.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt

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

        ### Build combined Mesh with nested point_data
        pred_selected = pred_mesh.point_data.select(*fields)
        true_selected = true_mesh.point_data.select(*fields)
        error_data: TensorDict = pred_selected.apply(  # ty: ignore[invalid-assignment]
            lambda p, t: p - t, true_selected
        )

        combined = Mesh(
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
        )

        ### Create subplot grid
        kind_data = {"true": true_selected, "pred": pred_selected, "error": error_data}
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

        for col, field_name in enumerate(fields):
            ### Compute shared vmin/vmax across truth and prediction
            true_vals: torch.Tensor = true_selected[field_name]  # ty: ignore[invalid-assignment]
            pred_vals: torch.Tensor = pred_selected[field_name]  # ty: ignore[invalid-assignment]
            is_vector = true_vals.ndim > 1 and true_vals.shape[-1] > 1
            true_scalars = (
                true_vals.norm(dim=-1) if is_vector else true_vals.reshape(-1)
            )
            pred_scalars = (
                pred_vals.norm(dim=-1) if is_vector else pred_vals.reshape(-1)
            )

            all_finite = torch.cat(
                [
                    true_scalars[torch.isfinite(true_scalars)],
                    pred_scalars[torch.isfinite(pred_scalars)],
                ]
            )
            shared_vmin = all_finite.min().item() if len(all_finite) > 0 else 0.0
            shared_vmax = all_finite.max().item() if len(all_finite) > 0 else 1.0
            if shared_vmin == shared_vmax:
                shared_vmin -= 1e-6
                shared_vmax += 1e-6

            for row, (key, label) in enumerate(kinds.items()):
                ax = axes[row, col]
                vals: torch.Tensor = kind_data[key][field_name]  # ty: ignore[invalid-assignment]

                if key == "error":
                    if is_vector:
                        finite_err = vals.norm(dim=-1)
                        finite_err = finite_err[torch.isfinite(finite_err)]
                        emax = finite_err.max().item() if len(finite_err) > 0 else 1.0
                        cmap, vmin, vmax = "Reds", 0.0, emax
                    else:
                        finite_err = vals.reshape(-1)
                        finite_err = finite_err[torch.isfinite(finite_err)]
                        emax = (
                            finite_err.abs().max().item()
                            if len(finite_err) > 0
                            else 1.0
                        )
                        cmap, vmin, vmax = "RdBu_r", -emax, emax
                else:
                    cmap, vmin, vmax = "turbo", shared_vmin, shared_vmax

                combined.draw(
                    point_scalars=vals,
                    ax=ax,
                    show=False,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    show_edges=False,
                )

                sm = mpl.cm.ScalarMappable(
                    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                    cmap=plt.get_cmap(cmap),
                )
                fig.colorbar(
                    sm,
                    ax=ax,
                    orientation="horizontal",
                    shrink=0.8,
                    fraction=0.03,
                    aspect=50,
                    pad=0.01,
                )

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
                    title = (
                        ".".join(field_name)
                        if isinstance(field_name, tuple)
                        else field_name
                    )
                    ax.set_title(title, fontsize=12, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(label, fontsize=12, fontweight="bold")

        plt.tight_layout(h_pad=0.1, w_pad=0)
        if show:
            plt.show()

        return combined

    @staticmethod
    def visualize_output_distributions(
        output_dict: TensorDict[str, Float[torch.Tensor, "..."]],
        show: bool = True,
    ) -> None:
        """Visualize distributions of output quantities with histograms.

        Creates a subplot grid showing the distribution of each output quantity,
        with special handling for vector fields (showing magnitude distributions).

        Args:
            output_dict: Dictionary of output tensors from preprocessing
            show: Whether to display the plot with plt.show()
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import polars as pl

        ### Helper function to get plottable values
        def get_plot_values(values: torch.Tensor) -> tuple[np.ndarray, str]:
            """Convert tensor to plottable array and determine label."""
            if values.ndim > 1 and values.shape[-1] > 1:
                # Vector quantity - return magnitude
                return torch.linalg.norm(
                    values, dim=-1
                ).detach().cpu().numpy(), " (magnitude)"
            else:
                return values.detach().cpu().numpy().flatten(), ""

        ### Create subplot grid
        n_outputs = len(output_dict.keys())
        n_cols = 3
        n_rows = (n_outputs + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_outputs > 1 else [axes]

        ### Plot distributions
        for idx, (key, values) in enumerate(output_dict.items()):
            ax = axes[idx]
            plot_values, suffix = get_plot_values(values)

            # Histogram with mean line
            ax.hist(plot_values, bins=50, alpha=0.7, edgecolor="black")
            mean = np.nanmean(plot_values)
            ax.axvline(
                mean,
                color="red",
                linestyle="--",
                label=f"{mean = :.2f}",
                alpha=0.7,
            )

            # Formatting
            ax.set_title(f"{key}{suffix} distribution")
            ax.set_xlabel("Value" if not suffix else "Magnitude")
            ax.set_yscale("log")
            ax.set_ylabel("Count (log scale)")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        if show:
            plt.show()

        ### Print summary statistics using Polars
        logger.info("\n### Summary Statistics ###")
        stats_data = {
            f"{key}{get_plot_values(values)[1]}": get_plot_values(values)[0]
            for key, values in output_dict.items()
        }
        df = pl.DataFrame(stats_data)
        # Replace NaN values with nulls so Polars handles them properly
        df = df.fill_nan(None)
        logger.info(f"\n{df.describe()}")


def compute_max_mesh_sizes(
    dataloader: DataLoader,
    device: torch.device,
    *,
    face_downsampling_ratio: float = 1.0,
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
        face_downsampling_ratio: Scale factor applied to cell counts. Use
            a value < 1.0 for training (downsampled meshes) and 1.0 for
            validation (full meshes).
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
            n_cells = (
                int(mesh.n_cells * face_downsampling_ratio)
                if face_downsampling_ratio != 1.0
                else mesh.n_cells
            )
            raw_maxes[bc_type]["n_cells"] = max(
                raw_maxes[bc_type]["n_cells"],
                n_cells,
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


if __name__ == "__main__":
    import os

    if not (_data_env := os.environ.get("AIRFRANS_DATA_DIR")):
        raise ValueError("AIRFRANS_DATA_DIR environment variable is not set.")
    data_dir = Path(_data_env)
    sample_paths = list(data_dir.iterdir())

    # Preprocess a sample
    sample = AirFRANSDataSet.preprocess(sample_paths[0])

    logger.info(f"Sample path: {sample_paths[0]}")
    logger.info(f"Interior mesh points: {sample.interior_mesh.points.shape}")
    logger.info(f"Output keys: {list(sample.interior_mesh.point_data.keys())}")
    logger.info(f"Boundary meshes: {list(sample.boundary_meshes.keys())}")

    # Visualize the output distributions
    output_dict = sample.interior_mesh.point_data
    AirFRANSDataSet.visualize_output_distributions(output_dict, show=True)

    AirFRANSDataSet.postprocess(
        pred_mesh=sample.interior_mesh,
        true_mesh=sample.interior_mesh,
    )
