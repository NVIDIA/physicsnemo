# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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
import logging
import glob
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.multiprocessing as mp

import deepwave
from deepwave import elastic

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(processName)s - %(message)s"
)


def classify_lithology(
    vp: np.ndarray,
    vs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify lithology element-wise based on Vp, Vs, and Vp/Vs ratio.

    Parameters
    ----------
    vp : np.ndarray
        P-wave velocity in m/s (2D array)
    vs : np.ndarray
        S-wave velocity in m/s (2D array)

    Returns
    -------
    lith : np.ndarray
        Array of rock type strings
    alpha : np.ndarray
        Gardner coefficients
    beta : np.ndarray
        Gardner coefficients
    salt_mask : np.ndarray
        Boolean mask to override density with fixed value

    .. note::

        Adapted from:
        - `Gardner et al. (1974), Geophysics, <https://doi.org/10.1190/1.1440465>`_
        - Mavko et al. (2009), "The Rock Physics Handbook"
        - Castagna et al. (1985), Geophysics, 50(4), 571-581
        - Gray & Head (2000), "Modeling, migration, and velocity analysis in salt", Geophysics
    """

    vpr: np.ndarray = vp / vs  # Vp/Vs ratio
    lith: np.ndarray = np.full(vp.shape, "Unknown", dtype=object)
    alpha: np.ndarray = np.zeros_like(vp, dtype=float)
    beta: np.ndarray = np.zeros_like(vp, dtype=float)

    # Lithology Masks
    shale_mask: np.ndarray = vpr > 2.0  # Castagna et al., 1985

    sandstone_mask: np.ndarray = (
        (vpr >= 1.6) & (vpr <= 2.2) & (vp >= 2500) & (vp <= 5000) & ~shale_mask
    )

    limestone_mask: np.ndarray = (
        (vp >= 5000)
        & (vp <= 6500)
        & (vpr >= 1.7)
        & (vpr <= 1.95)
        & ~shale_mask
        & ~sandstone_mask
    )

    dolomite_mask: np.ndarray = (
        (vp >= 5500)
        & (vp <= 7000)
        & (vpr >= 1.65)
        & (vpr <= 1.9)
        & ~shale_mask
        & ~sandstone_mask
        & ~limestone_mask
    )

    coal_mask: np.ndarray = (
        (vp < 3600)
        & (vpr > 1.8)
        & ~shale_mask
        & ~sandstone_mask
        & ~limestone_mask
        & ~dolomite_mask
    )

    anhydrite_mask: np.ndarray = (
        (vp >= 5800)
        & (vp <= 6800)
        & (vpr <= 1.8)
        & ~shale_mask
        & ~dolomite_mask
        & ~limestone_mask
    )

    # Salt: Vp ~4500 m/s, Vs ≈ 0 → large Vp/Vs
    salt_mask: np.ndarray = (
        (vp >= 4300)
        & (vp <= 4700)
        & (vs < 700)
        & (vpr >= 6.0)  # Vs near zero  # Very high Vp/Vs
    )

    # Assign Lithologies and Gardner Parameters
    # Gardner et al. (1974), Mavko et al. (2009)
    lith[shale_mask] = "Shale"
    alpha[shale_mask], beta[shale_mask] = 0.31, 0.2928

    lith[sandstone_mask] = "Sandstone"
    alpha[sandstone_mask], beta[sandstone_mask] = 0.25, 0.28

    lith[limestone_mask] = "Limestone"
    alpha[limestone_mask], beta[limestone_mask] = 0.30, 0.25

    lith[dolomite_mask] = "Dolomite"
    alpha[dolomite_mask], beta[dolomite_mask] = 0.29, 0.25

    lith[coal_mask] = "Coal"
    alpha[coal_mask], beta[coal_mask] = 0.24, 0.25

    lith[anhydrite_mask] = "Anhydrite"
    alpha[anhydrite_mask], beta[anhydrite_mask] = 0.27, 0.25

    lith[salt_mask] = "Salt"
    # Do not assign alpha/beta for salt, use fixed density

    # Fallback
    fallback_mask: np.ndarray = (alpha == 0) & (~salt_mask)
    lith[fallback_mask] = "Unknown"
    alpha[fallback_mask], beta[fallback_mask] = 0.31, 0.25  # Generic fallback

    return lith, alpha, beta, salt_mask


def compute_density(
    vp: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    salt_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute density using Gardner's rule.

    Parameters
    ----------
    vp : np.ndarray
        P-wave velocity in m/s
    alpha : np.ndarray
        Gardner coefficients (same shape as vp)
    beta : np.ndarray
        Gardner coefficients (same shape as vp)
    salt_mask : np.ndarray | None
        Optional boolean mask to fix salt density

    Returns
    -------
    rho : np.ndarray
        Estimated density (g/cm³)

    .. note::

        For salt, we override Gardner with a fixed value:
        :math:`\rho = 2.15 \text{ g/cm}^3`
        (Gray & Head, 2000; Mavko et al.)
    """
    rho: np.ndarray = alpha * vp**beta
    if salt_mask is not None:
        rho: np.ndarray = np.where(salt_mask, 2.15, rho)
    return rho.astype(float)


# Core processing function for a single file
def process_file(
    input_path: str,
    output_path: str,
    device_id: int | str,
    source_frequency: int = 15,
) -> tuple[str, str]:
    """
    Process a single file and save the results.

    Parameters
    ----------
    input_path : str
        Path to the input file.
    output_path : str
        Path to the output directory.
    device_id : int | str
        GPU ID to use for processing. If value provided is not an integer,
        the function will run on CPU.
    source_frequency : int, optional
        Source Ricker wavelet peak frequency in Hz. Defaults to 15.

    Returns
    -------
    tuple[str, str]
        Tuple containing the output file path and a status message.
    """

    try:
        device: torch.device = (
            torch.device(f"cuda:{device_id}")
            if isinstance(device_id, int)
            else torch.device("cpu")
        )

        original_data: Dict[str, np.ndarray] = np.load(input_path)
        vp_np: np.ndarray = original_data["vp"]
        vs_np: np.ndarray = original_data["vs"]

        if vp_np.ndim > 2:
            vp_np: np.ndarray = vp_np.reshape(-1, 70, 70)[0]
        if vs_np.ndim > 2:
            vs_np: np.ndarray = vs_np.reshape(-1, 70, 70)[0]

        # pr_np = compute_poisson_ratio(vp_np, vs_np)
        # rock_type = infer_rock_type(pr_np)
        # a, b = get_gardner_constants(rock_type)
        # rho_np = gardner_density(vp_np, a, b)

        lith, alpha, beta, salt_mask = classify_lithology(vp_np, vs_np)
        rho_np: np.ndarray = compute_density(vp_np, alpha, beta, salt_mask)

        vp: torch.Tensor = torch.from_numpy(vp_np).float().to(device)
        vs: torch.Tensor = torch.from_numpy(vs_np).float().to(device)
        rho: torch.Tensor = torch.from_numpy(rho_np).float().to(device)

        # ny: int = 70
        # nx: int = 70
        dx: float = 5.0
        nt: int = 1000
        dt: float = 0.001
        freq: int = source_frequency
        peak_time: float = 1.5 / freq
        n_shots: int = 5
        source_depth: int = 1
        receiver_depth: int = 1
        n_receivers_per_shot: int = 69

        source_locations: torch.Tensor = torch.zeros(
            n_shots, 1, 2, dtype=torch.long, device=device
        )
        source_locations[..., 0] = source_depth
        source_locations[:, 0, 1] = torch.arange(n_shots) * 17

        receiver_locations: torch.Tensor = torch.zeros(
            n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=device
        )
        receiver_locations[..., 0] = receiver_depth
        receiver_locations[:, :, 1] = torch.arange(n_receivers_per_shot).repeat(
            n_shots, 1
        )

        source_amplitudes = (
            deepwave.wavelets.ricker(freq, nt, dt, peak_time)
            .repeat(n_shots, 1, 1)
            .to(device)
            * 100000.0
        )

        receiver_amplitudes_z, receiver_amplitudes_x = elastic(
            *deepwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho),
            grid_spacing=dx,
            dt=dt,
            source_amplitudes_y=source_amplitudes,
            source_amplitudes_x=source_amplitudes,
            source_locations_y=source_locations,
            source_locations_x=source_locations,
            receiver_locations_y=receiver_locations,
            receiver_locations_x=receiver_locations,
            pml_freq=freq,
            pml_width=[20, 20, 20, 20],
        )[-2:]

        # Add back channel dimension
        vp_np = vp_np[None, ...]
        vs_np = vs_np[None, ...]
        rho_np = rho_np[None, ...]

        # Move time dimension to height
        vx_np = receiver_amplitudes_x.cpu().numpy().transpose(0, 2, 1)
        vz_np = receiver_amplitudes_z.cpu().numpy().transpose(0, 2, 1)

        output_data: Dict[str, np.ndarray | torch.Tensor] = {
            "vp": vp_np,
            "vs": vs_np,
            "rho": rho_np,
            "vx": vx_np,
            "vz": vz_np,
        }

        os.makedirs(output_path, exist_ok=True)
        output_filepath: str = os.path.join(output_path, os.path.basename(input_path))

        np.savez(output_filepath, **output_data)
        return (output_filepath, "Success")

    except Exception as e:
        logging.error(f"FAILED to process {input_path}: {e}", exc_info=True)
        return (input_path, f"Failed: {e}")


def process_file_wrapper(args_tuple):
    """
    Helper function to unpack arguments for use with multiprocessing.Pool's
    imap functions, which only accept a single argument.
    """
    return process_file(*args_tuple)


# Main function to discover files and distribute work
def main():
    """
    Finds all .npz files and distributes them, logging progress every 1000 files.
    """

    # Setup argument parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Process individual vs and vp samples in .npz files. "
        "Infers lithology and density for each sample and generate "
        "new .npz files with vx and vz."
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Path to the dataset directory containing the .npz files. "
        "Samples files are expected to be of the form in_dir/samples/train_sample_<idx>.npz "
        "or in_dir/samples/test_sample_<idx>.npz.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path to the output directory where the new .npz files will be saved. "
        "New files will be of the form out_dir/samples/train_sample_<idx>.npz "
        "or out_dir/samples/test_sample_<idx>.npz.",
    )
    parser.add_argument(
        "--source_frequency",
        type=int,
        default=15,
        help="Peak frequency (Hz) of the Ricker source wavelet used during "
        "forward modeling. Defaults to 15.",
    )
    args = parser.parse_args()

    dataset_path: Path = Path(args.in_dir) / "samples"
    output_path: Path = Path(args.out_dir) / "samples"
    file_list: list[str] = sorted(
        glob.glob(os.path.join(dataset_path, "train_sample_*.npz")),
        key=lambda x: int(Path(x).stem.split("_")[-1]),
    ) + sorted(
        glob.glob(os.path.join(dataset_path, "test_sample_*.npz")),
        key=lambda x: int(Path(x).stem.split("_")[-1]),
    )

    if not file_list:
        logging.warning(f"No files found in {dataset_path}. Please check paths.")
        return

    total_files: int = len(file_list)
    logging.info(f"Found {total_files} files to process in {dataset_path}.")
    logging.info(f"Saving files to {output_path}.")

    results: list[tuple[str, str]] = []
    num_gpus: int = torch.cuda.device_count()
    user_source_frequency: int = args.source_frequency

    if num_gpus == 0:
        logging.warning("No GPUs found. Running on CPU. This will be very slow.")
        args: list[tuple[str, str, str, int]] = [
            (filepath, output_path, "cpu", user_source_frequency)
            for filepath in file_list
        ]

        for i, arg in enumerate(args):
            results.append(process_file(*arg))
            if (i + 1) % 1000 == 0:
                logging.info(f"Processed {i + 1} / {total_files} files")
    else:
        logging.info(f"Found {num_gpus} GPUs. Starting parallel processing.")
        args: list[tuple[str, str, int, int]] = [
            (filepath, output_path, i % num_gpus, user_source_frequency)
            for i, filepath in enumerate(file_list)
        ]

        with mp.get_context("spawn").Pool(processes=num_gpus) as pool:
            iterator = pool.imap_unordered(process_file_wrapper, args)

            for i, result in enumerate(iterator):
                results.append(result)
                if (i + 1) % 1000 == 0:
                    logging.info(f"Processed {i + 1} / {total_files} files")

    success_count: int = sum(1 for r in results if r[1] == "Success")
    logging.info("Processing Complete.")
    logging.info(f"{success_count} / {total_files} files processed successfully.")

    failed_files = [r for r in results if r[1] != "Success"]
    if failed_files:
        logging.warning("Failed Files")
        for f, reason in failed_files:
            logging.warning(f"{os.path.basename(f)}: {reason}")


if __name__ == "__main__":
    main()
