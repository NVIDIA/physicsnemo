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

from pathlib import Path
import sys
import argparse
import requests
import os
import subprocess
import logging
from typing import Dict, List

import numpy as np
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(processName)s - %(message)s"
)

# URLs for HuggingFace dataset parts
HF_BASE_URL: str = "https://huggingface.co/datasets/ashynf/EFWI/resolve/main"
DATASETS: Dict[str, List[str]] = {
    "CFB": [
        f"{HF_BASE_URL}/CFB/CFB_split.zip",
        f"{HF_BASE_URL}/CFB/CFB_split.z01",
        f"{HF_BASE_URL}/CFB/CFB_split.z02",
        f"{HF_BASE_URL}/CFB/CFB_split.z03",
    ],
    "CFA": [
        f"{HF_BASE_URL}/CFA/CFA_split.zip",
        f"{HF_BASE_URL}/CFA/CFA_split.z01",
        f"{HF_BASE_URL}/CFA/CFA_split.z02",
        f"{HF_BASE_URL}/CFA/CFA_split.z03",
    ],
    "CVA": [
        f"{HF_BASE_URL}/CVA/CVA_split.zip",
        f"{HF_BASE_URL}/CVA/CVA_split.z01",
        f"{HF_BASE_URL}/CVA/CVA_split.z02",
    ],
    "CVB": [
        f"{HF_BASE_URL}/CVB/CVB_split.zip",
        f"{HF_BASE_URL}/CVB/CVB_split.z01",
        f"{HF_BASE_URL}/CVB/CVB_split.z02",
    ],
    "FFA": [
        f"{HF_BASE_URL}/FFA/FFA_split.zip",
        f"{HF_BASE_URL}/FFA/FFA_split.z01",
        f"{HF_BASE_URL}/FFA/FFA_split.z02",
        f"{HF_BASE_URL}/FFA/FFA_split.z03",
    ],
    "FFB": [
        f"{HF_BASE_URL}/FFB/FFB_split.zip",
        f"{HF_BASE_URL}/FFB/FFB_split.z01",
        f"{HF_BASE_URL}/FFB/FFB_split.z02",
        f"{HF_BASE_URL}/FFB/FFB_split.z03",
    ],
    "FVA": [
        f"{HF_BASE_URL}/FVA/FVA_split.zip",
        f"{HF_BASE_URL}/FVA/FVA_split.z01",
        f"{HF_BASE_URL}/FVA/FVA_split.z02",
    ],
    "FVB": [
        f"{HF_BASE_URL}/FVB/FVB_split.zip",
        f"{HF_BASE_URL}/FVB/FVB_split.z01",
        f"{HF_BASE_URL}/FVB/FVB_split.z02",
    ],
}
DATASETS_INFO: Dict[str, Dict[str, int]] = {
    "CFB": {"SAMPLES_PER_FILE": 500, "TRAIN_SAMPLES": 48000, "TEST_SAMPLES": 6000},
    "CFA": {"SAMPLES_PER_FILE": 500, "TRAIN_SAMPLES": 48000, "TEST_SAMPLES": 6000},
    "FFB": {"SAMPLES_PER_FILE": 500, "TRAIN_SAMPLES": 48000, "TEST_SAMPLES": 6000},
    "FFA": {"SAMPLES_PER_FILE": 500, "TRAIN_SAMPLES": 48000, "TEST_SAMPLES": 6000},
    "CVA": {"SAMPLES_PER_FILE": 500, "TRAIN_SAMPLES": 24000, "TEST_SAMPLES": 6000},
    "CVB": {"SAMPLES_PER_FILE": 500, "TRAIN_SAMPLES": 24000, "TEST_SAMPLES": 6000},
    "FVA": {"SAMPLES_PER_FILE": 500, "TRAIN_SAMPLES": 24000, "TEST_SAMPLES": 6000},
    "FVB": {"SAMPLES_PER_FILE": 500, "TRAIN_SAMPLES": 24000, "TEST_SAMPLES": 6000},
}


def download_file_from_url(url: str, local_filename: str, resume: bool = True) -> str:
    """
    Download a file from a direct URL and save it locally.

    Parameters
    ----------
    url : str
        The URL to download from.
    local_filename : str
        The path to save the file to.
    resume : bool, optional
        Whether to resume download of a multi-part zip archive. If True, the
        download will start from the last downloaded chunk. If False, the download
        will start from the beginning.

    Returns
    -------
    str
        The path to the downloaded file.
    """
    logging.info(f"Downloading {os.path.basename(local_filename)} from {url}...")

    response = requests.head(url)
    file_size: int = int(response.headers.get("content-length", 0))

    progress: tqdm = tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        desc=os.path.basename(local_filename),
    )

    if resume and os.path.exists(local_filename):
        logging.info(
            f"{os.path.basename(local_filename)} already exists, skipping download."
        )
        progress.close()
    else:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))
        progress.close()

    return local_filename


def download(name: str, resume: bool = True) -> None:
    """
    Download a dataset from Hugging Face.

    Parameters
    ----------
    name : str
        Name of the dataset to download.
    resume : bool, optional
        Whether to resume download of a multi-part zip archive. If True, the
        download will start from the last downloaded chunk. If False, the download
        will start from the beginning.
    """
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {name}")

    output_dir: Path = Path("./")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download all parts of the zip archive
    zip_parts: list[Path] = []
    for url in DATASETS[name]:
        filename: str = os.path.basename(url)
        output_path: Path = output_dir / filename
        download_file_from_url(url, output_path, resume)
        zip_parts.append(output_path)
    logging.info(f"All parts of {name} dataset downloaded successfully.")

    # Combine multi-part zip archive into a single file
    logging.info(f"Combining zip parts for {name} dataset...")
    combined_zip: Path = output_dir / "_temp_combined.zip"
    try:
        subprocess.run(
            [
                "zip",
                "-s",
                "0",
                str(zip_parts[0]),
                "--out",
                str(combined_zip),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to combine zip parts") from exc

    # Remove original zip parts
    for zip_part in zip_parts:
        zip_part.unlink()

    # Extract the combined zip archive
    logging.info(f"Extracting {name} dataset to {output_dir}...")
    try:
        subprocess.run(
            ["unzip", str(combined_zip), "-d", str(output_dir)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to extract combined zip archive") from exc

    # Cleanup combined archive
    combined_zip.unlink()

    logging.info(f"Download and extraction of {name} dataset completed.")


def reorganize(
    dataset_names: list[str], clean: bool = False, shuffle: bool = False
) -> None:
    """
    Reorganize the dataset by:
    1. Reorganizing files into individual samples
    2. Combine individual samples files into a single directory

    Parameters
    ----------
    dataset_names : list[str]
        List of dataset names to reorganize.
    clean : bool, optional
        Whether to delete original files after reorganizing.
    shuffle : bool, optional
        Whether to shuffle train and test samples.
    """
    # Process all datasets if 'all' is in the list
    if "all" in dataset_names:
        names: list[str] = list(DATASETS.keys())
    else:
        # Validate dataset names
        for name in dataset_names:
            if name not in DATASETS:
                raise ValueError(f"Unsupported dataset: {name}")
        names: list[str] = dataset_names

    # Size of the combined dataset
    train_samples_all: int = sum(DATASETS_INFO[name]["TRAIN_SAMPLES"] for name in names)
    test_samples_all: int = sum(DATASETS_INFO[name]["TEST_SAMPLES"] for name in names)
    total_samples_all: int = train_samples_all + test_samples_all

    # Setup shuffling
    if shuffle:
        logging.info("Shuffling enabled.")
        np.random.seed(123)
        # Generate a random set of file indices from 0 to total_samples-1
        random_indices: np.ndarray = np.random.permutation(total_samples_all).tolist()
    else:
        # If not shuffling, just use sequential indices
        random_indices: np.ndarray = np.arange(total_samples_all).tolist()

    output_dir: Path = Path("_".join(names)) / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        logging.info(f"Reorganizing {name} dataset to {output_dir}...")
        dataset_dir: Path = Path(f"./{name}")

        # Get list of file indices by looking at vs files
        vs_files: list[Path] = sorted(
            dataset_dir.glob("vs_*.npy"), key=lambda x: int(x.stem.split("_")[-1])
        )
        if not vs_files:
            logging.warning(
                f"No files found for dataset {name}. Skipping reorganizing."
            )
            continue

        # Keep track of processed files to delete later
        processed_files: set[Path] = set()

        # Process each file
        for file_idx, vs_file in enumerate(
            tqdm(vs_files, desc="Processing files", unit="file")
        ):
            file_num: int = int(vs_file.stem.split("_")[-1])

            # Define file paths
            vs_file_path: Path = dataset_dir / f"vs_{file_num}.npy"
            vp_file_path: Path = dataset_dir / f"vp_{file_num}.npy"
            data_x_file_path: Path = dataset_dir / f"data_x_{file_num}.npy"
            data_z_file_path: Path = dataset_dir / f"data_z_{file_num}.npy"

            # Add files to processed list
            processed_files.add(vs_file_path)
            processed_files.add(vp_file_path)
            processed_files.add(data_x_file_path)
            processed_files.add(data_z_file_path)

            # Load all quantities for this file index
            try:
                vs_data: np.ndarray = np.load(vs_file_path, mmap_mode="r")
                vp_data: np.ndarray = np.load(vp_file_path, mmap_mode="r")
                data_x: np.ndarray = np.load(data_x_file_path, mmap_mode="r")
                data_z: np.ndarray = np.load(data_z_file_path, mmap_mode="r")
            except FileNotFoundError as e:
                logging.warning(f"Error loading files for index {file_num}: {e}")
                continue

            # Get number of samples in this file
            num_samples: int = len(vs_data)

            # Create individual files for each sample
            for sample_idx in range(num_samples):
                sample_data: Dict[str, np.ndarray] = {
                    "vs": vs_data[sample_idx],
                    "vp": vp_data[sample_idx],
                    "ux": data_x[sample_idx],
                    "uz": data_z[sample_idx],
                }

                # Get next file index and remove it
                rnd_global_sample_idx: int = random_indices.pop(0)

                # Save the combined sample data
                is_train: bool = rnd_global_sample_idx < train_samples_all
                output_file: Path = (
                    output_dir
                    / f"{'train' if is_train else 'test'}_sample_{rnd_global_sample_idx}.npz"
                )
                np.savez(output_file, **sample_data)

                del sample_data

            # Explicitly delete memory-mapped arrays to free resources
            del vs_data, vp_data, data_x, data_z

            # Delete the original files after processing only if clean is True
            if clean:
                vs_file_path.unlink()
                vp_file_path.unlink()
                data_x_file_path.unlink()
                data_z_file_path.unlink()

                # Find and delete other files with the same index pattern
                # (like pm_* and pr_*)
                other_pattern_files: list[Path] = list(
                    dataset_dir.glob(f"*_{file_num}.npy")
                )
                for file_path in other_pattern_files:
                    if file_path not in processed_files:
                        try:
                            file_path.unlink()
                        except FileNotFoundError:
                            pass
                logging.info("All original files have been deleted.")
        logging.info(
            f"Reorganizing of {name} dataset to {output_dir} completed successfully!"
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and reorganize into individual samples files E-FWI datasets."
    )

    parser.add_argument("--download", action="store_true", help="Download the dataset")

    parser.add_argument(
        "--resume-download",
        action="store_true",
        help="Resume download of a multi-part zip archive. "
        "Otherwise, the download will start from the beginning.",
    )

    parser.add_argument(
        "--reorganize",
        action="store_true",
        help="Reorganize the dataset into individual samples files",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete original files after reorganizing",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle train and test samples using fixed random seed 123",
    )

    parser.add_argument(
        "--name",
        nargs="+",
        default=["all"],
        help="Names of datasets to process (default: all)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate dataset names
    if "all" not in args.name:
        for name in args.name:
            if name not in DATASETS:
                logging.error(
                    f"Error: Unknown dataset '{name}'"
                    f"Available datasets: {', '.join(DATASETS.keys())}"
                )
                sys.exit(1)

    # Check if at least one action is specified
    if not (args.download or args.reorganize):
        logging.error("Error: No action specified. Use --download or --reorganize")
        sys.exit(1)

    # Download if requested
    if args.download:
        if "all" in args.name:
            for dataset in DATASETS.keys():
                download(dataset, resume=args.resume_download)
        else:
            for dataset in args.name:
                download(dataset, resume=args.resume_download)

    # Reorganize if requested
    if args.reorganize:
        reorganize(args.name, args.clean, args.shuffle)
