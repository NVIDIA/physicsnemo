"""Configuration utilities for AirfRANS dataset paths."""

import platform
from pathlib import Path


def get_data_dir() -> Path:
    """
    Get the AirfRANS dataset directory based on the current hostname.

    Returns:
        Path to the AirfRANS Dataset directory.

    Raises:
        ValueError: If the hostname is not recognized.
    """
    hostname = platform.node()

    if hostname == "NV-pds":  # local
        return Path("/home/psharpe/gh/aerodynamics_datasets/airfrans/Dataset")
    elif hostname.endswith("eos.clusters.nvidia.com"):  # EOS
        return Path(
            "/lustre/fsw/coreai_modulus_cae/psharpe/aerodynamics_datasets/airfrans/Dataset"
        )
    elif hostname.startswith("nvl72"):  # OCI-HSG
        return Path(
            "/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/aerodynamics_datasets/airfrans/Dataset"
        )
    elif hostname.startswith("batch-block"):  # OCI-ORD
        return Path(
            "/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/aerodynamics_datasets/airfrans/Dataset"
        )
    else:
        raise ValueError(f"Unknown hostname: {hostname!r}")
