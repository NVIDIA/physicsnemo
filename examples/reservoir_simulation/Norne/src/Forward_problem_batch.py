"""
SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

=====================================================================
 NVIDIA PHYSICSNEMO SURROGATE RESERVOIR SIMULATION FORWARD MODELLING
 (BATCH PROCESSING VERSION)
=====================================================================
@Author : Clement Etienam

This module implements batch processing for reservoir simulation forward modelling
using NVIDIA PhyNeMo. It provides a machine learning framework for predicting
reservoir behavior using neural networks with batch processing capabilities.

Key Features:
- Batch processing for large-scale reservoir simulations
- Multi-GPU support for distributed training
- Neural network models for pressure and saturation prediction
- Comprehensive model evaluation and visualization
- Integration with MLflow for experiment tracking

Usage:
    python Forward_problem_batch.py --config-path=conf --config-name=DECK_CONFIG

Inputs:
    - Configuration file with model parameters
    - Training data from reservoir simulations
    - Test data for model evaluation

Outputs:
    - Trained neural network models
    - Prediction results with evaluation metrics
    - Visualization plots for model performance
"""

# -------------------- 📌 FUTURE IMPORTS -------------------------
# from __future__ import print_function

# 🛠 Standard Library
import os
import copy
import time
import pickle
import logging
import warnings
import multiprocessing
from datetime import timedelta
from pathlib import Path
from typing import Any

# 🔧 Third-party Libraries
import gzip
import scipy.io as sio
import numpy as np
import numpy.linalg
import numpy.matlib
from omegaconf import DictConfig
from cpuinfo import get_cpu_info
from filelock import FileLock

# 🔥 PhyNeMo & ML Libraries
import torch
import hydra
from hydra.utils import to_absolute_path
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad

from physicsnemo.launch.logging import (
    LaunchLogger,
    PythonLogger,
)
from physicsnemo.launch.logging.mlflow import initialize_mlflow
from physicsnemo.distributed import DistributedManager

# 📊 MLFlow & Logging
import mlflow
import mlflow.tracking
from forward.gradients_extract import (
    loss_func,
    loss_func_physics,
    Black_oil_peacemann,
)
from forward.binaries_extract import (
    Black_oil,
    train_polynomial_models,
)
from forward.machine_extract import (
    InitializeLoggers,
    check_and_remove_dirs,
)
from forward.simulator import (
    simulation_data_types,
)
from forward.utils.batch.batch_misc_operation_1 import load_and_setup_training_data
from data_extract.opm_extract_rates import read_compdats2
from forward.gradients_extract import clip_and_convert_to_float32
from forward.utils.batch.training_function import run_training_loop


# 🖥️ Detect GPU
def is_available() -> bool:
    """Check if NVIDIA GPU is available using native Python methods."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Forward problem")
    f_handler = logging.FileHandler(filename="read_vectors.log", mode="w")
    formatter = logging.Formatter(" %(asctime)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    logger.setLevel(logging.INFO)
    return logger


def initialize_environment() -> tuple[bool, int, logging.Logger]:
    """Initialize the environment and return GPU availability, operation mode, and logger."""
    logger = setup_logging()

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Log PyTorch and CUDA information
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")

    # Check GPU availability
    gpu_available = is_available()
    if gpu_available:
        logger.info("GPU Available with CUDA")
        try:
            # import cupy as cp
            operation_mode = 0
        except ImportError:
            operation_mode = 1
    else:
        logger.info("No GPU Available")
        operation_mode = 1

    # Log CPU information
    cpu_info = get_cpu_info()
    logger.info("CPU Info:")
    for key, value in cpu_info.items():
        logger.info(f"\t{key}: {value}")

    warnings.filterwarnings("ignore")
    return gpu_available, operation_mode, logger


def setup_directories(cfg: DictConfig, dist: Any, logger: logging.Logger) -> None:
    """Setup and clean directories for the training process."""
    if cfg.custom.fno_type == "FNO":
        directories_to_check = [
            "checkpoints",
            "__pycache__/",
            "../RUNS",
            "outputs/",
            "../MODELS/FNO/checkpoints_pressure",
            "../MODELS/FNO/checkpoints_saturation",
            "../MODELS/FNO/checkpoints_oil",
            "../MODELS/FNO/checkpoints_gas",
            "../MODELS/FNO/checkpoints_peacemann",
        ]
    else:
        directories_to_check = [
            "checkpoints_seq",
            "__pycache__/",
            "../RUNS",
            "outputs/",
            "../MODELS/PINO/checkpoints_pressure",
            "../MODELS/PINO/checkpoints_saturation",
            "../MODELS/PINO/checkpoints_oil",
            "../MODELS/PINO/checkpoints_gas",
            "../MODELS/PINO/checkpoints_peacemann",
        ]

    check_and_remove_dirs(directories_to_check, cfg.custom.file_response, logger)
    logger.info("|-----------------------------------------------------------------|")


def create_model_directories(cfg: DictConfig, logger: logging.Logger) -> None:
    """Create model checkpoint directories."""
    if cfg.custom.fno_type == "FNO":
        folders_to_create = [
            "../MODELS/FNO/checkpoints_pressure",
            "../MODELS/FNO/checkpoints_saturation",
            "../MODELS/FNO/checkpoints_oil",
            "../MODELS/FNO/checkpoints_gas",
            "../MODELS/FNO/checkpoints_peacemann",
        ]
    else:
        folders_to_create = [
            "../MODELS/PINO/checkpoints_pressure",
            "../MODELS/PINO/checkpoints_saturation",
            "../MODELS/PINO/checkpoints_oil",
            "../MODELS/PINO/checkpoints_gas",
            "../MODELS/PINO/checkpoints_peacemann",
        ]

    for folder in folders_to_create:
        absolute_path = to_absolute_path(folder)
        lock_path = absolute_path + ".lock"
        with FileLock(lock_path):
            if Path(absolute_path).exists():
                logger.info(f"Directory already exists: {absolute_path}")
            else:
                os.makedirs(absolute_path, exist_ok=True)
                logger.info(f"Created directory: {absolute_path}")
    logger.info("|-----------------------------------------------------------------|")


@hydra.main(version_base="1.2", config_path="conf", config_name="DECK_CONFIG")
def main(cfg: DictConfig) -> None:
    """Main function for batch forward problem solving."""
    # Initialize environment
    gpu_available, operation_mode, logger = initialize_environment()

    # Set up array backend
    if gpu_available:
        try:
            import cupy as cp
        except ImportError:
            import numpy as cp
    else:
        import numpy as cp

    # Initialize simulation data types
    (
        type_dict,
        ecl_extensions,
        dynamic_props,
        ecl_vectors,
        static_props,
        SUPPORTED_DATA_TYPES,
    ) = simulation_data_types()
    if cfg.custom.model_Distributed == 1:
        dist, logger = InitializeLoggers(cfg)
        if dist.rank == 0:
            if cfg.custom.fno_type == "FNO":
                directories_to_check = [
                    "checkpoints",
                    "__pycache__/",
                    "../RUNS",
                    "outputs/",
                    "../MODELS/FNO/checkpoints_pressure",
                    "../MODELS/FNO/checkpoints_saturation",
                    "../MODELS/FNO/checkpoints_oil",
                    "../MODELS/FNO/checkpoints_gas",
                    "../MODELS/FNO/checkpoints_peacemann",
                ]
            else:
                directories_to_check = [
                    "checkpoints_seq",
                    "__pycache__/",
                    "../RUNS",
                    "outputs/",
                    "../MODELS/PINO/checkpoints_pressure",
                    "../MODELS/PINO/checkpoints_saturation",
                    "../MODELS/PINO/checkpoints_oil",
                    "../MODELS/PINO/checkpoints_gas",
                    "../MODELS/PINO/checkpoints_peacemann",
                ]
            check_and_remove_dirs(
                directories_to_check, cfg.custom.file_response, logger
            )
            logger.info(
                "|-----------------------------------------------------------------|"
            )
    else:
        if cfg.custom.fno_type == "FNO":
            directories_to_check = [
                "checkpoints",
                "__pycache__/",
                "../RUNS",
                "outputs/",
                "mlruns",
                "../MODELS/FNO/checkpoints_pressure",
                "../MODELS/FNO/checkpoints_saturation",
                "../MODELS/FNO/checkpoints_oil",
                "../MODELS/FNO/checkpoints_gas",
                "../MODELS/FNO/checkpoints_peacemann",
            ]
        else:
            directories_to_check = [
                "checkpoints",
                "__pycache__/",
                "../RUNS",
                "outputs/",
                "mlruns",
                "../MODELS/PINO/checkpoints_pressure",
                "../MODELS/PINO/checkpoints_saturation",
                "../MODELS/PINO/checkpoints_oil",
                "../MODELS/PINO/checkpoints_gas",
                "../MODELS/PINO/checkpoints_peacemann",
            ]
        check_and_remove_dirs(directories_to_check, cfg.custom.file_response, logger)
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        DistributedManager.initialize()
        dist = DistributedManager()
        if "RANK" not in os.environ:
            os.environ["RANK"] = str(dist.rank)
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(dist.rank % torch.cuda.device_count())
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            device_id = dist.rank % gpu_count  # Map rank to available GPUs
            torch.cuda.set_device(device_id)
            logger.info(
                f"Process {dist.rank} is using GPU {device_id}: {torch.cuda.get_device_name(device_id)}"
            )
        else:
            logger.info(f"Process {dist.rank} is using CPU")
        initialize_mlflow(
            experiment_name="PhyNeMo-Reservoir Batch Modelling",
            experiment_desc="PhyNeMo launch development",
            run_name="Reservoir bacth forward modelling",
            run_desc="Reservoir batch forward modelling training",
            user_name="Clement Etienam",
            mode="offline",
        )
        logger = PythonLogger(name=" PhyNeMo Reservoir_Characterisation")
        LaunchLogger.initialize(use_mlflow=cfg.use_mlflow)  # PhyNeMo launch logger
    device = dist.device
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                PHYNEMO RESERVOIR CHARACTERISATION:              |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        if cfg.custom.model_Distributed == 1:
            logger.info(
                "|-----------------------------------------------------------------|"
            )
            logger.info(
                "|                     MULTI GPU USAGE MODEL :                     |"
            )
            logger.info(
                "|-----------------------------------------------------------------|"
            )
        else:
            logger.info(
                "|-----------------------------------------------------------------|"
            )
            logger.info(
                "|                     SINGLE GPU USAGE MODEL :                    |"
            )
            logger.info(
                "|-----------------------------------------------------------------|"
            )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                 TRAIN THE MODEL                                 |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
    oldfolder = os.getcwd()
    os.chdir(oldfolder)
    Relperm = int(cfg.custom.Relperm)
    # interest = cfg.custom.interest
    pde_method = int(cfg.custom.pde_method)
    params = {
        "k_rwmax": torch.tensor(0.3),
        "k_romax": torch.tensor(0.9),
        "k_rgmax": torch.tensor(0.8),
        "n": torch.tensor(2.0),
        "p": torch.tensor(2.0),
        "q": torch.tensor(2.0),
        "m": torch.tensor(2.0),
        "Swi": torch.tensor(0.1),
        "Sor": torch.tensor(0.2),
    }
    if cfg.custom.interest == "Yes":
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        folders_to_create = ["../RUNS", "../data"]
        if dist.rank == 0:
            if os.path.isfile(to_absolute_path("../data/conversions.mat")):
                os.remove(to_absolute_path("../data/conversions.mat"))
            for folder in folders_to_create:
                absolute_path = to_absolute_path(folder)
                lock_path = (
                    absolute_path + ".lock"
                )  # Use a lock file for synchronization
                with FileLock(lock_path):  # Only one process will create the directory
                    if Path(absolute_path).exists():
                        logger.info(f"Directory already exists: {absolute_path}")
                    else:
                        os.makedirs(absolute_path, exist_ok=True)
                        logger.info(f"Created directory: {absolute_path}")
        logger.info(
            "|-----------------------------------------------------------------|"
        )
    if cfg.custom.fno_type == "FNO":
        folders_to_create = [
            "../MODELS/FNO/checkpoints_saturation",
            "../MODELS/FNO/checkpoints_oil",
            "../MODELS/FNO/checkpoints_pressure",
            "../MODELS/FNO/checkpoints_gas",
            "../MODELS/FNO/checkpoints_peacemann",
        ]
    else:
        folders_to_create = [
            "../MODELS/PINO/checkpoints_saturation",
            "../MODELS/PINO/checkpoints_oil",
            "../MODELS/PINO/checkpoints_pressure",
            "../MODELS/PINO/checkpoints_gas",
            "../MODELS/PINO/checkpoints_peacemann",
        ]
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        for folder in folders_to_create:
            absolute_path = to_absolute_path(folder)
            lock_path = absolute_path + ".lock"  # Use a lock file for synchronization
            with FileLock(lock_path):  # Only one process will create the directory
                if Path(absolute_path).exists():
                    logger.info(f"Directory already exists: {absolute_path}")
                else:
                    os.makedirs(absolute_path, exist_ok=True)
                    logger.info(f"Created directory: {absolute_path}")
        logger.info(
            "|-----------------------------------------------------------------|"
        )
    nx = cfg.custom.PROPS.nx
    ny = cfg.custom.PROPS.ny
    nz = cfg.custom.PROPS.nz
    file_path = to_absolute_path("../data/conversions.mat")
    file_exists = os.path.isfile(file_path)
    if file_exists:
        mat = sio.loadmat(file_path)
        steppi = int(mat["steppi"])
        # steppi_indices = mat["steppi_indices"].flatten()
        N_ens = int(mat["N_ens"])
    else:
        steppi = cfg.custom.steppi
        # steppi_indices = np.linspace(1, 164, steppi, dtype=int)
        N_ens = cfg.custom.ntrain
    logger.info(f"Rank {dist.rank}: steppi = {steppi}, N_ens = {N_ens}")
    # oldfolder2 = os.getcwd()
    sourc_dir = cfg.custom.file_location
    source_dir = to_absolute_path(sourc_dir)  # ('../simulator_data')
    effective = np.genfromtxt(Path(source_dir) / "actnum.out", dtype="float")
    effective_i = np.reshape(effective, (nx, ny, nz), "F")
    SWOW = torch.tensor(np.array(np.vstack(cfg.custom.WELLSPECS.SWOW), dtype=float)).to(
        device
    )
    SWOG = torch.tensor(np.array(np.vstack(cfg.custom.WELLSPECS.SWOG), dtype=float)).to(
        device
    )
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                 Learning the interpolation machines    :        |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
    params1_swow, params2_swow = train_polynomial_models(SWOW, device)
    params1_swog, params2_swog = train_polynomial_models(SWOW, device)
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|      Converged  Learning the interpolation machines    :        |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
    BO = float(cfg.custom.PROPS.BO)
    BW = float(cfg.custom.PROPS.BW)
    UW = float(cfg.custom.PROPS.UW)
    UO = float(cfg.custom.PROPS.UO)
    SWI = cp.float32(cfg.custom.PROPS.SWI)
    SWR = cp.float32(cfg.custom.PROPS.SWR)
    CFO = cp.float32(cfg.custom.PROPS.CFO)
    p_atm = cp.float32(float(cfg.custom.PROPS.PATM))
    p_bub = cp.float32(float(cfg.custom.PROPS.PB))

    DZ = torch.tensor(100).to(device)
    RE = torch.tensor(0.2 * 100).to(device)
    Truee1 = np.genfromtxt(Path(source_dir) / "rossmary.GRDECL", dtype="float")
    Trueea = np.reshape(Truee1.T, (nx, ny, nz), "F")
    Trueea = np.reshape(Trueea, (-1, 1), "F")
    Trueea = Trueea * effective.reshape(-1, 1)
    if dist.rank == 0:
        navail = multiprocessing.cpu_count()
        logger.info(f"Available CPU cores: {navail}")
    njobs = max(1, multiprocessing.cpu_count() // 5)  # Ensure at least 1 core is used
    if dist.rank == 0:
        logger.info(f"Using {njobs} cores for parallel processing.")
    sourc_dir = cfg.custom.file_location
    source_dir = to_absolute_path(sourc_dir)  # ('../simulator_data')

    gass, producers, injectors = read_compdats2(
        to_absolute_path(cfg.custom.COMPLETIONS_DATA),
        to_absolute_path(cfg.custom.SUMMARY_DATA),
    )  # filename
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                         PRINT WELLS                           : |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info("gas injectors wells")
        logger.info(f"Gas injectors: {gass}")
        logger.info("producer well")
        logger.info(f"Producers: {producers}")
        logger.info("water injector well")
        logger.info(f"Injectors: {injectors}")
    well_measurements = cfg.custom.well_measurements
    lenwels = len(well_measurements)
    input_variables = cfg.custom.input_properties
    output_variables = cfg.custom.output_properties
    N_pr = len(producers)  # Number of producers
    well_names = [entry[-1] for entry in producers]  # Producer well names
    well_namesg = [entry[-1] for entry in gass]  # gas injectors well names
    well_namesw = [entry[-1] for entry in injectors]  # water injectors well names
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                         PRINT WELL NAMES                      : |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info("producer well names")
        logger.info(f"Producer well names: {well_names}")
        logger.info("gas injectors well names")
        logger.info(f"Gas injector well names: {well_namesg}")
        logger.info("water injector well names")
        logger.info(f"Water injector well names: {well_namesw}")

    with gzip.open(to_absolute_path("../data/static.pkl.gz"), "rb") as f2:
        mat = pickle.load(f2)
    X_data1 = mat
    for key, value in X_data1.items():
        if dist.rank == 0:
            logger.info(f"For key '{key}':")
            logger.info(f"\tContains inf: {np.isinf(value).any()}")
            logger.info(f"\tContains -inf: {np.isinf(-value).any()}")
            logger.info(f"\tContains NaN: {np.isnan(value).any()}")
            logger.info(
                "|-----------------------------------------------------------------|"
            )
    perm_ensemble = X_data1["ensemble"]
    poro_ensemble = X_data1["ensemblep"]
    # fault_ensemble = X_data1["ensemblefault"]
    perm_ensemble = clip_and_convert_to_float32(perm_ensemble)
    poro_ensemble = clip_and_convert_to_float32(poro_ensemble)
    SWI = torch.from_numpy(np.array(SWI)).to(device)
    SWR = torch.from_numpy(np.array(SWR)).to(device)
    UW = torch.from_numpy(np.array(UW)).to(device)
    BW = torch.from_numpy(np.array(BW)).to(device)
    UO = torch.from_numpy(np.array(UO)).to(device)
    BO = torch.from_numpy(np.array(BO)).to(device)
    p_bub = torch.from_numpy(np.array(p_bub)).to(device)
    p_atm = torch.from_numpy(np.array(p_atm)).to(device)
    CFO = torch.from_numpy(np.array(CFO)).to(device)
    mat = sio.loadmat(to_absolute_path("../data/conversions.mat"))
    minK = mat["minK"]
    maxK = mat["maxK"]
    # minT = mat["minT"]
    minP = mat["minP"]
    maxP = mat["maxP"]
    # min_inn_fcn = mat["min_inn_fcn"]
    max_inn_fcn = mat["max_inn_fcn"]
    # min_out_fcn = mat["min_out_fcn"]
    max_out_fcn = mat["max_out_fcn"]
    target_min = 0.01
    target_max = 1
    max_inn_fcnx = torch.from_numpy(max_inn_fcn).to(device)
    max_out_fcnx = torch.from_numpy(max_out_fcn).to(device)

    training_setup = load_and_setup_training_data(
        # Configuration
        input_variables,
        output_variables,
        cfg,
        dist,
        # Model parameters
        N_ens,
        nx,
        ny,
        nz,
        steppi,
        maxP,
        # Well configuration
        N_pr,
        lenwels,
        # Additional parameters
        effective_i,
    )

    labelled_loader_train = training_setup["labelled_loader_train"]
    labelled_loader_testt = training_setup["labelled_loader_testt"]
    composite_model = training_setup["composite_model"]
    combined_optimizer = training_setup["combined_optimizer"]
    input_keys = training_setup["input_keys"]
    input_keys_peacemann = training_setup["input_keys_peacemann"]
    output_keys_peacemann = training_setup["output_keys_peacemann"]
    output_keys_pressure = training_setup["output_keys_pressure"]
    output_keys_saturation = training_setup["output_keys_saturation"]
    output_keys_gas = training_setup["output_keys_gas"]
    output_keys_oil = training_setup["output_keys_oil"]
    use_epoch = training_setup["use_epoch"]
    neededM = training_setup["neededM"]
    MODELS = training_setup["MODELS"]
    MODELS_C = training_setup["MODELS_C"]
    SCHEDULER = training_setup["SCHEDULER"]

    if "PRESSURE" in output_variables:
        surrogate_pressure = MODELS["PRESSURE"]
        optimizer_pressure = MODELS_C["pressure"]
        scheduler_pressure = SCHEDULER["PRESSURE"]
    if "SGAS" in output_variables:
        surrogate_gas = MODELS["SGAS"]
        optimizer_gas = MODELS_C["gas"]
        scheduler_gas = SCHEDULER["SGAS"]
    if "SWAT" in output_variables:
        surrogate_saturation = MODELS["SATURATION"]
        optimizer_saturation = MODELS_C["saturation"]
        scheduler_saturation = SCHEDULER["SATURATION"]
    if "SOIL" in output_variables:
        surrogate_oil = MODELS["SOIL"]
        optimizer_oil = MODELS_C["oil"]
        scheduler_oil = SCHEDULER["SOIL"]
    surrogate_peacemann = MODELS["PEACEMANN"]
    optimizer_peacemann = MODELS_C["peacemann"]
    scheduler_peacemann = SCHEDULER["PEACEMANN"]

    @StaticCaptureTraining(
        model=composite_model,
        optim=combined_optimizer,
        logger=logger,
        use_amp=False,
        use_graphs=True,
    )
    def training_step(
        model,
        inputin,
        inputin_p,
        TARGETS,
        cfg,
        device,
        output_keys_saturation,
        steppi,
        output_variables,
        training_step_metrics,
        UO,
        BO,
        UW,
        BW,
        DZ,
        RE,
        max_inn_fcnx,
        max_out_fcnx,
        params,
        p_bub,
        p_atm,
        CFO,
        Relperm,
        SWI,
        SWR,
        SWOW,
        SWOG,
        params1_swow,
        params2_swow,
        params1_swog,
        params2_swog,
        N_pr,
        lenwels,
        neededM,
        nx,
        ny,
        nz,
        target_min,
        target_max,
        minK,
        maxK,
        minP,
        maxP,
        pde_method,
        max_inn_fcn,
        max_out_fcn,
        epoch,
    ):
        # Prepare input tensors
        # max_epoch = cfg.training.max_steps
        tensors = [
            value for value in inputin.values() if isinstance(value, torch.Tensor)
        ]
        input_tensor = torch.cat(tensors, dim=1)
        input_tensor_p = inputin_p["X"]
        nz = input_tensor.shape[2]
        # batch_size = input_tensor.shape[0]

        # Setup chunking - only if nz > 30
        if nz > cfg.custom.allowable_size:
            fno_expected_nz = max(1, int(nz * 0.1))
            chunk_size = fno_expected_nz
            num_chunks = (nz + chunk_size - 1) // chunk_size
        else:
            fno_expected_nz = chunk_size = nz
            num_chunks = 1

        # Initialize accumulators
        loss = 0
        metrics_accumulator = {
            f"{var}_loss": 0.0
            for var in ["pressure", "water", "oil", "gas", "peacemann"]
        }
        if cfg.custom.fno_type == "PINO":
            pino_metrics = {
                "pressured": 0.0,
                "saturationd": 0.0,
                "gasd": 0.0,
                "peacemanned": 0.0,
            }

        # Process main model chunks
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, nz)
            current_chunk_size = end_idx - start_idx

            # Extract chunks
            input_temp = input_tensor[:, :, start_idx:end_idx, :, :]
            target_chunks = {}

            # Extract target chunks
            for var in output_variables:
                if var == "PRESSURE":
                    target_chunks["pressure"] = {
                        "pressure": TARGETS["PRESSURE"]["pressure"][
                            :, :, start_idx:end_idx, :, :
                        ]
                    }
                elif var == "SWAT":
                    target_chunks["saturation"] = {
                        "water_sat": TARGETS["SATURATION"]["water_sat"][
                            :, :, start_idx:end_idx, :, :
                        ]
                    }
                elif var == "SOIL":
                    target_chunks["oil"] = {
                        "oil_sat": TARGETS["OIL"]["oil_sat"][
                            :, :, start_idx:end_idx, :, :
                        ]
                    }
                elif var == "SGAS":
                    target_chunks["gas"] = {
                        "gas_sat": TARGETS["GAS"]["gas_sat"][
                            :, :, start_idx:end_idx, :, :
                        ]
                    }

            # Pad if needed (only when chunking)
            if nz > cfg.custom.allowable_size:
                pad_size = fno_expected_nz - current_chunk_size
                if pad_size > 0:
                    input_temp = torch.nn.functional.pad(
                        input_temp, (0, 0, 0, 0, 0, pad_size)
                    )
                    for target_type in target_chunks.values():
                        for key in target_type:
                            target_type[key] = torch.nn.functional.pad(
                                target_type[key], (0, 0, 0, 0, 0, pad_size)
                            )

            # Model predictions
            predictions = {}
            if "PRESSURE" in output_variables:
                predictions["pressure"] = model(input_temp, mode="pressure")["pressure"]
            if "SGAS" in output_variables:
                predictions["gas"] = model(input_temp, mode="gas")["gas"]
            if "SWAT" in output_variables:
                predictions["water"] = model(input_temp, mode="saturation")[
                    "saturation"
                ]
            if "SOIL" in output_variables:
                predictions["oil"] = model(input_temp, mode="oil")["oil"]

            # Compute losses
            chunk_loss = 0
            if "PRESSURE" in output_variables:
                pressure_loss = loss_func(
                    predictions["pressure"],
                    target_chunks["pressure"]["pressure"],
                    "eliptical",
                    cfg.loss.weights.pressure,
                    p=2.0,
                )
                chunk_loss += pressure_loss
                metrics_accumulator["pressure_loss"] += pressure_loss.item()

            if "SWAT" in output_variables:
                water_loss = loss_func(
                    predictions["water"],
                    target_chunks["saturation"]["water_sat"],
                    "hyperbolic",
                    cfg.loss.weights.water_sat,
                    p=2.0,
                )
                chunk_loss += water_loss
                metrics_accumulator["water_loss"] += water_loss.item()

            if "SOIL" in output_variables:
                oil_loss = loss_func(
                    predictions["oil"],
                    target_chunks["oil"]["oil_sat"],
                    "hyperbolic",
                    cfg.loss.weights.oil_sat,
                    p=2.0,
                )
                chunk_loss += oil_loss
                metrics_accumulator["oil_loss"] += oil_loss.item()

            if "SGAS" in output_variables:
                gas_loss = loss_func(
                    predictions["gas"],
                    target_chunks["gas"]["gas_sat"],
                    "hyperbolic",
                    cfg.loss.weights.gas_sat,
                    p=2.0,
                )
                chunk_loss += gas_loss
                metrics_accumulator["gas_loss"] += gas_loss.item()

            loss += chunk_loss

            # PINO physics loss
            if (
                cfg.custom.fno_type == "PINO"
                and epoch % max(1, int(0.01 * cfg.training.max_steps)) == 0
            ):
                input_varr = {
                    **{
                        k: v[:, :, start_idx:end_idx, :, :]
                        if isinstance(v, torch.Tensor) and v.dim() > 2
                        else v
                        for k, v in inputin.items()
                    },
                    "pressure": predictions.get("pressure"),
                    "water_sat": predictions.get("water"),
                    "gas_sat": predictions.get("gas"),
                    "oil_sat": predictions.get("oil"),
                }

                # Pad if needed
                if nz > cfg.custom.allowable_size and pad_size > 0:
                    for key in input_varr:
                        if (
                            isinstance(input_varr[key], torch.Tensor)
                            and input_varr[key].dim() > 2
                        ):
                            input_varr[key] = torch.nn.functional.pad(
                                input_varr[key], (0, 0, 0, 0, 0, pad_size)
                            )

                evaluate = Black_oil(
                    input_varr,
                    neededM,
                    SWI,
                    SWR,
                    UW,
                    BW,
                    UO,
                    BO,
                    nx,
                    ny,
                    current_chunk_size
                    if (nz <= cfg.custom.allowable_size or pad_size == 0)
                    else chunk_size,
                    SWOW,
                    SWOG,
                    target_min,
                    target_max,
                    minK,
                    maxK,
                    minP,
                    maxP,
                    p_bub,
                    p_atm,
                    CFO,
                    Relperm,
                    params,
                    pde_method,
                    RE,
                    max_inn_fcn,
                    max_out_fcn,
                    DZ,
                    device,
                    params1_swow,
                    params2_swow,
                    params1_swog,
                    params2_swog,
                )

                f_pressure2 = loss_func_physics(
                    evaluate["pressured"], cfg.loss.weights.pressured
                )
                f_water2 = loss_func_physics(
                    evaluate["saturationd"], cfg.loss.weights.saturationd
                )
                f_gas2 = loss_func_physics(evaluate["gasd"], cfg.loss.weights.gasd)

                chunk_loss += f_pressure2 + f_water2 + f_gas2
                pino_metrics["pressured"] += f_pressure2.item()
                pino_metrics["saturationd"] += f_water2.item()
                pino_metrics["gasd"] += f_gas2.item()

            loss += chunk_loss

        # Process peacemann (no chunking)
        outputs_p = model(input_tensor_p, mode="peacemann")
        peacemann_pred = outputs_p["peacemann"]
        target_peacemann = {
            "Y": TARGETS.get("PEACEMANN", {}).get("Y", torch.zeros_like(peacemann_pred))
        }

        peacemann_loss = loss_func(
            peacemann_pred, target_peacemann["Y"], "peaceman", cfg.loss.weights.Y, p=2.0
        )
        loss += peacemann_loss
        metrics_accumulator["peacemann_loss"] = peacemann_loss.item()

        # Peacemann PINO loss
        if (
            cfg.custom.fno_type == "PINO"
            and epoch % max(1, int(0.01 * cfg.training.max_steps)) == 0
        ):
            inputs1 = {**inputin_p, "Y": peacemann_pred}
            evaluate = Black_oil_peacemann(
                inputs1,
                UO,
                BO,
                UW,
                BW,
                DZ,
                RE,
                device,
                max_inn_fcnx,
                max_out_fcnx,
                params,
                p_bub,
                p_atm,
                steppi,
                CFO,
                Relperm,
                SWI,
                SWR,
                SWOW,
                SWOG,
                params1_swow,
                params2_swow,
                params1_swog,
                params2_swog,
                N_pr,
                lenwels,
            )

            f_peacemann2 = loss_func_physics(
                evaluate["peacemanned"], cfg.loss.weights.peacemanned
            )
            loss += f_peacemann2
            pino_metrics["peacemanned"] = f_peacemann2.item()

        # Average metrics
        for key in metrics_accumulator:
            training_step_metrics[key] = metrics_accumulator[key] / (
                num_chunks if key != "peacemann_loss" else 1
            )

        if cfg.custom.fno_type == "PINO":
            for key in pino_metrics:
                training_step_metrics[key] = pino_metrics[key] / (
                    num_chunks if key != "peacemanned" else 1
                )
        # loss = loss/batch_size
        return loss

    @StaticCaptureEvaluateNoGrad(
        model=composite_model, logger=logger, use_amp=False, use_graphs=True
    )
    def validation_step(
        model,
        inputin,
        inputin_p,
        TARGETS,
        cfg,
        device,
        output_keys_saturation,
        steppi,
        output_variables,
        val_step_metrics,
    ):
        # Prepare input tensors
        # batch_size = input_tensor.shape[0]
        tensors = [
            value for value in inputin.values() if isinstance(value, torch.Tensor)
        ]
        input_tensor = torch.cat(tensors, dim=1)
        input_tensor_p = inputin_p["X"]
        nz = input_tensor.shape[2]
        # batch_size = input_tensor.shape[0]

        # Setup chunking - only if nz > 30
        if nz > cfg.custom.allowable_size:
            fno_expected_nz = max(1, int(nz * 0.1))
            chunk_size = fno_expected_nz
            num_chunks = (nz + chunk_size - 1) // chunk_size
        else:
            fno_expected_nz = chunk_size = nz
            num_chunks = 1

        # Initialize accumulators
        loss = 0
        metrics_accumulator = {
            f"{var}_loss": 0.0
            for var in ["pressure", "water", "oil", "gas", "peacemann"]
        }

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, nz)
            current_chunk_size = end_idx - start_idx

            # Extract chunks
            input_temp = input_tensor[:, :, start_idx:end_idx, :, :]
            target_chunks = {}

            # Extract target chunks
            if "PRESSURE" in output_variables:
                target_chunks["pressure"] = {
                    "pressure": TARGETS["PRESSURE"]["pressure"][
                        :, :, start_idx:end_idx, :, :
                    ]
                }
            if "SWAT" in output_variables:
                target_chunks["saturation"] = {
                    "water_sat": TARGETS["SATURATION"]["water_sat"][
                        :, :, start_idx:end_idx, :, :
                    ]
                }
            if "SOIL" in output_variables:
                target_chunks["oil"] = {
                    "oil_sat": TARGETS["OIL"]["oil_sat"][:, :, start_idx:end_idx, :, :]
                }
            if "SGAS" in output_variables:
                target_chunks["gas"] = {
                    "gas_sat": TARGETS["GAS"]["gas_sat"][:, :, start_idx:end_idx, :, :]
                }

            # Pad if needed (only when chunking)
            if nz > cfg.custom.allowable_size:
                pad_size = fno_expected_nz - current_chunk_size
                if pad_size > 0:
                    input_temp = torch.nn.functional.pad(
                        input_temp, (0, 0, 0, 0, 0, pad_size)
                    )
                    for target_type in target_chunks.values():
                        for key in target_type:
                            target_type[key] = torch.nn.functional.pad(
                                target_type[key], (0, 0, 0, 0, 0, pad_size)
                            )

            # Model predictions
            predictions = {}
            if "PRESSURE" in output_variables:
                predictions["pressure"] = model(input_temp, mode="pressure")["pressure"]
            if "SGAS" in output_variables:
                predictions["gas"] = model(input_temp, mode="gas")["gas"]
            if "SWAT" in output_variables:
                predictions["water"] = model(input_temp, mode="saturation")[
                    "saturation"
                ]
            if "SOIL" in output_variables:
                predictions["oil"] = model(input_temp, mode="oil")["oil"]

            # Compute losses
            chunk_loss = 0
            if "PRESSURE" in output_variables:
                pressure_loss = loss_func(
                    predictions["pressure"],
                    target_chunks["pressure"]["pressure"],
                    "eliptical",
                    cfg.loss.weights.pressure,
                    p=2.0,
                )
                chunk_loss += pressure_loss
                metrics_accumulator["pressure_loss"] += pressure_loss.item()

            if "SWAT" in output_variables:
                water_loss = loss_func(
                    predictions["water"],
                    target_chunks["saturation"]["water_sat"],
                    "hyperbolic",
                    cfg.loss.weights.water_sat,
                    p=2.0,
                )
                chunk_loss += water_loss
                metrics_accumulator["water_loss"] += water_loss.item()

            if "SOIL" in output_variables:
                oil_loss = loss_func(
                    predictions["oil"],
                    target_chunks["oil"]["oil_sat"],
                    "hyperbolic",
                    cfg.loss.weights.oil_sat,
                    p=2.0,
                )
                chunk_loss += oil_loss
                metrics_accumulator["oil_loss"] += oil_loss.item()

            if "SGAS" in output_variables:
                gas_loss = loss_func(
                    predictions["gas"],
                    target_chunks["gas"]["gas_sat"],
                    "hyperbolic",
                    cfg.loss.weights.gas_sat,
                    p=2.0,
                )
                chunk_loss += gas_loss
                metrics_accumulator["gas_loss"] += gas_loss.item()

            loss += chunk_loss

        # Process peacemann (no chunking)
        outputs_p = model(input_tensor_p, mode="peacemann")
        peacemann_pred = outputs_p["peacemann"]
        target_peacemann = {
            "Y": TARGETS.get("PEACEMANN", {}).get("Y", torch.zeros_like(peacemann_pred))
        }

        peacemann_loss = loss_func(
            peacemann_pred, target_peacemann["Y"], "peaceman", cfg.loss.weights.Y, p=2.0
        )
        loss += peacemann_loss
        metrics_accumulator["peacemann_loss"] = peacemann_loss.item()

        # Average metrics
        for key in metrics_accumulator:
            if key != "peacemann_loss":
                val_step_metrics[key] = metrics_accumulator[key] / num_chunks
            else:
                val_step_metrics[key] = metrics_accumulator[key]
        # loss = loss/batch_size
        return loss

    training_step_metrics = {}
    val_step_metrics = {}
    best_pressure = copy.deepcopy(surrogate_pressure)
    best_gas = copy.deepcopy(surrogate_gas)
    best_peacemann = copy.deepcopy(surrogate_peacemann)
    best_saturation = copy.deepcopy(surrogate_saturation)
    best_oil = copy.deepcopy(surrogate_oil)
    start_time = time.time()
    run_training_loop(
        dist=dist,
        logger=logger,
        cfg=cfg,
        mlflow=mlflow,
        use_epoch=use_epoch,
        output_variables=output_variables,
        surrogate_pressure=surrogate_pressure,
        surrogate_gas=surrogate_gas,
        surrogate_saturation=surrogate_saturation,
        surrogate_oil=surrogate_oil,
        surrogate_peacemann=surrogate_peacemann,
        labelled_loader_train=labelled_loader_train,
        labelled_loader_testt=labelled_loader_testt,
        composite_model=composite_model,
        input_keys=input_keys,
        input_keys_peacemann=input_keys_peacemann,
        output_keys_pressure=output_keys_pressure,
        output_keys_gas=output_keys_gas,
        output_keys_saturation=output_keys_saturation,
        output_keys_oil=output_keys_oil,
        output_keys_peacemann=output_keys_peacemann,
        training_step=training_step,
        validation_step=validation_step,
        training_step_metrics=training_step_metrics,
        val_step_metrics=val_step_metrics,
        steppi=steppi,
        UO=UO,
        BO=BO,
        UW=UW,
        BW=BW,
        DZ=DZ,
        RE=RE,
        max_inn_fcnx=max_inn_fcnx,
        max_out_fcnx=max_out_fcnx,
        params=params,
        p_bub=p_bub,
        p_atm=p_atm,
        CFO=CFO,
        Relperm=Relperm,
        SWI=SWI,
        SWR=SWR,
        SWOW=SWOW,
        SWOG=SWOG,
        params1_swow=params1_swow,
        params2_swow=params2_swow,
        params1_swog=params1_swog,
        params2_swog=params2_swog,
        N_pr=N_pr,
        lenwels=lenwels,
        neededM=neededM,
        nx=nx,
        ny=ny,
        nz=nz,
        target_min=target_min,
        target_max=target_max,
        minK=minK,
        maxK=maxK,
        minP=minP,
        maxP=maxP,
        pde_method=pde_method,
        max_inn_fcn=max_inn_fcn,
        max_out_fcn=max_out_fcn,
        scheduler_pressure=scheduler_pressure,
        scheduler_saturation=scheduler_saturation,
        scheduler_oil=scheduler_oil,
        scheduler_gas=scheduler_gas,
        scheduler_peacemann=scheduler_peacemann,
        optimizer_pressure=optimizer_pressure,
        optimizer_saturation=optimizer_saturation,
        optimizer_oil=optimizer_oil,
        optimizer_gas=optimizer_gas,
        optimizer_peacemann=optimizer_peacemann,
        best_pressure=best_pressure,
        best_gas=best_gas,
        best_saturation=best_saturation,
        best_oil=best_oil,
        best_peacemann=best_peacemann,
    )
    if dist.rank == 0:
        mlflow.end_run()
        text = "  Training Converged   "
        logger.info(text)
        logger.info("")
        elapsed_time_secs2 = time.time() - start_time
        msg = (
            "Reservoir Modelling training with Nvidia PhyNeMo took: %s secs (Wall clock time)"
            % timedelta(seconds=round(elapsed_time_secs2))
        )
        logger.info(msg)
        logger.info(msg)
        logger.info("")


if __name__ == "__main__":
    main()
