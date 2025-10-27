"""
SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
 NVIDIA PHYSICSNEMO SURROGATE RESERVOIR SIMULATION DATA EXTRACTION
=====================================================================
@Author : Clement Etienam
"""

# 🛠 Standard Library
import os
import pickle
import logging
import warnings
from omegaconf import DictConfig
import gzip
import scipy.io as sio
import numpy as np
import numpy.linalg
import numpy.matlib
import multiprocessing
from shutil import rmtree
from cpuinfo import get_cpu_info
from filelock import FileLock
from gstools.random import MasterRNG

# 🏎️ Parallel Processing & Multiprocessing
from joblib import Parallel, delayed

# 🔥 Torch & PhyNeMo
import torch

import hydra
from hydra.utils import to_absolute_path

from physicsnemo.launch.logging import (
    LaunchLogger,
    PythonLogger,
)
from physicsnemo.launch.logging.mlflow import initialize_mlflow
from physicsnemo.distributed import DistributedManager

# 🎨 Externenal libraries
from forward.simulator import round_array_to_4dp
from forward.gradients_extract import (
    clip_and_convert_to_float32,
    clean_dict_arrays,
    scale_tensor_abs as scale_command,
    scale_tensor_abs_pressure as scale_command_pressure,
    scale_tensor_absS as scale_commandS,
    scale_tensor_absSin as scale_commandSin,
)
from forward.binaries_extract import check_and_remove_dirs
from forward.machine_extract import Get_data_FFNN
from data_extract.opm_extract_rates import (
    read_compdats2,
    read_compdats,
    safe_mean_std,
)
from data_extract.opm_extract_props_geostats import (
    process_task,
    Geta_all,
    copy_files,
    save_files,
    Run_simulator,
)


def simulation_data_types():
    """Return common Eclipse/Flow dictionaries for parsing keywords.

    Provides SUPPORTED_DATA_TYPES, type_dict, ecl_extensions, dynamic_props,
    ecl_vectors, and static_props used by parsing helpers in this module.
    """
    SUPPORTED_DATA_TYPES = {
        "INTE": (4, "i", 1000),
        "REAL": (4, "f", 1000),
        "LOGI": (4, "i", 1000),
        "DOUB": (8, "d", 1000),
        "CHAR": (8, "8s", 105),
        "MESS": (8, "8s", 105),
        "C008": (8, "8s", 105),
    }

    type_dict = {
        b"INTE": "i",
        b"CHAR": "8s",
        b"REAL": "f",
        b"DOUB": "d",
        b"LOGI": "4s",
        b"MESS": "?",
    }

    ecl_extensions = [
        ".DATA",
        ".DBG",
        ".ECLEND",
        ".EGRID",
        ".FEGRID",
        ".FGRID",
        ".FINIT",
        ".FINSPEC",
        ".FRFT",
        ".FRSSPEC",
        ".FSMSPEC",
        ".FUNRST",
        ".FUNSMRY",
        ".GRID",
        ".INIT",
        ".INSPEC",
        ".MSG",
        ".PRT",
        ".RFT",
        ".RSM",
        ".RSSPEC",
        ".SMSPEC",
        ".UNRST",
        ".UNSMRY",
        ".dbprtx",
    ]

    dynamic_props = [
        "SEQNUM",
        "PRESSURE",
        "SWAT",
        "SGAS",
        "SOIL",
        "RS",
        "RV",
        "RSSAT",
        "RVSAT",
        "STATES",
        "OWC",
        "OGC",
        "GWC",
        "EOWC",
        "EOGC",
        "OILAPI",
        "SDENO",
        "FIPOIL",
        "RFIPOIL",
        "FIPGAS",
        "RFIPGAS",
        "FIPWAT",
        "RFIPWAT",
        "SFIPOIL",
        "SFIPGAS",
        "SFIPWAT",
        "SFIPPLY",
        "RFIPPLY",
        "SFIPSAL",
        "RFIPSAL",
        "SFIPSOL",
        "SFIPGGI",
        "RFIPOIL",
        "RFIPGAS",
        "RFIPWAT",
        "RFIPSOL",
        "RFIPGGI",
        "OIL-POTN",
        "GAS-POTN",
        "WAT-POTN",
        "POLYMER",
        "PADS",
        "PLYTRRFA",
        "POLYMAX",
        "SALT",
        "TEMP",
        "XMF",
        "YMF",
        "ZMF",
        "SSOL",
        "PBUB",
        "PDEW",
        "SURFACT",
        "SURFADS",
        "SURFMAX",
        "SURFCNM",
        "SURFST",
        "GGI",
        "WAT-PRES",
        "GAS-PRES",
        "OIL-VISC",
        "WAT-VISC",
        "GAS-VISC",
        "OIL-DEN",
        "WAT-DEN",
        "GAS-DEN",
        "DRAINAGE",
        "DRAINMIN",
        "PCOW",
        "PCOG",
        "1OVERBO",
        "1OVERBW",
        "1OVERBG",
        "POT_CORR",
        "OILKR",
        "WATKR",
        "GASKR",
        "HYDH",
        "HYDHFW",
        "PORV",
        "RPORV",
        "FOAM",
        "FOAMADS",
        "FOAMMAX",
        "FOAMDCY",
        "FOAMCNM",
        "FOAM_HL",
        "FOAMMOB",
        "ALKALINE",
        "ALKADS",
        "ALKMAX",
        "STMALK",
        "SFADALK",
        "PLADALK",
        "PADMAX",
        "CATSURF",
        "CATROCK",
        "ESALSUR",
        "ESALPLY",
        "COALGAS",
        "COALSOLV",
        "GASSATC",
        "MLANG",
        "MLANGSLV",
        "SWMIN",
        "SWMAX",
        "ISTHW",
        "SOMAX",
        "ISTHG",
        "SGMIN",
        "SGMAX",
        "PRESROCC",
        "CNV_OIL",
        "CNV_WAT",
        "CNV_GAS",
        "CNV_PLY",
        "TRANEXX",
        "TRANEXY",
        "TRANEXZ",
        "EXCAVNUM",
        "CNV_SAL",
        "CNV_SOL",
        "CNV_GGI",
        "CNV_DPRE",
        "CNV_DWAT",
        "CNV_DGAS",
        "CNV_DPLY",
        "CNV_DSAL",
        "CNV_DSOL",
        "CNV_DGGI",
        "CONV_VBR",
        "CONV_PRU",
        "CONV_NEW",
        "FLOOILI+",
        "FLOOILJ+",
        "FLOOILK+",
        "FLOGASI+",
        "FLOGASJ+",
        "FLOGASK+",
        "FLOWATI+",
        "FLOWATJ+",
        "FLOWATK+",
        "FLROILI+",
        "FLROILJ+",
        "FLROILK+",
        "FLRGASI+",
        "FLRGASJ+",
        "FLRGASK+",
        "FLRWATI+",
        "FLRWATJ+",
        "FLRWATK+",
        "VOILI+",
        "VOILJ+",
        "VOILK+",
        "VGASI+",
        "VGASJ+",
        "VGASK+",
        "VWATI+",
        "VWATJ+",
        "VWATK+",
        "FLOOILN+",
        "FLOGASN+",
        "FLOWATN+",
        "FLOOILL+",
        "FLOGASL+",
        "FLOWATL+",
        "FLOOILA+",
        "FLOGASA+",
        "FLOWATA+",
        "FLROILN+",
        "FLRGASN+",
        "FLRWATN+",
        "FLROILL+",
        "FLRGASL+",
        "FLRWATL+",
        "FLROILA+",
        "FLRGASA+",
        "FLRWATA+",
    ]

    ecl_vectors = [
        "COPR",
        "COPT",
        "CWFR",
        "CWIR",
        "CWPR",
        "CWPT",
        "FGIR",
        "FGIT",
        "FGLIR",
        "FGOR",
        "FGORH",
        "FGPR",
        "FGPT",
        "FLPR",
        "FLPT",
        "FMCTP",
        "FMWWO",
        "FMWWT",
        "FODEN",
        "FOE",
        "FOIP",
        "FOPR",
        "FOPRF",
        "FOPRH",
        "FOPRS",
        "FOPT",
        "FOPTH",
        "FPR",
        "FVIR",
        "FVIT",
        "FVPR",
        "FVPT",
        "FWCT",
        "FWCTH",
        "FWIP",
        "FWIR",
        "FWIT",
        "FWPR",
        "FWPT",
        "GGOR",
        "GGPR",
        "GGPT",
        "GOPR",
        "GOPT",
        "GVIR",
        "GVIT",
        "GVPR",
        "GVPT",
        "GWCT",
        "GWIR",
        "GWPR",
        "MSUMLINS",
        "RGPV",
        "RHPV",
        "ROE",
        "ROEW",
        "ROPV",
        "ROSAT",
        "RPR",
        "RRPV",
        "RWPV",
        "TCPU",
        "TIME",
        "WBHP",
        "WBHPH",
        "WBP",
        "WBP4",
        "WBP9",
        "WGIR",
        "WGIT",
        "WGLIR",
        "WGOR",
        "WGORH",
        "WGPR",
        "WGPRH",
        "WGPTH",
        "WLPR",
        "WLPRH",
        "WLPT",
        "WLPTH",
        "WMCON",
        "WMCTL",
        "WOPR",
        "WOPRH",
        "WOPT",
        "WOPTH",
        "WPI",
        "WTHP",
        "WTICIW1",
        "WTICIW2",
        "WTIRIW1",
        "WTIRIW2",
        "WTPCIW1",
        "WTPCIW2",
        "WTPRIW1",
        "WTPRIW2",
        "WWCT",
        "WWCTH",
        "WWIR",
        "WWIRH",
        "WWIT",
        "WWITH",
        "WWPR",
        "WWPRH",
        "WWPT",
        "WWPTH",
        "YEARS",
    ]

    static_props = [
        "DEPTH",
        "DX",
        "DR",
        "DY",
        "DTHETA",
        "DZ",
        "PORO",
        "PERMX",
        "PERMR",
        "PERMI",
        "PERMY",
        "PERMTHT",
        "PERMJ",
        "PERMZ",
        "PERMK",
        "MULTX",
        "MULTR",
        "MULTI",
        "MULTY",
        "MULTTHT",
        "MULTJ",
        "MULTZ",
        "MULTK",
        "TRANX",
        "TRANR",
        "TRANI",
        "TRANY",
        "TRANTHT",
        "TRANJ",
        "TRANZ",
        "TRANK",
        "DIFFMX",
        "DIFFMR",
        "DIFFMI",
        "DIFFMY",
        "DIFFMTHT",
        "DIFFMJ",
        "DIFFMZ",
        "DIFFMK",
        "DIFFX",
        "DIFFR",
        "DIFFI",
        "DIFFY",
        "DIFFTHT",
        "DIFFJ",
        "DIFFZ",
        "DIFFK",
        "DIFFTX",
        "DIFFTR",
        "DIFFTI",
        "DIFFTY",
        "DIFFTTHT",
        "DIFFTJ",
        "DIFFTZ",
        "DIFFTK",
        "HEATTX",
        "HEATTR",
        "HEATTY",
        "HEATTTHT",
        "MLANGI",
        "GASSATC",
        "MLNGSLVI",
        "MLANG",
        "GASSATC",
        "MLANGSLV",
        "AQUIFERN",
        "DOMAINS",
        "ENDNUM",
        "EQLNUM",
        "FIPNUM",
        "FLUXNUM",
        "KRO",
        "KRORW",
        "KRW",
        "KRWR",
        "MINPVV",
        "MULTNUM",
        "MULTPV",
        "MULTX",
        "MULTX-",
        "MULTY",
        "MULTY-",
        "MULTZ",
        "MULTZ-",
        "NTG",
        "OPERNUM",
        "PCW",
        "PORV",
        "PVTNUM",
        "SATNUM",
        "SOWCR",
        "SWATINIT",
        "SWCR",
        "SWL",
        "SWLPC",
        "SWU",
        "TOPS",
        "TRANNNC",
    ]

    return (
        type_dict,
        ecl_extensions,
        dynamic_props,
        ecl_vectors,
        static_props,
        SUPPORTED_DATA_TYPES,
    )


(
    type_dict,
    ecl_extensions,
    dynamic_props,
    ecl_vectors,
    static_props,
    SUPPORTED_DATA_TYPES,
) = simulation_data_types()


# 🖥️ Detect GPU
def is_available():
    """Check if NVIDIA GPU is available using `nvidia-smi`."""
    return os.system("nvidia-smi") == 0


# 🚨 Suppress Warnings
warnings.filterwarnings("ignore")


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        f_handler = logging.FileHandler(filename="read_vectors.log", mode="w")
        formatter = logging.Formatter(" %(asctime)s - %(levelname)s - %(message)s")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
        logger.setLevel(logging.INFO)
    warnings.filterwarnings("ignore")
    return logger


@hydra.main(version_base="1.2", config_path="conf", config_name="DECK_CONFIG")
def main(cfg: DictConfig) -> None:
    # Environment introspection (moved from import time)
    logger = setup_logging()
    logger.info("PyTorch Version: %s", torch.__version__)
    logger.info("CUDA Version: %s", torch.version.cuda)
    logger.info("cuDNN Version: %s", torch.backends.cudnn.version())
    logger.info("CUDA Available: %s", torch.cuda.is_available())
    # GPU detection and selection of array backend
    if is_available():
        logger.info("GPU Available with CUDA")
        # import cupy as cp  # type: ignore
    else:
        logger.info("No GPU Available")
        # import numpy as cp  # type: ignore

    # CPU info
    cpu_info = get_cpu_info()
    logger.info("CPU Info:")
    for k, v in cpu_info.items():
        logger.info(f"\t{k}: {v}")
    if cfg.custom.fno_type == "FNO":
        directories_to_check = [
            "__pycache__/",
            "outputs/",
        ]
    else:
        directories_to_check = [
            "__pycache__/",
            "outputs/",
        ]
        check_and_remove_dirs(directories_to_check, cfg.custom.file_response)
        logger.info(
            "|-----------------------------------------------------------------|"
        )

        # Initialize distributed manager
        DistributedManager.initialize()
        dist = DistributedManager()
        if "RANK" not in os.environ:
            os.environ["RANK"] = str(dist.rank)
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(dist.rank % torch.cuda.device_count())

        # Assign GPU or CPU
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
            experiment_name="PhyNeMo-Reservoir Modelling",
            experiment_desc="PhyNeMo launch development",
            run_name="Reservoir forward modelling",
            run_desc="Reservoir forward modelling training",
            user_name="Clement Etienam",
            mode="offline",
        )

        # General python logger
        logger = PythonLogger(name="PhyNeMo Reservoir_Characterisation")
        LaunchLogger.initialize(use_mlflow=cfg.use_mlflow)  # PhyNeMo launch logger

    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                DATA EXTRACTION MODULE :                         |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
    oldfolder = os.getcwd()
    os.chdir(oldfolder)

    if cfg.custom.interest == "Yes":
        folders_to_create = ["../RUNS", "../data"]

        # bb = os.path.isfile(to_absolute_path('../data/conversions.mat'))
        if dist.rank == 0:
            if os.path.isfile(to_absolute_path("../data/conversions.mat")):
                os.remove(to_absolute_path("../data/conversions.mat"))
            for folder in folders_to_create:
                absolute_path = to_absolute_path(folder)
                lock_path = (
                    absolute_path + ".lock"
                )  # Use a lock file for synchronization
                with FileLock(lock_path):  # Only one process will create the directory
                    if os.path.exists(absolute_path):
                        logger.info(f"Directory already exists: {absolute_path}")
                    else:
                        os.makedirs(absolute_path, exist_ok=True)
                        logger.info(f"Created directory: {absolute_path}")
    if dist.rank == 0:
        logger.info(
            "|-----------------------------------------------------------------|"
        )

        for folder in folders_to_create:
            absolute_path = to_absolute_path(folder)
            lock_path = absolute_path + ".lock"  # Use a lock file for synchronization
            with FileLock(lock_path):  # Only one process will create the directory
                if os.path.exists(absolute_path):
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

    # steppi,steppi_indices,N_ens = None, None,None
    # --- If file exists, load the data ---
    if file_exists:
        mat = sio.loadmat(file_path)
        steppi = int(mat["steppi"])
        steppi_indices = mat["steppi_indices"].flatten()
        N_ens = int(mat["N_ens"])
    # --- If file does not exist, get values from cfg.custom ---
    else:
        steppi = cfg.custom.steppi
        # steppi_indices = cfg.custom.steppi_indices
        steppi_indices = np.linspace(1, 164, steppi, dtype=int)
        N_ens = cfg.custom.ntrain

    # Print confirmation on each rank
    logger.info(f"Rank {dist.rank}: steppi = {steppi}, N_ens = {N_ens}")

    oldfolder2 = os.getcwd()

    sourc_dir = cfg.custom.file_location
    source_dir = to_absolute_path(sourc_dir)  # ('../simulator_data')

    effective = np.genfromtxt(os.path.join(source_dir, "actnum.out"), dtype="float")

    filenamea = os.path.basename(cfg.custom.DECK)

    filenameui = os.path.splitext(filenamea)[0]
    if cfg.custom.numerical_solver == "flow":
        if cfg.custom.model_Distributed == 2:
            string_simulation = f"mpirun --allow-run-as-root -np 1 flow {filenamea} --parsing-strictness=low --enable-ecl-output=true"
        else:
            string_simulation = f"mpirun --oversubscribe --allow-run-as-root -np 25 flow {filenamea} --parsing-strictness=low --enable-ecl-output=true"

    else:
        # f'"ecl100 {os.path.basename(plan["custom"]["DECK"])}"'
        # string_simulation= 'ecl100' + filenamea
        string_simulation = f"ecl100 {filenamea}"

    Truee1 = np.genfromtxt(os.path.join(source_dir, "rossmary.GRDECL"), dtype="float")

    Trueea = np.reshape(Truee1.T, (nx, ny, nz), "F")
    Trueea = np.reshape(Trueea, (-1, 1), "F")
    Trueea = Trueea * effective.reshape(-1, 1)
    if dist.rank == 0:
        navail = multiprocessing.cpu_count()
        logger.info(f"Available CPU cores: {navail}")
    njobs = max(1, multiprocessing.cpu_count() // 5)  # Ensure at least 1 core is used
    # njobs = 1

    if dist.rank == 0:
        logger.info(f"Using {njobs} cores for parallel processing.")

    sourc_dir = cfg.custom.file_location
    source_dir = to_absolute_path(sourc_dir)  # ('../simulator_data')
    # dest_dir = 'path_to_folder_B'

    minn = float(cfg.custom.PROPS.minn)
    maxx = float(cfg.custom.PROPS.maxx)
    minnp = float(cfg.custom.PROPS.minnp)
    maxxp = float(cfg.custom.PROPS.maxxp)

    # producers = cfg.custom.WELLSPECS.producer_wells

    filename = to_absolute_path(cfg.custom.COMPLETIONS_DATA)
    filename_fault = to_absolute_path(cfg.custom.FAULT_DATA)
    FAULT_INCLUDE = cfg.custom.FAULT_INCLUDE
    PERMX_INCLUDE = cfg.custom.PERMX_INCLUDE
    PORO_INCLUDE = cfg.custom.PORO_INCLUDE

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
        logger.info(str(gass))
        logger.info("producer well")
        logger.info(str(producers))
        logger.info("water injector well")
        logger.info(str(injectors))

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
        logger.info(str(well_names))
        logger.info("gas injectors well names")
        logger.info(str(well_namesg))
        logger.info("water injector well names")
        logger.info(str(well_namesw))

    compdat_data = read_compdats(filename, well_names)  # OIl
    compdat_datag = read_compdats(filename, well_namesg)  # gas
    compdat_dataw = read_compdats(filename, well_namesw)  # water

    if cfg.custom.interest == "Yes":
        minn = float(cfg.custom.PROPS.minn)
        maxx = float(cfg.custom.PROPS.maxx)
        minnp = float(cfg.custom.PROPS.minnp)
        maxxp = float(cfg.custom.PROPS.maxxp)

        # perm_ensemble, poro_ensemble, fault_ensemble = None, None,None
        if cfg.custom.geostats.type_geo == 1:
            x = np.arange(nx)
            y = np.arange(ny)
            z = np.arange(nz)
            seed_master = MasterRNG(20170519)

            if dist.rank == 0:
                logger.info(
                    "|-----------------------------------------------------------------|"
                )
                logger.info(
                    "|    Generating Ensemble of permeability and porosity fields    : |"
                )
                logger.info(
                    "|-----------------------------------------------------------------|"
                )

                results = Parallel(n_jobs=njobs)(
                    delayed(process_task)(
                        k,
                        x,
                        y,
                        z,
                        seed_master(),
                        minn,
                        maxx,
                        minnp,
                        maxxp,
                        cfg.custom.geostats.var,
                        cfg.custom.geostats.len_scale,
                    )
                    for k in range(N_ens)
                )

                # Allocate arrays after knowing the size from the first result
                perm_ensemble = np.zeros((len(results[0][0]), N_ens))
                poro_ensemble = np.zeros((len(results[0][1]), N_ens))

                # Store the results in the arrays
                for k, (fout, fout1) in enumerate(results):
                    perm_ensemble[:, k] = fout
                    poro_ensemble[:, k] = fout1

                ensemble_size = len(results[0][0])  # Use the same size as perm_ensemble
                fault_ensemble = np.zeros((ensemble_size, N_ens))
                for i in range(N_ens):
                    X = np.clip(
                        0.5 + 0.15 * np.random.randn(ensemble_size), 0, 1
                    )  # Transform standard normal to (0,1)
                    fault_ensemble[:, i] = X  # Store in the ensemble

                X_ensemble = {
                    "ensemble": perm_ensemble,
                    "ensemblep": poro_ensemble,
                    "ensemblefault": fault_ensemble,
                }

                with gzip.open(
                    to_absolute_path("../data/static.pkl.gz"), "wb"
                ) as f1:
                    pickle.dump(X_ensemble, f1)

                logger.info(
                    "|-----------------------------------------------------------------|"
                )
                logger.info(
                    "|             Converged Generating Ensemble    :                  |"
                )
                logger.info(
                    "|-----------------------------------------------------------------|"
                )

                # logger.info('here')
        else:
            if dist.rank == 0:
                perm_ensemble = np.genfromtxt(
                    os.path.join(source_dir, "sgsim.out"), dtype="float"
                )
                # np.genfromtxt(os.path.join(source_dir, "sgsim.out"), dtype='float')
                poro_ensemble = np.genfromtxt(
                    os.path.join(source_dir, "sgsimporo.out"), dtype="float"
                )
                fault_ensemble = np.genfromtxt(
                    os.path.join(source_dir, "faultensemble.dat"), dtype="float"
                )

                perm_ensemble = clip_and_convert_to_float32(perm_ensemble)
                poro_ensemble = clip_and_convert_to_float32(poro_ensemble)
                fault_ensemble = clip_and_convert_to_float32(fault_ensemble)

                X_ensemble = {
                    "ensemble": perm_ensemble,
                    "ensemblep": poro_ensemble,
                    "ensemblefault": fault_ensemble,
                }

                with gzip.open(
                    to_absolute_path("../data/static.pkl.gz"), "wb"
                ) as f1:
                    pickle.dump(X_ensemble, f1)
                # logger.info('here')
                logger.info(
                    "|-----------------------------------------------------------------|"
                )
                logger.info(
                    "|             Converged Generating Ensemble    :                  |"
                )
                logger.info(
                    "|-----------------------------------------------------------------|"
                )

    with gzip.open(to_absolute_path("../data/static.pkl.gz"), "rb") as f2:
        mat = pickle.load(f2)

    X_data1 = mat
    del mat
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
    fault_ensemble = X_data1["ensemblefault"]

    perm_ensemble = clip_and_convert_to_float32(perm_ensemble)
    poro_ensemble = clip_and_convert_to_float32(poro_ensemble)

    if dist.rank == 0:
        for kk in range(N_ens):
            path_out = to_absolute_path("../RUNS/Realisation" + str(kk))
            os.makedirs(path_out, exist_ok=True)

        Parallel(n_jobs=njobs, backend="loky", verbose=10)(
            delayed(copy_files)(
                source_dir, to_absolute_path("../RUNS/Realisation" + str(kk))
            )
            for kk in range(N_ens)
        )

        Parallel(n_jobs=njobs, backend="loky", verbose=10)(
            delayed(save_files)(
                perm_ensemble[:, kk],
                poro_ensemble[:, kk],
                fault_ensemble[:, kk],
                to_absolute_path("../RUNS/Realisation" + str(kk)),
                oldfolder,
                FAULT_INCLUDE,
                PERMX_INCLUDE,
                PORO_INCLUDE,
            )
            for kk in range(N_ens)
        )

        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                 RUN FLOW SIMULATOR FOR ENSEMBLE                  |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )

        Parallel(n_jobs=njobs, backend="loky", verbose=5)(
            delayed(Run_simulator)(
                to_absolute_path("../RUNS/Realisation" + str(kk)),
                oldfolder2,
                string_simulation,
            )
            for kk in range(N_ens)
        )

        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                 EXECUTED RUN of  FLOW SIMULATION FOR ENSEMBLE   |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )

        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                 DATA CURRATION IN PROCESS                       |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )
        N = N_ens
        pressure = []
        Sgas = []
        Swater = []
        Soil = []
        Faulta = []
        Timea = []
        Qga = []
        Qoa = []
        Qwa = []

        permeability = np.zeros((N, 1, nx, ny, nz))
        porosity = np.zeros((N, 1, nx, ny, nz))
        actnumm = np.zeros((N, 1, nx, ny, nz))
        for i in range(N):
            folder = to_absolute_path("../RUNS/Realisation" + str(i))

            return_values = Geta_all(
                folder,
                nx,
                ny,
                nz,
                oldfolder,
                steppi,
                steppi_indices,
                filenameui,
                injectors,
                gass,
                filename,
                compdat_datag,
                compdat_dataw,
                compdat_data,
                input_variables,
                output_variables,
                filename_fault,
                FAULT_INCLUDE,
            )

            # Dynamically extract variables using dictionary keys
            Pr = (
                return_values.get("PRESSURE", None)
                if "PRESSURE" in output_variables
                else None
            )
            sw = return_values.get("SWAT", None) if "SWAT" in output_variables else None
            sg = return_values.get("SGAS", None) if "SGAS" in output_variables else None
            so = return_values.get("SOIL", None) if "SOIL" in output_variables else None

            Time = return_values.get("Time")  # Mandatory value
            # actnum = return_values.get("actnum", None)  # Uncomment if needed
            QG = return_values.get("QG")  # Extract gas rates
            QW = return_values.get("QW")  # Extract water rates
            QO = return_values.get("QO")  # Extract oil rates

            # Extract FAULT if present in input_variables
            flt = (
                return_values.get("FAULT", None) if "FAULT" in input_variables else None
            )

            # Process the extracted variables
            if "PRESSURE" in output_variables:
                Pr = round_array_to_4dp(clip_and_convert_to_float32(Pr))
                pressure.append(Pr)
            if "SWAT" in output_variables:
                sw = round_array_to_4dp(clip_and_convert_to_float32(sw))
                Swater.append(sw)
            if "SGAS" in output_variables:
                sg = round_array_to_4dp(clip_and_convert_to_float32(sg))
                Sgas.append(sg)
            if "SOIL" in output_variables:
                so = round_array_to_4dp(clip_and_convert_to_float32(so))
                Soil.append(so)

            tt = round_array_to_4dp(clip_and_convert_to_float32(Time))
            QG = round_array_to_4dp(clip_and_convert_to_float32(QG))
            QW = round_array_to_4dp(clip_and_convert_to_float32(QW))
            QO = round_array_to_4dp(clip_and_convert_to_float32(QO))

            if "FAULT" in input_variables:
                flt = round_array_to_4dp(clip_and_convert_to_float32(flt))
                Faulta.append(flt)

            Timea.append(tt)
            Qwa.append(QW)
            Qga.append(QG)
            logger.info(str(QG.shape))
            Qoa.append(QO)

            permeability[i, 0, :, :, :] = np.reshape(
                perm_ensemble[:, i], (nx, ny, nz), "F"
            )
            porosity[i, 0, :, :, :] = np.reshape(poro_ensemble[:, i], (nx, ny, nz), "F")
            actnumm[i, 0, :, :, :] = np.reshape(effective, (nx, ny, nz), "F")

            # Clean up variables
            del Pr, sw, sg, tt, QG, QW, QO
            if "FAULT" in input_variables:
                del flt
            # gc.collect

        # Stack the lists to create final arrays
        if "PRESSURE" in output_variables:
            pressure = np.stack(pressure, axis=0)
        if "SGAS" in output_variables:
            Sgas = np.stack(Sgas, axis=0)
        if "SOIL" in output_variables:
            Soil = np.stack(Soil, axis=0)
        if "SWAT" in output_variables:
            Swater = np.stack(Swater, axis=0)
        if "FAULT" in input_variables:
            Faulta = np.stack(Faulta, axis=0)[:, None, :, :, :]
        Timea = np.stack(Timea, axis=0)
        Qga = np.stack(Qga, axis=0)
        Qoa = np.stack(Qoa, axis=0)
        Qwa = np.stack(Qwa, axis=0)

        if "PRESSURE" in output_variables and pressure is not None:
            pressure = np.asarray(pressure)
            pressure[pressure <= 0] = 0

        ini_pressure = cfg.custom.PROPS.P1 * np.ones(
            (N, 1, nx, ny, nz), dtype=np.float32
        )
        ini_sat = cfg.custom.PROPS.S1 * np.ones((N, 1, nx, ny, nz), dtype=np.float32)

        maxP1 = np.max(ini_pressure)

        compdat_data = read_compdats(filename, well_names)
        inn_fcn, out_fcn = Get_data_FFNN(
            oldfolder,
            N,
            pressure,
            Sgas,
            Swater,
            Soil,
            permeability,
            Timea,
            steppi,
            steppi_indices,
            N_pr,
            producers,
            compdat_data,
            filenameui,
            well_measurements,
            lenwels,
        )

        inn_fcn[np.isnan(inn_fcn)] = 0.0
        out_fcn[np.isnan(out_fcn)] = 0.0

        logger.info(str(out_fcn.shape))

        Qa = Qwa + Qga + Qoa

        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|                 DATA CURRATED                                   |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )

        target_min = 0.01
        target_max = 1.0

        if "PERM" in input_variables:
            permeability[np.isnan(permeability)] = 0.0
            permeability[np.isinf(permeability)] = 0.0

        Timea[np.isnan(Timea)] = 0.0
        if "PRESSURE" in output_variables:
            pressure[np.isnan(pressure)] = 0.0
            pressure[np.isinf(pressure)] = 0.0
        Qwa[np.isnan(Qwa)] = 0.0
        Qga[np.isnan(Qga)] = 0.0
        Qa[np.isnan(Qa)] = 0.0

        Timea[np.isinf(Timea)] = 0.0

        Qwa[np.isinf(Qwa)] = 0.0
        Qga[np.isinf(Qga)] = 0.0
        Qa[np.isinf(Qa)] = 0.0

        if "PERM" in input_variables:
            minK, maxK, permeabilityx = scale_command(
                permeability, target_min, target_max
            )  # Permeability
        minT, maxT, Timex = scale_command(Timea, target_min, target_max)  # Time

        if "PRESSURE" in output_variables:
            maxP = np.max(pressure)
            maxP = max(maxP1, maxP)

            # change this     5000
            presstemp = cfg.custom.PROPS.P1 * np.ones(
                (N, steppi, nx, ny, nz), dtype=np.float32
            )
            pressultimate = np.concatenate((presstemp, pressure), axis=0)

            meanPp, stdPp = safe_mean_std(pressultimate)
        meanSgp, stdSgp = safe_mean_std(Sgas)
        meanSop, stdSop = safe_mean_std(Soil)
        meanSwp, stdSwp = safe_mean_std(Swater)

        # if dist.rank ==0:

        logger.info(
            "-----------------------------------------------------------------------"
        )
        logger.info(f"Pressure - Mean: {meanPp:.4f}, Standard Deviation: {stdPp:.4f}")
        logger.info(f"Sgas - Mean: {meanSgp:.4f}, Standard Deviation: {stdSgp:.4f}")
        logger.info(f"Soil - Mean: {meanSop:.4f}, Standard Deviation: {stdSop:.4f}")
        logger.info(f"Swater - Mean: {meanSwp:.4f}, Standard Deviation: {stdSwp:.4f}")
        logger.info(
            "-----------------------------------------------------------------------"
        )

        if "PRESSURE" in output_variables:
            minP, maxP, pressurex = scale_command_pressure(pressure, maxP)  # pressure

        minQw, maxQw, Qwx = scale_command(Qwa, target_min, target_max)  # Qw
        minQg, maxQg, Qgx = scale_command(Qga, target_min, target_max)  # Qg
        minQ, maxQ, Qx = scale_command(Qa, target_min, target_max)  # Q

        if "PERM" in input_variables:
            permeabilityx[np.isnan(permeabilityx)] = 0.0
            permeabilityx[np.isinf(permeabilityx)] = 0.0
        Timex[np.isnan(Timex)] = 0.0
        if "PRESSURE" in output_variables:
            pressurex[np.isnan(pressurex)] = 0.0
            pressurex[np.isinf(pressurex)] = 0.0

        Qwx[np.isnan(Qwx)] = 0.0
        Qgx[np.isnan(Qgx)] = 0.0
        Qx[np.isnan(Qx)] = 0.0

        Timex[np.isinf(Timex)] = 0.0

        Qwx[np.isinf(Qwx)] = 0.0
        Qgx[np.isinf(Qgx)] = 0.0
        Qx[np.isinf(Qx)] = 0.0

        if "PINI" in input_variables:
            ini_pressure[np.isnan(ini_pressure)] = 0.0
            ini_pressurex = ini_pressure / maxP

            ini_pressurex = clip_and_convert_to_float32(ini_pressurex)

            ini_pressurex[np.isnan(ini_pressurex)] = 0.0
            ini_pressurex[np.isinf(ini_pressurex)] = 0.0

        if "PORO" in input_variables:
            porosity[np.isnan(porosity)] = 0.0
            porosity[np.isinf(porosity)] = 0.0

        if "FAULT" in input_variables:
            Faulta[np.isnan(Faulta)] = 0.0
            Faulta[np.isinf(Faulta)] = 0.0

        if "SWAT" in output_variables:
            Swater[np.isnan(Swater)] = 0.0
            Swater[np.isinf(Swater)] = 0.0

        if "SOIL" in output_variables:
            Soil[np.isnan(Soil)] = 0.0
            Soil[np.isinf(Soil)] = 0.0

        if "SGAS" in output_variables:
            Sgas[np.isnan(Sgas)] = 0.0
            Sgas[np.isinf(Sgas)] = 0.0

        actnumm[np.isnan(actnumm)] = 0.0

        if "SINI" in input_variables:
            ini_sat[np.isnan(ini_sat)] = 0.0
            ini_sat[np.isinf(ini_sat)] = 0.0

        actnumm[np.isinf(actnumm)] = 0.0

        # Initialize the dictionary dynamically based on input and output variables
        X_data1 = {}

        # Add input variables if they exist
        if "PERM" in input_variables:
            X_data1["permeability"] = permeabilityx
        if "PORO" in input_variables:
            X_data1["porosity"] = porosity
        if "FAULT" in input_variables:
            X_data1["Fault"] = Faulta
        if "PINI" in input_variables:
            X_data1["Pini"] = ini_pressurex
        if "SINI" in input_variables:
            X_data1["Sini"] = ini_sat

        # Add output variables if they exist
        if "PRESSURE" in output_variables:
            X_data1["Pressure"] = pressurex
        if "SWAT" in output_variables:
            X_data1["Water_saturation"] = Swater
        if "SGAS" in output_variables:
            X_data1["Gas_saturation"] = Sgas
        if "SOIL" in output_variables:
            # Soil =
            X_data1["Oil_saturation"] = Soil

        # Add other necessary variables
        X_data1["Time"] = Timea
        X_data1["actnum"] = actnumm
        X_data1["Qw"] = Qwa
        X_data1["Qg"] = Qga
        X_data1["Q"] = Qa
        X_data1["Qo"] = Qoa

        X_data1 = {k: v for k, v in X_data1.items() if v is not None}

        X_data1 = clean_dict_arrays(X_data1)

        del permeabilityx, porosity, pressurex, Swater, permeability, Timex
        # gc.collect

        del ini_pressure, Timea, pressure, ini_sat, Sgas, ini_pressurex, Qwx, Qgx, Qx
        # gc.collect

        meanX, stdX = safe_mean_std(inn_fcn)
        meanY, stdY = safe_mean_std(out_fcn)

        min_inn_fcn, max_inn_fcn, inn_fcnx = scale_command(
            inn_fcn, target_min, target_max
        )
        min_out_fcn, max_out_fcn, out_fcnx = scale_command(
            out_fcn, target_min, target_max
        )

        inn_fcnx2, max_inn_fcn2, min_inn_fcn2 = scale_commandSin(inn_fcn, N_pr)
        out_fcnx2, max_out_fcn2, min_out_fcn2 = scale_commandS(out_fcn, lenwels, N_pr)

        inn_fcnx = clip_and_convert_to_float32(inn_fcnx)
        out_fcnx = clip_and_convert_to_float32(out_fcnx)

        inn_fcnx2 = clip_and_convert_to_float32(inn_fcnx2)
        out_fcnx2 = clip_and_convert_to_float32(out_fcnx2)

        X_data2 = {"X": inn_fcnx, "Y": out_fcnx, "X2": inn_fcnx2, "Y2": out_fcnx2}
        for key in X_data2.keys():
            X_data2[key][np.isnan(X_data2[key])] = 0.0  # Convert NaN to 0
            X_data2[key][np.isinf(X_data2[key])] = 0.0  # Convert infinity to 0

        del inn_fcnx, inn_fcn, out_fcnx, out_fcn, inn_fcnx2, out_fcnx2

        with gzip.open(
            to_absolute_path("../data/data_train_peaceman.pkl.gz"), "wb"
        ) as f3:
            pickle.dump(X_data2, f3)

        with gzip.open(
            to_absolute_path("../data/data_test_peaceman.pkl.gz"), "wb"
        ) as f4:
            pickle.dump(X_data2, f4)

        with gzip.open(to_absolute_path("../data/data_train.pkl.gz"), "wb") as f5:
            pickle.dump(X_data1, f5)

        with gzip.open(to_absolute_path("../data/data_test.pkl.gz"), "wb") as f6:
            pickle.dump(X_data1, f6)

        sio.savemat(
            to_absolute_path("../data/conversions.mat"),
            {
                "minK": minK,
                "maxK": maxK,
                "minT": minT,
                "maxT": maxT,
                "minP": minP,
                "maxP": maxP,
                "minQW": minQw,
                "maxQW": maxQw,
                "minQg": minQg,
                "maxQg": maxQg,
                "minQ": minQ,
                "maxQ": maxQ,
                "min_inn_fcn": min_inn_fcn,
                "max_inn_fcn": max_inn_fcn,
                "min_out_fcn": min_out_fcn,
                "max_out_fcn": max_out_fcn,
                "min_inn_fcn2": min_inn_fcn2,
                "max_inn_fcn2": max_inn_fcn2,
                "min_out_fcn2": min_out_fcn2,
                "max_out_fcn2": max_out_fcn2,
                "steppi": steppi,
                "steppi_indices": steppi_indices,
                "N_ens": N_ens,
                "N_pr": N_pr,
                "lenwels": lenwels,
                "effective": actnumm,
                "meanPp": meanPp,
                "stdPp": stdPp,
                "meanSgp": meanSgp,
                "stdSgp": stdSgp,
                "meanSop": meanSop,
                "stdSop": stdSop,
                "meanSwp": meanSwp,
                "stdSwp": stdSwp,
                "train_interest": cfg.custom.train_interest,
                "meanX": meanX,
                "stdX": stdX,
                "meanY": meanY,
                "stdY": stdY,
            },
            do_compression=True,
        )

        logger.info(
            "|-----------------------------------------------------------------|"
        )
        logger.info(
            "|      DATA SAVED & REMOVE FOLDERS USED FOR THE RUN               |"
        )
        logger.info(
            "|-----------------------------------------------------------------|"
        )

        for jj in range(N_ens):
            folderr = to_absolute_path("../RUNS/Realisation" + str(jj))
            rmtree(folderr)
        rmtree(to_absolute_path("../RUNS"))


if __name__ == "__main__":
    main()
