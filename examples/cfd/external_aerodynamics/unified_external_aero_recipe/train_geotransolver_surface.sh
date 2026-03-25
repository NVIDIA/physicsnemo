#!/bin/bash
#SBATCH --account=coreai_modulus_cae
#SBATCH --job-name=geotransolver_surface-drivaer_ml
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=4:00:00
#SBATCH --output=geotransolver_surface-drivaer_ml_%j.out
#SBATCH --error=geotransolver_surface-drivaer_ml_%j.err
#SBATCH --partition=batch

# Paths
export USER_LUSTRE=/lustre/fsw/portfolios/coreai/users/coreya
export GROUP_LUSTRE=/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae
export HOME=${USER_LUSTRE}

# Container setup
CONTAINER_IMAGE=$GROUP_LUSTRE/containers/pytorch26.01-py3.sqsh
CONTAINER_MOUNTS="$USER_LUSTRE:/user_data/,$GROUP_LUSTRE:/group_data,$HOME:/root/,/lustre:/lustre,/tmp:/tmp"

# Virtual environment path
VENV_PATH="$USER_LUSTRE/venvs/geotransolver2/"

WORKDIR="$USER_LUSTRE/workdir/physicsnemo/examples/cfd/external_aerodynamics/unified_external_aero_recipe/"

# Hydra (conf/train_surface.yaml)
TRAIN_SCRIPT="src/train.py --config-name train_surface"

AUGMENT=false
PRECISION="bfloat16"
SAMPLING_RESOLUTION=200000

# Include each data source (false => Hydra ~data.<name> so that source is dropped).
INCLUDE_DRIVAER_ML=false
INCLUDE_SHIFT_SUV_ESTATE=true
INCLUDE_SHIFT_SUV_FASTBACK=true

# run_id: geotransolver/surface/<datasets>_<aug|noaug>_<precision>_<sampling_resolution>
RUN_PARTS=()
[[ "${INCLUDE_DRIVAER_ML}" == "true" ]] && RUN_PARTS+=(drivaer_ml)
[[ "${INCLUDE_SHIFT_SUV_ESTATE}" == "true" ]] && RUN_PARTS+=(shift_suv_estate)
[[ "${INCLUDE_SHIFT_SUV_FASTBACK}" == "true" ]] && RUN_PARTS+=(shift_suv_fastback)

DATASET_SLUG=$(IFS='_'; echo "${RUN_PARTS[*]}")
[[ -n "${DATASET_SLUG}" ]] || DATASET_SLUG="none"

AUG_TAG="noaug"
[[ "${AUGMENT}" == "true" ]] && AUG_TAG="aug"

RUN_ID="geotransolver/surface/${DATASET_SLUG}_${AUG_TAG}_${PRECISION}_${SAMPLING_RESOLUTION}"

EXTRA_HYDRA_OVERRIDES=""

OVERRIDES="run_id=${RUN_ID} augment=${AUGMENT} precision=${PRECISION} dataset.sampling_resolution=${SAMPLING_RESOLUTION} "
[[ "${INCLUDE_DRIVAER_ML}" == "true" ]] || OVERRIDES+='~data.drivaer_ml '
[[ "${INCLUDE_SHIFT_SUV_ESTATE}" == "true" ]] || OVERRIDES+='~data.shift_suv_estate '
[[ "${INCLUDE_SHIFT_SUV_FASTBACK}" == "true" ]] || OVERRIDES+='~data.shift_suv_fastback '
OVERRIDES+="${EXTRA_HYDRA_OVERRIDES}"

echo "Overrides: ${OVERRIDES}"

# Launch the job with container
# As far as I know, environment variables are evaluated *before* the container is launched.
# So use the right paths for the right space!

export PATH="/cm/local/apps/slurm/current/bin:${PATH}"

srun --ntasks-per-node=4 \
     --container-image=${CONTAINER_IMAGE} \
     --container-mounts ${CONTAINER_MOUNTS} \
     bash -c "

        # Set up virtual environment
        source ${VENV_PATH}/bin/activate

        # This is where I have the training script in the container:
        cd ${WORKDIR}

        # Run the training script with overrides
        python ${TRAIN_SCRIPT} ${OVERRIDES}
     "
