#!/bin/bash
#SBATCH --account=coreai_modulus_cae
#SBATCH --job-name=transolver_volume-drivaer_ml
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=4:00:00
#SBATCH --output=transolver_volume-drivaer_ml_%A.out
#SBATCH --error=transolver_volume-drivaer_ml_%A.err
#SBATCH --partition=batch

# Paths
export USER_LUSTRE=/lustre/fsw/portfolios/coreai/users/coreya
export GROUP_LUSTRE=/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae
export HOME=${USER_LUSTRE}

# Container setup
CONTAINER_IMAGE=$GROUP_LUSTRE/containers/pytorch26.01-py3.sqsh
CONTAINER_MOUNTS="$USER_LUSTRE:/user_data/,$GROUP_LUSTRE:/group_data,$HOME:/root/,/lustre:/lustre,/tmp:/tmp"

# Virtual environment path
VENV_PATH="$USER_LUSTRE/venvs/shard_tensor_benchmarks/"

WORKDIR="$USER_LUSTRE/workdir/shard_tensor_benchmarks/examples/cfd/external_aerodynamics/transformer_models/"

# Hydra (src/conf/transolver_volume.yaml)
TRAIN_SCRIPT="src/train.py --config-name transolver_volume"

PRECISION="bfloat16"
SAMPLING_RESOLUTION_PER_GPU=200000

# DrivAer ML Zarr dataset paths (set these to your Zarr directories):
ZARR_TRAIN_PATH="/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/drivaer_aws/domino/train/"
ZARR_VAL_PATH="/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/drivaer_aws/domino/val/"

# Domain parallelism: number of GPUs collaborating on one sample.
# world_size must be divisible by this. Set to 1 to disable.
DOMAIN_PARALLEL_SIZE=8

# Total resolution scales with domain parallel size so each GPU
# keeps SAMPLING_RESOLUTION_PER_GPU points.
TOTAL_RESOLUTION=$((SAMPLING_RESOLUTION_PER_GPU * DOMAIN_PARALLEL_SIZE))

NODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=${SLURM_NTASKS_PER_NODE:-1}
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

RUN_ID="transolver/volume/drivaer_ml_${PRECISION}_res${TOTAL_RESOLUTION}_${SAMPLING_RESOLUTION_PER_GPU}ppg_dp${DOMAIN_PARALLEL_SIZE}_${TOTAL_GPUS}gpu"

EXTRA_HYDRA_OVERRIDES=""

OVERRIDES="run_id=${RUN_ID} "
OVERRIDES+="precision=${PRECISION} "
OVERRIDES+="data.resolution=${TOTAL_RESOLUTION} "
OVERRIDES+="data.train.data_path=${ZARR_TRAIN_PATH} "
OVERRIDES+="data.val.data_path=${ZARR_VAL_PATH} "
OVERRIDES+="domain_parallel_size=${DOMAIN_PARALLEL_SIZE} "
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
