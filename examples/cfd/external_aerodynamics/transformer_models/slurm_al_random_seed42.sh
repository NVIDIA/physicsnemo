#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J drivaer-al-random-s42
#SBATCH -t 03:45:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --dependency=singleton
#SBATCH -o ./slurm_logs/%x_%j.out
#SBATCH -e ./slurm_logs/%x_%j.err

readonly _lustre="/lustre/fsw/coreai_modulus_cae/ktangsali"
readonly _data_root="/lustre/fsw/coreai_modulus_cae"

readonly _cont_mounts="${_lustre}:/workspace:rw,${_data_root}/:/data:rw"
readonly _cont_image="nvcr.io/nvidia/physicsnemo/physicsnemo:26.03"

RUN_CMD="
    nvidia-smi && \
    cd /workspace/active_learning/pr-prep/modulus/ && \
    pip install --break-system-packages . && \
    pip install --break-system-packages gpytorch && \
    cd examples/cfd/external_aerodynamics/transformer_models && \
    pip install --break-system-packages -r requirements.txt && \
    mkdir -p slurm_logs && \
    torchrun \
        --nproc_per_node=8 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=\$(hostname) \
        --rdzv_id=\${SLURM_JOB_ID} \
        --nnodes \${SLURM_JOB_NUM_NODES} \
        src/active_learning/run_al.py \
        --config-name=al_config \
        ++initial_checkpoint=runs/geotransolver/surface/gp_head_experiment/checkpoints_combined \
        ++manifest_dir=src/active_learning/manifests \
        ++data.train.data_path=/data/datasets/drivaerstar/surface_files_zarr/class_F/train \
        ++data.val.data_path=/data/datasets/drivaerstar/surface_files_zarr/class_F/val \
        ++data.resolution=51200 \
        ++data.geometry_sampling=51200 \
        ++data.return_mesh_features=true \
        ++acquisition=random \
        ++random_seed=42 \
        ++run_id=geotransolver/surface/al_random_seed_42
"

echo "Running on hosts: $(echo $(scontrol show hostname))"
srun -A coreai_modulus_cae \
     --container-image="${_cont_image}" \
     --container-mounts="${_cont_mounts}" \
     --ntasks-per-node=1 \
     bash -c "${RUN_CMD}"
