#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J coreai_modulus_cae-modulus:train_finetune
#SBATCH -t 02:58:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --dependency=singleton
#SBATCH -o ./sbatch_logs_finetune/multi/%x_%j.out
#SBATCH -e ./sbatch_logs_finetune/multi/%x_%j.err


#--mail-type=FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT_90,END
#set -euxo pipefail

# code directory (change this to your own code directory)
readonly _code_root="/lustre/rranade/modulus_dev/modulus_demo/modulus_rishi/modulus/"

# mount the data and code directories
readonly _cont_mounts="/lustre/fsw/coreai_modulus_cae/:/lustre/"

# pull image from NGC registry

readonly _cont_image='/lustre/fsw/coreai_modulus_cae/rranade/physicsnemo25.06.sqsh'
export NPROC_PER_NODE=8
export TOTAL_GPU=$(($SLURM_JOB_NUM_NODES * $NPROC_PER_NODE))
# train the model
RUN_CMD="python -u src/train.py exp_tag=30 model.model_type=combined model.surface_points_sample=54000 model.volume_points_sample=54000 model.loss_function.loss_type=mse model.activation=relu model.surface_sampling_algorithm=solution_weighted"
# nohup torchrun --nnodes 1 --nproc_per_node 8 src/train.py exp_tag=1 model.model_type=combined model.surface_points_sample=54000 model.volume_points_sample=54000 model.loss_function.loss_type=mse model.activation=relu > output_train_1_finetune.log &

echo "Running on hosts: $(echo $(scontrol show hostname))"
srun -A coreai_modulus_cae  \
     --container-image="${_cont_image}" \
     --container-mounts="${_cont_mounts}" \
     --ntasks-per-node=8 \
     bash -c "
     ldconfig
     set -x
     export WORLD_SIZE=${TOTAL_GPU}
     export WORLD_RANK=\${PMIX_RANK}
     export HDF5_USE_FILE_LOCKING=FALSE
     export CUDNN_V8_API_ENABLED=1
     export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
     unset TORCH_DISTRIBUTED_DEBUG
     cd /lustre/rranade/modulus_dev/modulus/physicsnemo
     pip install torchinfo
     apt install rsync -y  
     rsync -av physicsnemo/* /usr/local/lib/python3.12/dist-packages/physicsnemo
     cd /lustre/rranade/modulus_dev/modulus/physicsnemo/examples/cfd/external_aerodynamics/domino_nim_finetuning
     ${RUN_CMD}"