#!/bin/bash

#--mail-type=FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT_90,END
#set -euxo pipefail

export NPROC_PER_NODE=8
export TOTAL_GPU=$(($SLURM_JOB_NUM_NODES * $NPROC_PER_NODE))

# train the model
RUN_CMD="python -u train.py exp_tag=10"
#RUN_CMD="python process_data.py"

echo "Running on hosts: $(echo $(scontrol show hostname))"
ldconfig
set -x
export WORLD_SIZE=${TOTAL_GPU}
export WORLD_RANK=\${PMIX_RANK}
export HDF5_USE_FILE_LOCKING=FALSE
export CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
unset TORCH_DISTRIBUTED_DEBUG
cd /lustre/snidhan/physicsnemo-work/physicsnemo/
pip install warp-lang
pip install torchinfo
pip install timm==1.0.14
rsync -av /lustre/snidhan/physicsnemo-work/physicsnemo/physicsnemo/* /usr/local/lib/python3.10/dist-packages/physicsnemo
cd /lustre/snidhan/physicsnemo-work/physicsnemo/examples/cfd/external_aerodynamics/domino/src/
${RUN_CMD}