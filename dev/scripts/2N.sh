#!/bin/bash
#SBATCH -J job_name            # set job name
#SBATCH -N 2                   # number of nodes
#SBATCH --ntasks-per-node=1    # number of tasks per node
#SBATCH -p partition           # need partition name
#SBATCH -o %x.%j.out           # output log
#SBATCH -e %x.%j.err           # error log
#SBATCH --gpus-per-node=1      # number of GPU per node
# job runtime limit
##SBATCH -t 01:00:00

SQSH=/path/to/physicsnemo_25.03.sqsh  
CNAME=physicsnemo_25_03
WORKDIR=/workspace
HOST_WORKDIR=$PWD

set -euo pipefail

# 用srun讓每個節點做同樣的事
srun --label bash -lc "
  set -euo pipefail
  if ! enroot list | grep -q '^${CNAME}\$'; then
    enroot create -n '${CNAME}' '${SQSH}'
  fi
  enroot start \
    --mount '${HOST_WORKDIR}:${WORKDIR}' \
    --env PIP_CACHE_DIR=/tmp/pipcache \
    --env REPO_URL=https://github.com/runberry/physicsnemo.git \
    --env REPO_DIR=${WORKDIR}/physicsnemo \
    --env BRANCH=exp/tp1 \
    '${CNAME}' bash -lc 'cd ${WORKDIR} && chmod +x hpl.sh && ./hpl.sh'
"

# 如果有 Pyxis srun可寫
: <<'COMMENT'
srun --label \
     --mpi=pmix \
     --export=ALL,PIP_CACHE_DIR=/tmp/pipcache,REPO_URL=https://github.com/runberry/physicsnemo.git,REPO_DIR=/workspace/physicsnemo,BRANCH=exp/tp1 \
     --container-image=$SQSH \
     --container-mounts=$PWD:/workspace \
     --container-workdir=/workspace \
     bash -lc "chmod +x hpl.sh && ./hpl.sh"
COMMENT