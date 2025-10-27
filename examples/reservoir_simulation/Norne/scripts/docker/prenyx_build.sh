#!/bin/bash
#############################################################
# Author: Clement Etienam (cetienam@nvidia.com)
#############################################################


srun \
    -p b200-a01r \
    -N 1 \
    -A coreai_devtech_all -J coreai_devtech_all-total_rft_2025:dev \
    --ntasks-per-node=8 \
    --comment="Reservoir modelling with PhysicsNemo" \
    -t 04:00:00 \
    --container-image="gitlab-master.nvidia.com/globalenergyteam/customers/total/total_rfp_reservoir:athena_x86" \
    --container-mounts=/lustre/fsw/coreai_devtech_all/cetienam/physicsnemo:/workspace/project \
    --container-workdir=/workspace/project \
    --pty /bin/bash
