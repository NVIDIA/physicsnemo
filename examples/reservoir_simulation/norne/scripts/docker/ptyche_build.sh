#!/bin/bash
#############################################################
# Author: Clement Etienam (cetienam@nvidia.com)
#############################################################

srun \
    -p 36x2-a01r \
    -N 1 \
    -A coreai_devtech_all \
    -J coreai_devtech_all-pangea-geos-rfp-2024:dev \
    --ntasks-per-node=4 \
    --comment="Interactive GEOSX container run" \
    -t 05:00:00 \
    --mpi=pmix \
    --container-image="gitlab-master.nvidia.com/globalenergyteam/customers/total/total_rfp_reservoir:athena_armm" \
    --container-mounts=/lustre/fsw/coreai_devtech_all/cetienam/physicsnemo_publish:/workspace/project \
    --container-mount-home \
    --container-workdir=/workspace/project \
    --no-container-remap-root \
    --pty bash


