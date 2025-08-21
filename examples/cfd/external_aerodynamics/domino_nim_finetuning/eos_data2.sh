#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J coreai_modulus_cae-modulus:data_processing_npy_baseline
#SBATCH -t 01:58:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --dependency=singleton
#SBATCH -o ./sbatch_logs/multi/%x_%j.out
#SBATCH -e ./sbatch_logs/multi/%x_%j.err


#--mail-type=FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT_90,END
#set -euxo pipefail

# code directory (change this to your own code directory)
readonly _code_root="/lustre/rranade/modulus_dev/modulus_demo/modulus_rishi/modulus/"

# mount the data and code directories
readonly _cont_mounts="/lustre/fsw/coreai_modulus_cae/:/lustre/"

# pull image from NGC registry

# readonly _cont_image='/lustre/fsw/coreai_modulus_cae/rranade/cont_modulus.sqsh'

readonly _cont_image='/lustre/fsw/coreai_modulus_cae/coreya/physicsnemo25.06.sqsh'

export NPROC_PER_NODE=8
export TOTAL_GPU=$(($SLURM_JOB_NUM_NODES * $NPROC_PER_NODE))
# train the model
# RUN_CMD="cd /code/wistron/computex && torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=$(hostname) --rdzv_id=${SLURM_JOB_ID} --nnodes ${SLURM_JOB_NUM_NODES} 
#RUN_CMD="cd /code/wistron/computex && torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=$(hostname) --rdzv_id=${SLURM_JOB_ID} --nnodes ${SLURM_JOB_NUM_NODES} train.py >> log.train.1"
# RUN_CMD="torchrun --nnodes 1 --nproc_per_node 8 src/train.py exp_tag=5 model.model_type=combined model.volume_points_sample=8192 model.surface_points_sample=8192 model.geom_points_sample=300000 model.geometry_rep.geo_processor.base_filters=8 model.aggregation_model.base_layer=256 data.input_dir=/lustre/rranade/modulus_dev/data/combined_data_test/ data.input_dir_val=/lustre/rranade/modulus_dev/data/combined_data_test_val/  > output_train_5_1.log"
# RUN_CMD="python -u src/train.py exp_tag=20 model.model_type=surface model.volume_points_sample=36144 model.surface_points_sample=36144 model.geometry_local.volume_radii=[0.05,0.25,1.0] model.geometry_local.surface_radii=[0.05,0.25,1.0] model.geometry_local.volume_neighbors_in_radius=[64,128,256] model.geometry_local.surface_neighbors_in_radius=[64,128,256] model.nn_basis_functions.fourier_features=true model.geometry_encoding_type=stl model.use_sdf_in_basis_func=true model.geometry_rep.geo_conv.volume_radii=[0.05,0.1,0.5,1.0,2.5,5.0,10.0]"
# RUN_CMD="python -u src/test.py exp_tag=23 model.model_type=surface eval.save_path=/lustre/rranade/modulus_dev/data/DS_Crash_Test_3 eval.checkpoint_name=DoMINO.0.499.pt"
# RUN_CMD="python -u src/test.py exp_tag=4 model.model_type=surface eval.save_path=/lustre/rranade/modulus_dev/data/DS_Crash_Test_mesh_filtered_relu4 eval.checkpoint_name=DoMINO.0.393.pt model.activation=relu"
# nohup torchrun --nnodes 1 --nproc_per_node 8 src/test.py exp_tag=60 model.model_type=surface eval.save_path=/lustre/rranade/modulus_dev/data/DS_Crash_Test_mesh eval.checkpoint_name=DoMINO.0.426.pt model.geometry_rep.geo_processor.self_attention=false model.num_neighbors_surface=1 > output_train_1_60.log &
# nohup torchrun --nnodes 1 --nproc_per_node 8 src/train.py model.surface_points_sample=80000 exp_tag=1 model.volume_points_sample=80000 > output_train_1_1.log &
# nohup torchrun --nnodes 1 --nproc_per_node 8 src/train_physics.py exp_tag=64 model.model_type=combined model.volume_points_sample=10244 model.surface_points_sample=10244 model.use_sdf_in_basis_func=false data.input_dir=/lustre/rranade/modulus_dev/data/aws_data_all/ data.input_dir_val=/lustre/rranade/modulus_dev/data/aws_data_all_val/  > output_train_47_1.log &
RUN_CMD="python -u src/process_data_baseline.py"

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