#!/bin/bash
# Job name:
#SBATCH --job-name=evaluate_performance
#
# Account:
#SBATCH --account=nvr_earth2_e2
#SBATCH --output=out/R-%A.%a.out
#SBATCH --error=out/R-%A.%a.err
#
# Partition:
#SBATCH --partition=grizzly,grizzly2,polar,polar2,polar3
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors:
#SBATCH --cpus-per-task=4
#
#Number of GPUs
#SBATCH --gpus=8
#
# Wall clock limit:
#SBATCH --time=4:00:00
#
#SBATCH --array=3,5,10,15,20,25,30,35,40,43,45,46,47,48,49,50
#
## Command(s) to run:
srun --container-image=nvcr.io\#nvidia/pytorch:23.12-py3 python /root/daobs/sda/experiments/corrdiff/evaluation/evaluate_performance_obs_sweep_parallel.py 

