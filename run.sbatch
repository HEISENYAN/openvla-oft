#!/bin/bash


#SBATCH --job-name=yanzhengyang
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --output=slurm.out


nvidia-smi

module purge
#module load slurm
#module avail cuda
module load slurm
#module load cuda12.2/toolkit/12.2.2
#module avail tensorrt
source /home/congcong/miniconda3/bin/activate openvla-oft

#nvidia-smi
#nvcc -V
export PYTHONUNBUFFERED=1

#python -u pipeline.py
#python -u pipeline-scenario-2.py #world_model_config.py
bash vla-server.bash