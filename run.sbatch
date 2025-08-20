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
#bash vla-server.bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /home/congcong/tensorflow_datasets/ \
  --dataset_name aloha_game \
  --run_root_dir /home/congcong/yanzhengyang/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --batch_size 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100005 \
  --use_val_set False \
  --val_freq 10000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "heisen0928-the-hong-kong-polytechnic-university" \
  --wandb_project "aloha_game" \
  --run_id_note parallel_dec--25_acts_chunk--continuous_acts--L1_regression--3rd_person_img--left_right_wrist_imgs--proprio_state