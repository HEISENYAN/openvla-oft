wm=/home/agilex/checkpoints/09/23/aloha_wm_128_grpo_b64/global_step_15
base=/home/agilex/checkpoints/09/14/openvla-7b+aloha_game+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--aloha_game_100_09_14--100000_chkpt
python vla-scripts/deploy.py \
  --pretrained_checkpoint $base \
  --use_l1_regression False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --center_crop True \
  --use_proprio False \
  --unnorm_key aloha_game
  #/home/agilex/checkpoints/09/04/aloha_game_100_fix/openvla-7b+aloha_game+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--aloha_game_100_8_fix--100000_chkpt \
    #--pretrained_checkpoint /home/agilex/checkpoints/openvla-7b+aloha_game+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--aloha_game_300_8--100000_chkpt \
    # home/agilex/checkpoints/aloha_game_100/09/03/aloha_game_100/openvla-7b+aloha_game+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--aloha_game_100_8--100000_chkpt
    #--pretrained_checkpoint /home/agilex/checkpoints/aloha_game_100/aloha_game_100/openvla-7b+aloha_game+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--aloha_game_100_8--100000_chkpt \