python vla-scripts/deploy.py \
  --pretrained_checkpoint /home/agilex/checkpoints/openvla-7b+aloha_game+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--aloha_game_300_8--50000_chkpt \
  --use_l1_regression False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --center_crop True \
  --use_proprio False \
  --unnorm_key aloha_game

    #--pretrained_checkpoint /home/agilex/checkpoints/openvla-7b+aloha_game+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--aloha_game_300_8--100000_chkpt \