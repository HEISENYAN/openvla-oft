python vla-scripts/deploy.py \
  --pretrained_checkpoint /home/congcong/yanzhengyang/aloha_adjust_bottle_100000 \
  --use_l1_regression True \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --center_crop True \
  --unnorm_key aloha_adjust_bottle_tfds