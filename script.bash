1. roscore 

roslaunch astra_camera multi_camera.launch

rqt_image_view

rostopic list

rostopic echo /camera_f/color/image_raw

rostopic echo /puppet/joint_left

#inference: 
roslaunch piper start_ms_piper.launch mode:=1 auto_enalble:=true

#data collection
roslaunch piper start_ms_piper.launch mode:=0 auto_enalble:=false

python collect_data_master_gripper.py --task_name test --dataset_dir ./data --episode_idx 0 --max_timesteps 100 --frame_rate 10

python replay_data.py --dataset_dir ./data --task_name test --episode_idx 0

python collect_data.py --task_name original --dataset_dir ./data --episode_idx 0 --max_timesteps 100 --frame_rate 10

python replay_data.py --dataset_dir /home/agilex/adjust_bottle --task_name adjust_bottle --episode_idx 33





