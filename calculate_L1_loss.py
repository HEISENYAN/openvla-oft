#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
#Robot Control Import
import torch
import numpy as np
import os
import pickle
import argparse
from einops import rearrange

import collections
from collections import deque

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import time
import threading
import math
import threading
import tensorflow_datasets as tfds

import sys
sys.path.append("./")
sys.path.append("./openvla-oft/")
#Openvla-oft Import
import logging
import socket
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import tqdm

from experiments.robot.openvla_utils import (
    get_action_from_server,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    set_seed_everywhere,
)
task_config = {'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']}

# import l1 loss calculation
from prismatic.training.train_utils import compute_actions_l1_loss

#main config
@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 50                    # Number of actions to execute open-loop before requerying policy

    use_vla_server: bool = True                      # Whether to query remote VLA server for actions
    vla_server_url: Union[str, Path] = "127.0.0.1"            # Remote VLA server URL (set to 127.0.0.1 if on same machine)

    #################################################################################################################
    # ALOHA environment-specific parameters
    #################################################################################################################
    num_rollouts_planned: int = 50                   # Number of test rollouts
    max_steps: int = 300                           # Max number of steps per rollout
    use_relative_actions: bool = False               # Whether to use relative actions (delta joint angles)
    publish_rate: int = 100
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    policy_type: str = "openvla-oft"

    seed: int = 7                                    # Random Seed (for reproducibility)
    #################################################################################################################
    # Task Specifications
    #################################################################################################################
    task_description :str = "Grab the bottle with right or left gripper"
    # fmt: on

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None

# openvla functions
#########################################
def get_server_endpoint(cfg: GenerateConfig):
    """Get the server endpoint for remote inference."""
    ip_address = socket.gethostbyname(cfg.vla_server_url)
    return f"http://{ip_address}:8777/act"

def prepare_observation(img,left_wrist_img, right_wrist_img,qpos,resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    #img = get_aloha_image(obs)
    #left_wrist_img, right_wrist_img = get_aloha_wrist_images(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    left_wrist_img_resized = resize_image_for_policy(left_wrist_img, resize_size)
    right_wrist_img_resized = resize_image_for_policy(right_wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "left_wrist_image": left_wrist_img_resized,
        "right_wrist_image": right_wrist_img_resized,
        "state": qpos,
    }

    return observation, img_resized, left_wrist_img_resized, right_wrist_img_resized

def run_openvla_oft(
    cfg: GenerateConfig,
    task_description: str,
    server_endpoint: str,
    resize_size,
    ros_operator
):
    # Initialize action queue
    action_queue = deque() # maxlen=cfg.num_open_loop_steps

    # Setup
    t = 0
   
    
    episode_start_time = time.time()
    total_model_query_time = 0.0
    left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
    ros_operator.puppet_arm_publish_continuous(left0,right0)
    ds, info = tfds.load(
        'aloha_adjust_bottle_tfds:1.0.1',  # 数据集名称
        data_dir='/home/agilex/cobot_magic',  # 新目录路径
        split='train',  # 指定拆分，例如 'train' 或 'test'
        with_info=True  # 获取数据集元数据
    )
    #ds.shuffle(buffer_size=50)
    ds_iter = iter(ds.take(1))  # Take one sample
    sample = next(ds_iter)
    step_iter = iter(sample['steps'])
    while t < cfg.max_steps:
        obs_actions = []
        obs_img_fronts = []
        obs_img_lefts = []
        obs_img_rights = []
        obs_states = []

        
        for _ in range(25):
            try:
                step = next(step_iter)
            except:
                assert False

            obs_actions.append(step['action'].numpy())
            obs_img_fronts.append(step['observation']['image'].numpy())
            obs_img_lefts.append(step['observation']['left_wrist_image'].numpy())
            obs_img_rights.append(step['observation']['right_wrist_image'].numpy())
            obs_states.append(step['observation']['state'].numpy())

        observation, img_resized, left_wrist_resized, right_wrist_resized = prepare_observation(obs_img_fronts[0],obs_img_lefts[0],obs_img_rights[0],obs_states[0],resize_size)
        observation["instruction"] = task_description

        actions = get_action_from_server(observation, server_endpoint)
        #actions = obs_actions


        # predicted_actions = torch.from_numpy(np.array(actions))  
        # ground_truth_actions = torch.from_numpy(np.array(obs_actions)) 
        # actions = obs_actions
        #ground_truth_curr_action = ground_truth_actions[:, 0]
        #predicted_curr_action = predicted_actions[:, 0]
        #ground_truth_next_actions = ground_truth_actions[:, 1:]
        #predicted_next_actions = predicted_actions[:, 1:]
        #print(f"Step {t}, l1 curr_action_l1_loss = {curr_action_l1_loss}, l1 next_actions_l1_loss = {next_actions_l1_loss}")
        t += 25
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            for action in actions:
                left_action = action[:7]
                right_action = right0
                # ros_operator.puppet_arm_publish_continuous(left_action, right_action)
                ros_operator.puppet_arm_publish(left_action, right_action)
                rate.sleep() 
            break
        continue
    return 

        # Apply temporal ensembling to smooth the action chunk
        # Initialize action history if it doesn't exist
        # if not hasattr(get_action_from_server, 'action_history'):
        #     get_action_from_server.action_history = deque(maxlen=2)  # Keep last 5 action chunks
        
        # Add current actions to history
        # get_action_from_server.action_history.append(actions)
        
        # # Apply temporal smoothing if we have enough history
        # if len(get_action_from_server.action_history) >= 2:
        #     # Simple exponential moving average for temporal ensembling
        #     alpha = 0.7  # Smoothing factor (0.7 for current, 0.3 for history)
        #     smoothed_actions = []
            
        #     for i in range(len(actions)):
        #         action_timestep = []
        #         for j in range(len(actions[i])):
        #             # Weighted average: current action gets higher weight
        #             current_val = actions[i][j]
        #             # Get average of previous predictions for this timestep and joint
        #             prev_vals = [hist[i][j] for hist in list(get_action_from_server.action_history)[:-1]]
        #             if prev_vals:
        #                 prev_avg = np.mean(prev_vals)
        #                 # Apply exponential moving average
        #                 smoothed_val = alpha * current_val + (1 - alpha) * prev_avg
        #             else:
        #                 smoothed_val = current_val
        #             action_timestep.append(smoothed_val)
        #         smoothed_actions.append(action_timestep)
            
        #     actions = smoothed_actions
        #     print(f"Applied temporal ensembling with alpha={alpha}")
        

    #         actions = actions[: cfg.num_open_loop_steps]
    #         #total_model_query_time += time.time() - model_query_start_time
    #         action_queue.extend(actions)
    #         # actions = []`
            
    #         #Get action from queue
    #         rate = rospy.Rate(30)
    #         while len(action_queue) > 0 and not rospy.is_shutdown():
    #            action = action_queue.popleft()
    #            # new_action = np.linspace(last_action, action, 20)
    #            # last_action = action
    #            left_action = action[:7]
    #            right_action = action[7:14]
    #            # TODO
    #            # ros_operator.puppet_arm_publish_continuous(left_action, right_action)
    #            ros_operator.puppet_arm_publish(left_action, right_action)
    #            num += 1
    #            rate.sleep() 
    #            # rate.sleep()
    #         #    for act in new_action:
    #         #     left_action = act[:7]
    #         #     right_action = act [7:14]
    #         #     ros_operator.puppet_arm_publish(left_action, right_action)
    #         #     rate.sleep()
    #         #    num += 1
    #         # Calculate L1 loss between predicted actions and current state
    #         # while len(action_queue) > 0:
    #         #     # Get the first action from queue for L1 loss calculation
    #         #     predicted_action = action_queue.popleft()  # Use first action as prediction
    #         #     # Extract left and right arm actions
    #         #     predicted_left_action = predicted_action[:7]
    #         #     predicted_right_action = predicted_action[7:14]
                
    #         #     # Use current qpos as ground truth (current state)
    #         #     # This represents the "no change" baseline
    #         #     ground_truth_left = qpos[:7]   # First 7 joints for left arm
    #         #     ground_truth_right = qpos[7:14]  # Next 7 joints for right arm
                
    #         #     # Calculate L1 loss for left and right arms
    #         #     left_arm_l1_loss = np.mean(np.abs(np.array(predicted_left_action) - np.array(ground_truth_left)))
    #         #     right_arm_l1_loss = np.mean(np.abs(np.array(predicted_right_action) - np.array(ground_truth_right)))
                
    #         #     # Calculate total L1 loss
    #         #     total_l1_loss = (left_arm_l1_loss + right_arm_l1_loss) / 2.0
    #         #     l1_loss += total_l1_loss
    #         #     # Print L1 loss information
    #         #     print(f"Step {t}: L1 Loss - Left: {left_arm_l1_loss:.6f}, Right: {right_arm_l1_loss:.6f}, Total: {total_l1_loss:.6f}")
    #         #     print(f"Total L1 loss: {l1_loss:.6f}")
    #         #     num += 1
    #             # Alternative: Calculate L1 loss using the imported function if we have tokenized actions
    #             # Note: This would require converting actions to token IDs first
    #             # if 'action_tokenizer' in locals():
    #             #     # Convert continuous actions to token IDs (if needed)
    #             #     # predicted_tokens = action_tokenizer.encode_actions_to_token_ids(predicted_action)
    #             #     # ground_truth_tokens = action_tokenizer.encode_actions_to_token_ids(np.concatenate([ground_truth_left, ground_truth_right]))
    #             #     # l1_loss = compute_actions_l1_loss(action_tokenizer, predicted_tokens, ground_truth_tokens, mask)
    #             #     pass
    #         # t += 1
    #     # print(f"Average L1 loss: {l1_loss/num:.6f}")
    # except (KeyboardInterrupt, Exception) as e:
    #     print(e)

    # episode_end_time = time.time()

    # # Get success feedback from user
    # user_input = input("Success? Enter 'y' or 'n': ")
    # success = True if user_input.lower() == "y" else False

    # # Calculate episode statistics
    # episode_stats = {
    #     "success": success,
    #     "total_steps": t,
    #     "model_query_time": total_model_query_time,
    #     "episode_duration": episode_end_time - episode_start_time,
    # }

    # return (
    #     episode_stats,
    #     # replay_images,
    #     # replay_images_resized,
    #     # replay_images_left_wrist_resized,
    #     # replay_images_right_wrist_resized,
    # )
class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100
        rate = rospy.Rate(200)

        left_arm = None
        right_arm = None

        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)

        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def ctrl_callback(self, msg):
        self.ctrl_state_lock.acquire()
        self.ctrl_state = msg.data
        self.ctrl_state_lock.release()

    def get_ctrl_state(self):
        self.ctrl_state_lock.acquire()
        state = self.ctrl_state
        self.ctrl_state_lock.release()
        return state

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        #rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        #rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        #rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)


def get_arguments():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='aloha_mobile_dummy', required=False)
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000, required=False)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
    parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name', default='dataset_stats.pkl', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', default='ACT', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=8, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5, required=False)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=1e-4, required=False)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)", required=False)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features", required=False)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=True, required=False)

    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=14, required=False)
    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=1e-5, required=False)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
    parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2', default='l1', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer", required=False)
    parser.add_argument('--pre_norm', action='store_true', required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=100, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=0, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=32, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)
    
    args = parser.parse_args()
    return args




def eval_aloha(cfg: GenerateConfig, ros_operator) -> None:
    """Main function to evaluate a trained policy in a real-world ALOHA environment."""

    # Set random seed
    #set_seed_everywhere(cfg.seed)

    # Setup logging
    #log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Get ALOHA environment
    #env = get_aloha_env()

    # Get server endpoint for remote inference
    server_endpoint = get_server_endpoint(cfg)

    # Initialize task description
    task_description = cfg.task_description

    #Start Policy:
    if cfg.policy_type == "openvla-oft":
    #Run policy
        episode_stats, replay_images, replay_images_resized, replay_images_left_wrist, replay_images_right_wrist = (
            run_openvla_oft(cfg, task_description, server_endpoint, resize_size, ros_operator)
        )
    else:
        raise NotImplemented

def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    cfg = GenerateConfig
    #ros_operator = None
    eval_aloha(GenerateConfig,ros_operator)
    #pisode_stats, replay_images, replay_images_resized, replay_images_left_wrist, replay_images_right_wrist = (
    #        run_episode(cfg, task_description, server_endpoint, resize_size, log_file)
    #    )
    #model_inference(args, config, ros_operator, save_episode=True)


if __name__ == '__main__':
    main()
# python act/inference.py --ckpt_dir ~/train0314/