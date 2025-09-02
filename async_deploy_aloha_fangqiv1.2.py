#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
# Robot Control Import
import torch
import numpy as np
import os
import pickle
import argparse
from einops import rearrange

import collections
from collections import deque, defaultdict

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

import sys
sys.path.append("./")
sys.path.append("./openvla-oft/")
# Openvla-oft Import
import logging
import socket
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import tqdm

import tty
import termios, select

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

# ===================== 全局：动作生产-消费（线程安全） =====================
from threading import Thread, Event, Lock

# 生产者线程控制
_action_prod_thread = None
_action_stop_event = Event()
_action_lock = Lock()

# 为每个 expected timestep 维护一个动作队列： expected_t -> deque[record]
# record: {"action": ..., "created_timestep": int, "expected_timestep": int, "latency": float, "created_time": float}
_action_queues = defaultdict(deque)

# t=0 action 0 1 2 3 4 5
# 


# “排期游标”：决定下一个 chunk 的 expected_t 起点（不断累加）
#_schedule_cursor = 0

# 全局创建计数器：严格递增，用于标识“最新创建”（与 expected_t 无关）
_created_counter = 0

# 新增：
_control_t = 0   # 主循环当前控制步（由主循环每拍刷新）

def _enqueue_chunk_to_expected_queues(chunk, latency, _schedule_cursor):
    global _created_counter, _control_t

    if chunk is None:
        return

    # 统一成 list
    if isinstance(chunk, np.ndarray):
        if chunk.ndim == 1:
            chunk = [chunk]
        else:
            chunk = [chunk[i] for i in range(chunk.shape[0])]
    elif not isinstance(chunk, (list, tuple)):
        chunk = [chunk]

    with _action_lock:
        start_expected_t = _schedule_cursor
        created_snapshot = _control_t     # ★ 关键：创建时主循环的控制步（墙钟语义）

        for i, a in enumerate(chunk):
            record = {
                "action": a,
                "created_timestep": created_snapshot,        # ★ 用快照，而不是创建序号
                "created_seq": _created_counter,             # 可选：保留一个严格递增的创建序号做调试
                "expected_timestep": start_expected_t + i,
                "latency": latency,
                "created_time": time.time(),
            }
            _action_queues[start_expected_t + i].append(record)
            _created_counter += 1

        _schedule_cursor += _control_t
        #_control_t += len(chunk)


def pick_latest_for_expected_t(expected_timestep):
    """
    若该 expected_timestep 已经有候选 action，返回“最新创建”的那一个（队尾）；否则 None。
    """
    with _action_lock:
        q = _action_queues.get(expected_timestep, None)
        if not q:
            return None
        # 最新创建在队尾
        latest = q[-1]
        # 可选：只保留最近K条，避免无限增长（打开以下两行示例）
        # while len(q) > 4:
        #     q.popleft()
        return latest

def clear_action_buffer():
    """episode 开始前清空所有队列与计数器。"""
    global _schedule_cursor, _created_counter
    with _action_lock:
        _action_queues.clear()
        _schedule_cursor = 0
        _created_counter = 0
        _control_t = 0        # 新增

# ===================== 配置 =====================
@dataclass
class GenerateConfig:
    # fmt: off
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"
    center_crop: bool = True
    num_open_loop_steps: int = 50
    use_vla_server: bool = True
    vla_server_url: Union[str, Path] = "127.0.0.1"

    #################################################################################################################
    # ALOHA environment-specific parameters
    #################################################################################################################
    num_rollouts_planned: int = 50
    max_steps: int = 1500
    use_relative_actions: bool = False
    publish_rate: int = 50

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None
    policy_type: str = "openvla-oft"
    seed: int = 7

    #################################################################################################################
    # Task Specifications
    #################################################################################################################
    task_description :str = "Insert the square into the stick"
    # fmt: on

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None

# =============== openvla functions ===============
def get_server_endpoint(cfg: GenerateConfig):
    """Get the server endpoint for remote inference."""
    ip_address = socket.gethostbyname(cfg.vla_server_url)
    return f"http://{ip_address}:8777/act"

def prepare_observation(img,left_wrist_img, right_wrist_img,qpos,resize_size):
    """Prepare observation for policy input."""
    img_resized = resize_image_for_policy(img, resize_size)
    left_wrist_img_resized = resize_image_for_policy(left_wrist_img, resize_size)
    right_wrist_img_resized = resize_image_for_policy(right_wrist_img, resize_size)
    observation = {
        "full_image": img_resized,
        # "left_wrist_image": left_wrist_img_resized,
        # "right_wrist_image": right_wrist_img_resized,
        # "state": qpos,
    } 
    return observation, img_resized, left_wrist_img_resized, right_wrist_img_resized

# ===================== 生产者线程：反复请求 action chunk =====================
def _action_producer_loop(cfg: GenerateConfig, server_endpoint, resize_size, ros_operator):
    """
    后台线程：循环
      1) 读帧并组装 observation
      2) 请求服务端，返回一个 action chunk
      3) 将 chunk 拆成单步，按 expected_t 放入各自队列（记录 created/expected）
    主线程每个 timestep 从对应 expected_t 的队列取“最新创建”的 action 执行。
    """
    print_flag_local = True
    global _control_t
    rate = rospy.Rate(cfg.publish_rate)  # 控制请求频率；可根据模型吞吐调整
    while not _action_stop_event.is_set() and not rospy.is_shutdown():
        with _action_lock:
            _schedule_cursor = _control_t
        # 采帧（必要时等待）
        result = ros_operator.get_frame()
        if not result:
            if print_flag_local:
                print("async syn fail")
                print_flag_local = False
            rate.sleep()
            continue
        print_flag_local = True
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = result

        qpos = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
        observation, _, _, _ = prepare_observation(img_front, img_left, img_right, qpos, resize_size)
        observation["instruction"] = cfg.task_description

        # 请求 chunk
        start_time = time.time()
        try:
            actions_chunk = get_action_from_server(observation, server_endpoint)
        except Exception as e:
            print("get_action_from_server error:", e)
            rate.sleep()
            continue
        latency = time.time() - start_time

        # 入各 expected_t 队列
        _enqueue_chunk_to_expected_queues(actions_chunk, latency, _schedule_cursor)

        # 节流：避免无限堆积（可按需调整）
        with _action_lock:
            # 估个总量：所有队列长度之和
            total_len = sum(len(q) for q in _action_queues.values())
        if total_len > 3 * cfg.num_open_loop_steps:
            time.sleep(0.02)
        else:
            rate.sleep()

# ===================== 主循环：逐 timestep 消费 =====================
def run_openvla_oft(
    cfg: GenerateConfig,
    task_description: str,
    server_endpoint: str,
    resize_size,
    ros_operator
):
    # Setup
    t = 0
    replay_images = []
    replay_images_resized = []
    replay_images_left_wrist_resized = []
    replay_images_right_wrist_resized = []

    # 初始位姿
    right0 = [0.0553 , 1.7357, -0.7476 , 0.0677 , 0.3858 ,-0.0457 , 0.0644]
    left0  = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    last_right_action = np.array(right0)

    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("按回车开始测试")
    
    # 清空缓冲，启动生产者线程（只启动一次）
    clear_action_buffer()
    global _action_prod_thread
    if _action_prod_thread is None or not _action_prod_thread.is_alive():
        _action_stop_event.clear()
        _action_prod_thread = Thread(
            target=_action_producer_loop,
            args=(cfg, server_endpoint, resize_size, ros_operator),
            daemon=True
        )
        _action_prod_thread.start()

    # 平滑到初始位姿
    

    episode_start_time = time.time()
    print_flag = True
    global _control_t
    try:
        print("按空格结束采集")
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        rate = rospy.Rate(50)  # 控制周期
        time.sleep(1)
        while t < cfg.max_steps and not rospy.is_shutdown():
            with _action_lock:
                _control_t = t
            step_start_time = time.time()

            # （仅用于保存重放图像；推理由后台线程完成）
            # result = ros_operator.get_frame()
            # if not result:
            #     if print_flag:
            #         print("syn fail")
            #         print_flag = False
            #     rate.sleep()
            #     continue
            # print_flag = True

            # (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
            #  puppet_arm_left, puppet_arm_right, robot_base) = result

            # qpos = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
            # _, img_resized, left_wrist_resized, right_wrist_resized = prepare_observation(
            #     img_front, img_left, img_right, qpos, resize_size
            # )
            # replay_images_resized.append(img_resized)
            # replay_images_left_wrist_resized.append(left_wrist_resized)
            # replay_images_right_wrist_resized.append(right_wrist_resized)

            # === 关键：对 expected timestep = t，从队列中取“最新创建”的 action ===
            head = pick_latest_for_expected_t(t)
            if head is None:
                # 该 timestep 的动作还没准备好；等待下一拍
                assert False , "No action in queue"
                #continue
            rate.sleep()
            # print(f"Timestep: {t}, expected timestep = {head['expected_timestep']}, created timestep = {head['created_timestep']}")
            print(f"Timestep: {t}, expected={head['expected_timestep']}, created_at_control_t={head['created_timestep']}, created_seq={head.get('created_seq')}")

            # 拿到要执行的动作（根据你的模型维度调整切片）
            right_action = np.array(head["action"]).astype(float).tolist()
            # 如果返回包含双臂/夹爪等，可在此拆分：
            # right_action, left_action = right_action[:7], right_action[7:14]
            left_action  = list(left0)

            # 安全检查（保留你的阈值）
            prev_curr_l1 = np.mean(np.abs(np.array(right_action)[0:6] - last_right_action[0:6]))
            assert prev_curr_l1 < 0.5, "移动角度过大"

            right_state = np.array(ros_operator.puppet_arm_right_deque[-1].position)
            prev_state_l1 = np.mean(np.abs(right_state[0:6] - last_right_action[0:6]))
            assert prev_state_l1 < 0.3, "与目标状态差异过大"

            last_right_action = np.array(right_action)

            # 发布
            ros_operator.puppet_arm_publish(left_action, right_action)

            # 时间推进
            t += 1
            rate.sleep()

            # 按键停止
            r, w, x = select.select([sys.stdin], [], [], 0)
            if r:
                key = sys.stdin.read(1)
                if key == ' ':
                    print("结束单次测试")
                    break

        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        termios.tcflush(fd, termios.TCIFLUSH)

    except Exception as e:
        print("Exception:", e)

    episode_end_time = time.time()

    # 如需在一次 episode 后停生产者，可解注：
    # _action_stop_event.set()
    # if _action_prod_thread is not None:
    #     _action_prod_thread.join(timeout=2.0)

    user_input = input("Success? Enter 'y' or 'n': ")
    success = True if user_input.lower() == "y" else False

    episode_stats = {
        "success": success,
        "total_steps": t,
    }
    return (
        episode_stats,
        replay_images,
        replay_images_resized,
        replay_images_left_wrist_resized,
        replay_images_right_wrist_resized,
    )

# ===================== ROS Operator（与你原版一致） =====================
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
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]; vel_msg.linear.y = 0; vel_msg.linear.z = 0
        vel_msg.angular.x = 0; vel_msg.angular.y = 0; vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None; right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep(); continue
            else:
                break

        left_symbol  = [1 if left[i]  - left_arm[i]  > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True; step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff  = [abs(left[i]  - left_arm[i])  for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]; flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]; flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = ['joint0','joint1','joint2','joint3','joint4','joint5','joint6']
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100; rate = rospy.Rate(200)
        left_arm = None; right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep(); continue
            else:
                break
        traj_left_list  = np.linspace(left_arm, left,  num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)
        for i in range(len(traj_left_list)):
            traj_left  = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1]  = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = ['joint0','joint1','joint2','joint3','joint4','joint5','joint6']
            self.puppet_arm_left_publisher.publish(JointState(
                header=joint_state_msg.header, name=joint_state_msg.name, position=traj_left))
            self.puppet_arm_right_publisher.publish(JointState(
                header=joint_state_msg.header, name=joint_state_msg.name, position=traj_right))
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
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
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

# ===================== 其他：参数 / 入口 =====================
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, default='aloha_mobile_dummy', required=False)
    parser.add_argument('--max_publish_step', action='store', type=int, default=10000, required=False)
    parser.add_argument('--ckpt_name', action='store', type=str, default='policy_best.ckpt', required=False)
    parser.add_argument('--ckpt_stats_name', action='store', type=str, default='dataset_stats.pkl', required=False)
    parser.add_argument('--policy_class', action='store', type=str, default='ACT', required=False)
    parser.add_argument('--batch_size', action='store', type=int, default=8, required=False)
    parser.add_argument('--seed', action='store', type=int, default=0, required=False)
    parser.add_argument('--num_epochs', action='store', type=int, default=2000, required=False)
    parser.add_argument('--lr', action='store', type=float, default=1e-5, required=False)
    parser.add_argument('--weight_decay', type=float, default=1e-4, required=False)
    parser.add_argument('--dilation', action='store_true', required=False)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), required=False)
    parser.add_argument('--masks', action='store_true', required=False)
    parser.add_argument('--kl_weight', action='store', type=int, default=10, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, default=3200, required=False)
    parser.add_argument('--temporal_agg', action='store', type=bool, default=True, required=False)
    parser.add_argument('--state_dim', action='store', type=int, default=14, required=False)
    parser.add_argument('--lr_backbone', action='store', type=float, default=1e-5, required=False)
    parser.add_argument('--backbone', action='store', type=str, default='resnet18', required=False)
    parser.add_argument('--loss_function', action='store', type=str, default='l1', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, default=8, required=False)
    parser.add_argument('--dropout', default=0.1, type=float, required=False)
    parser.add_argument('--pre_norm', action='store_true', required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, default='/camera_r/color/image_raw', required=False)

    parser.add_argument('--img_front_depth_topic', action='store', type=str, default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, default='/camera_r/depth/image_raw', required=False)

    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, default='/puppet/joint_right', required=False)

    parser.add_argument('--robot_base_topic', action='store', type=str, default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, default=25, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, default=0, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, default=32, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, default=[0.01,0.01,0.01,0.01,0.01,0.01,0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store', type=bool, default=False, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, default=False, required=False)
    args = parser.parse_args()
    return args

def eval_aloha(cfg: GenerateConfig, ros_operator) -> None:
    success_episode = 0
    total_episodes = 0
    resize_size = get_image_resize_size(cfg)
    server_endpoint = get_server_endpoint(cfg)
    task_description = cfg.task_description

    if cfg.policy_type == "openvla-oft":
        while True:
            episode_stats, replay_images, replay_images_resized, replay_images_left_wrist, replay_images_right_wrist = (
                run_openvla_oft(cfg, task_description, server_endpoint, resize_size, ros_operator)
            )
            if episode_stats["success"]:
                success_episode += 1
            total_episodes += 1
            success_rate = success_episode / total_episodes
            print("="*100)
            print("当前测试次数:", total_episodes)
            print("此次是否成功:", episode_stats["success"])
            print("当前成功率:", success_rate*100, "%")
            print("当前成功次数:", success_episode, "总测试次数:", total_episodes)
    else:
        raise NotImplementedError

def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    cfg = GenerateConfig()                 # 注意：要实例化！
    eval_aloha(cfg, ros_operator)

if __name__ == '__main__':
    main()
# python act/inference.py --ckpt_dir ~/train0314/
