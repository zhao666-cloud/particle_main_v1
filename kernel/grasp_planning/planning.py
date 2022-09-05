"""
Function:Grasp_Planning
Date:2022/08/30
Author:Zhao
"""
import gym
import json
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from kernel.grasp_planning.utils import select_grasp_pose,draw_gripper


if __name__ == "__main__":
    env = gym.make('env_gym:ur5_env-v0', object=0)
    obs = env.reset()
    data = np.loadtxt('/media/zcl/file2/particle_main_v1/kernel/part_seg/part_seg/data16/3/points4.txt')
    pos,ori,gripper_key_points,boundary = select_grasp_pose(data,3)
    draw_gripper(env,gripper_key_points)
    while 1:
        pass