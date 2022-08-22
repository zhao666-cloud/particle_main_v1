import json

import gym
import numpy as np
from scipy.spatial.transform import Rotation as R

env = gym.make('env_gym:ur5_env-v0')
obs = env.reset()
with open("gripper.json",'r') as f:
    row_data = json.load(f)
def genera_gripper(pos,ori,gripper):
    key_points = []
    k = R.from_euler('x',180,degrees=True).as_matrix()
    o = R.from_euler('zyx',ori,degrees=True).as_matrix()
    joint_point = -np.array([0,0,gripper["center2joint"]])
    key_points.append(joint_point)
    handle = joint_point - np.array([0,0,gripper["handle"]])
    key_points.append(handle)
    left_corner = joint_point + np.array([gripper["joint2corner"],0,0])
    key_points.append(left_corner)
    right_corner = joint_point - np.array([gripper["joint2corner"], 0, 0])
    key_points.append(right_corner)
    left_point = left_corner + np.array([0,0,gripper["gripper"]])
    key_points.append(left_point)
    right_point = right_corner + np.array([0,0,gripper["gripper"]])
    key_points.append(right_point)
    key_points = np.array(key_points)
    key_points = np.dot(k, np.array(key_points).T).T
    key_points = np.dot(o,np.array(key_points).T).T
    for i,p in enumerate(key_points):
        key_points[i] = p+np.array(pos)
    return key_points
def draw_gripper(key_points):
    joint_point,handle,left_corner,right_corner,left_point,right_point = key_points
    env.draw_line(left_point,left_corner)
    env.draw_line(left_corner,joint_point)
    env.draw_line(joint_point,right_corner)
    env.draw_line(right_corner,right_point)
    env.draw_line(joint_point,handle)
key_points1 = genera_gripper([0.5,0.5,0.5],[0,0,0],row_data)
key_points2 = genera_gripper([0.5,0.5,0.5],[90,0,0],row_data)
draw_gripper(key_points1)
draw_gripper(key_points2)
while 1:
    pass
