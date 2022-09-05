import os
import torch
import gym
import numpy as np
import sys
import cv2
from kernel.generate_particles.fc_gen import FCGenerator
from kernel.part_seg.part_seg.part_seg import part_seg
from kernel.generate_particles.recon_particle import generate_point_cloud
from kernel.utils import save_data,get_position
from kernel.part_seg.pointnet2_part_seg_msg.pointnet2_utils import farthest_point_sample,index_points
from kernel.args import get_args
from kernel.grasp_planning.utils import select_grasp_pose
ROOT_PATH = os.path.dirname(__file__)
sys.path.append('../part_seg/part_seg')
args = get_args()
env = gym.make('env_gym:ur5_env-v0',object=1)
obs = env.reset()
#estimate object rough pos
top_mask = env.get_top_iamge(np.array([0.5,0.5,0.001]))
x0,y0 = get_position(top_mask,1,1)
#reconstruction
rgb,seg,m_r_list,m_t_list,intrinsic = env.get_viode(x0,y0)

particles = generate_point_cloud(env,m_r_list,m_t_list,intrinsic,seg,rgb,x0,y0)
sign = part_seg(particles.cpu().data.numpy(),'aabest_model.pth')
print(particles.shape,sign.shape)
data = np.concatenate((particles.cpu().data.numpy(),sign.reshape(-1,1)),axis=1)
pos, ori, gripper_key_points, boundary = select_grasp_pose(data, 3)
env.show_part_particle(particles,sign)
# i = 15
# try:
#     np.savetxt(f'data/{i}/{i}.txt',particles.detach().cpu().numpy())
# except:
#     os.mkdir(f'data/{i}')
#     np.savetxt(f'data/{i}/{i}.txt', particles.detach().cpu().numpy())
while 1:
    pass
#TODO pyflex
