import os
import torch
import gym
import numpy as np
import cv2
from kernel.generate_particles.fc_gen import FCGenerator

from kernel.generate_particles.recon_particle import generate_point_cloud
from kernel.utils import save_data,get_position
from kernel.part_seg.pointnet2_part_seg_msg.pointnet2_utils import farthest_point_sample,index_points
from kernel.args import get_args

ROOT_PATH = os.path.dirname(__file__)
print(ROOT_PATH)
args = get_args()
env = gym.make('env_gym:ur5_env-v0')
obs = env.reset()
#estimate object rough pos
top_mask = env.get_top_iamge(np.array([0.5,0.5,0.001]))
x0,y0 = get_position(top_mask,1,1)
#reconstruction
rgb,seg,m_r_list,m_t_list,intrinsic = env.get_viode(x0,y0)
#save_data(rgb,seg,m_r_list,m_t_list,intrinsic)
particles = generate_point_cloud(env,m_r_list,m_t_list,intrinsic,seg,rgb,x0,y0)
#sample_particles = farthest_point_sample(particles.unsqueeze(0),500)
#new_points = index_points(particles.unsqueeze(0),sample_particles)
#particles = torch.tensor(np.loadtxt('particle.txt'))
env.show_particle(particles)
np.savetxt('li.txt',particles.detach().cpu().numpy())
while 1:
    pass
#TODO pyflex
