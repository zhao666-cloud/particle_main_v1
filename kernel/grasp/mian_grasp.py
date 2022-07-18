import os
import torch
import gym
import numpy as np
import cv2
from kernel.generate_particles.fc_gen import FCGenerator

from kernel.generate_particles.recon_particle import generate_point_cloud
from kernel.utils import save_data,get_position
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
env.show_particle(particles)
while 1:
    pass
#TODO pyflex
