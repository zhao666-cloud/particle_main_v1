import gym
import numpy as np
#
# sign_table = {'cone':0,'cylinder':1,'part_ring':2,'part_sphere':3}
# env = gym.make('env_gym:ur5_env-v0',object=False)
# obs = env.reset()
# particles = np.loadtxt('li.txt')
# seg_sign = np.loadtxt('li_part.txt')
# env.show_part_particle(particles,seg_sign)
# while 1:
#     pass

particles = np.loadtxt('li.txt')
seg_sign = np.loadtxt('li_part.txt')
for p in particles:
    print(p)