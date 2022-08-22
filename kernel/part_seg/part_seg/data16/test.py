import os

import numpy as np
import gym
import torch
import time
from scipy.spatial.transform import Rotation as R
def sample(k):
    r = np.random.uniform(0.9*k,k)
    theta = np.random.uniform(0,2*np.pi)
    y = r*np.sin(theta)
    z = np.random.uniform(-0.01,0.02)
    x = r * np.cos(theta) + np.sqrt((1-z**2/0.03**2)*0.01**2)
    return [x,y,z]
def flat_sample(k1,k2):
    a = np.random.uniform(0.9*k1,k1)
    b = np.random.uniform(0.9*k2,k2)
    theta = np.random.uniform(0,2*np.pi)
    y = b*np.sin(theta)
    z = np.random.uniform(-0.02,0.02)
    x = a * np.cos(theta) + np.sqrt((1 - z ** 2 / 0.04 ** 2) * 0.1 ** 2)
    return [x,y,z]
env = gym.make('env_gym:ur5_env-v0', object=0)
obs = env.reset()
s = 4
for i in [9]:
    data = np.loadtxt(f'{i}/points4.txt')
    points = data[:,:3]
    sign = data[:,3]
    mask = sign.round(0)==s
    if sum(mask) == 0:
        continue
    print(i)
    points = points[mask]
    particles = []
    mean = points.mean(axis=0)
    left_points = []
    right_points = []
    for point in points:
        if point[0] < mean[0]:
            left_points.append(point)
        else:
            right_points.append(point)
    left_points = np.array(left_points)
    right_points = np.array(right_points)
    max_y = np.sort(left_points[:,1])[-15:].sum()/15
    min_y = np.sort(left_points[:,1])[:15].sum()/15
    max_z = np.sort(right_points[:,2])[-15:].sum()/25
    min_z = np.sort(right_points[:,2])[:15].sum()/25
    # points = points - mean
    left_mean = right_points.mean(axis=0)
    left_points = right_points - left_mean
    k1 = (max_y - min_y)/2
    k2 = (max_z - min_z)/2
    ori = [0,0,-90]
    ori_matrix = R.from_euler('zyx',ori,degrees=True).as_matrix()
    new_points = np.dot(ori_matrix,left_points.T).T
    for _ in range(400):
        p = sample(k2)
        particles.append(p)
    particles = np.array(particles).reshape(-1,3)
    particles[:,0] = particles[:,0] + 0.0025
    temp_points = torch.tensor(new_points).unsqueeze(0).repeat(400,1,1).cuda()
    temp_particles = torch.tensor(particles).unsqueeze(1).repeat(1,temp_points.shape[1],1).cuda()
    temp = np.argsort(np.array(torch.pow(temp_points - temp_particles,2).cpu()).sum(axis=2))[:,:2].reshape(-1,)
    print(len(set(temp)))
    if len(set(temp)) < int(new_points.shape[0]/10):
        a = range(400)
        append = np.random.choice(a,int(new_points.shape[0]/10)-len(set(temp)),replace=False)
        new_points = np.concatenate((new_points,particles[append]),axis=0)
        show = np.zeros((new_points.shape[0]))
        show[temp]=s
        show[-append.shape[0]:] = s
    else:
        show = np.zeros((new_points.shape[0]))
        show[temp]=s
    new_data = np.concatenate((new_points,show.reshape(-1,1)),axis=1)
    #np.savetxt(f'data/9/points.txt',new_data)
    #env.show_particle(particles)
    res = env.show_part_particle(new_points,show)
    #time.sleep(5)
    while 1:
        pass
    #env.hide_particle(res)