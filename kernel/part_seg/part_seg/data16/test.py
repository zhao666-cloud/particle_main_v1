import os

import numpy as np
import gym
import torch
import time
from scipy.spatial.transform import Rotation as R
def sample(minp,maxp):
    dis = maxp - minp
    particles = []
    y1 = np.arange(-dis[1]/2,dis[1]/2,dis[1]/10)
    z1 = np.arange(-dis[2]/2,dis[2]/2,dis[1]/10)
    yy1, zz1 = np.meshgrid(y1, z1)
    xx1 = -np.ones_like(yy1)*dis[0]/2
    for i in range(xx1.shape[0]):
        for j in range(xx1.shape[1]):
            particles.append([xx1[i,j],yy1[i,j],zz1[i,j]])

    x2 = np.arange(-dis[0]/2,dis[0]/2,dis[0]/10)
    y2 = np.arange(-dis[1]/2,dis[1]/2,dis[0]/10)
    yy2, xx2 = np.meshgrid(y2, x2)
    zz2 = -np.ones_like(yy2)*dis[2]/2
    for i in range(yy2.shape[0]):
        for j in range(yy2.shape[1]):
            particles.append([xx2[i,j],yy2[i,j],zz2[i,j]])

    y1 = np.arange(-dis[1]/2,dis[1]/2,dis[1]/11)
    z1 = np.arange(-dis[2]/2,dis[2]/2,dis[1]/11)
    yy1, zz1 = np.meshgrid(y1, z1)
    xx1 = np.ones_like(yy1)*dis[0]/2
    for i in range(xx1.shape[0]):
        for j in range(xx1.shape[1]):
            particles.append([xx1[i,j],yy1[i,j],zz1[i,j]])

    x2 = np.arange(-dis[0]/2,dis[0]/2,dis[0]/10)
    y2 = np.arange(-dis[1]/2,dis[1]/2,dis[0]/10)
    yy2, xx2 = np.meshgrid(y2, x2)
    zz2 = np.ones_like(yy2)*dis[2]/2
    for i in range(yy2.shape[0]):
        for j in range(yy2.shape[1]):
            particles.append([xx2[i,j],yy2[i,j],zz2[i,j]])

    x3 = np.arange(-dis[0]/2,dis[0]/2,dis[0]/10)
    z3 = np.arange(-dis[2]/2,dis[2]/2,dis[0]/10)
    xx3 ,zz3 = np.meshgrid(x3, z3)
    yy3 = -np.ones_like(xx3)*dis[1]/2
    for i in range(xx3.shape[0]):
        for j in range(xx3.shape[1]):
            particles.append([xx3[i,j],yy3[i,j],zz3[i,j]])

    x3 = np.arange(-dis[0]/2,dis[0]/2,dis[0]/10)
    z3 = np.arange(-dis[2]/2,dis[2]/2,dis[0]/10)
    xx3 ,zz3 = np.meshgrid(x3, z3)
    yy3 = np.ones_like(xx3)*dis[1]/2
    for i in range(xx3.shape[0]):
        for j in range(xx3.shape[1]):
            particles.append([xx3[i,j],yy3[i,j],zz3[i,j]])

    return np.array(particles)
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
s = 0
for i in [9]:
    data = np.loadtxt(f'{i}/points4.txt')
    points = data[:,:3]
    sign = data[:,3]
    mask = sign.round(0)==s
    if sum(mask) == 0:
        continue
    print(i)
    points = points[mask]
    mean = points.mean(axis=0)
    max_x = np.sort(points[:,0])[-15:].sum()/15
    min_x = np.sort(points[:,0])[:15].sum()/15
    max_y = np.sort(points[:,1])[-15:].sum()/15
    min_y = np.sort(points[:,1])[:15].sum()/15
    max_z = np.sort(points[:,2])[-15:].sum()/15
    min_z = np.sort(points[:,2])[:15].sum()/15
    maxp = np.array([max_x,max_y,max_z])
    minp = np.array([min_x,min_y,min_z])
    points = points - mean
    ori = [0,0,0]
    ori_matrix = R.from_euler('zyx',ori,degrees=True).as_matrix()
    new_points = np.dot(ori_matrix,points.T).T
    particles = sample(minp,maxp)
    print(particles.shape)
    temp_points = torch.tensor(new_points).unsqueeze(0).repeat(particles.shape[0],1,1).cuda()
    temp_particles = torch.tensor(particles).unsqueeze(1).repeat(1,temp_points.shape[1],1).cuda()
    temp = np.argsort(np.array(torch.pow(temp_points - temp_particles,2).cpu()).sum(axis=2))[:,0].reshape(-1,)
    print(len(set(temp)))
    if len(set(temp)) < int(new_points.shape[0]/10):
        a = range(particles.shape[0])
        append = np.random.choice(a,int(new_points.shape[0]/10)-len(set(temp)),replace=False)
        new_points = np.concatenate((new_points,particles[append]),axis=0)
        show = np.zeros((new_points.shape[0]))
        show[temp]=10
        show[-append.shape[0]:] = 10
    else:
        show = np.zeros((new_points.shape[0]))
        show[temp]=10
    new_data = np.concatenate((new_points,show.reshape(-1,1)),axis=1)
    np.savetxt(f'data/19/points.txt',new_data)
    #env.show_particle(particles)
    res = env.show_part_particle(new_points,show)
    #time.sleep(5)
    while 1:
        pass
    #env.hide_particle(res)