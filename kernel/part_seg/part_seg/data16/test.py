import os

import numpy as np
import gym
import torch
import time
from scipy.spatial.transform import Rotation as R
from kernel.grasp_planning.utils import generate_gripper,draw_gripper
import json
import random
def sphere_sample(minp,maxp):
    R = np.sqrt(sum((maxp-minp)**2))/2
    r = np.random.uniform(0.95*R,R)
    theta = np.random.uniform(0,2*np.pi)
    alpha = np.random.uniform(0,2*np.pi)
    x = r*np.sin(theta)*np.cos(alpha)
    y = r*np.sin(theta)*np.sin(alpha)
    z = r*np.cos(theta)
    return [x,y,z]
def X_times(a,b):
    x1 = a[0]
    y1 = a[1]
    z1 = a[2]
    x2 = b[0]
    y2 = b[1]
    z2 = b[2]
    x3 = y1*z2-y2*z1
    y3 = -x1*z2+x2*z1
    z3 = x1*y2-x2*y1
    return [x3,y3,z3]
#boundary_point
def grasp_points(data,sign=0):
    s = sign
    points = data[:,:3]
    sign = data[:,3]
    mask = sign.round(0)==s
    if sum(mask) < 20:
        return None
    points = points[mask]
    mean = points.mean(axis=0)
    max_x = np.sort(points[:,0])[-20:].sum()/20
    min_x = np.sort(points[:,0])[:20].sum()/20
    max_y = np.sort(points[:,1])[-20:].sum()/20
    min_y = np.sort(points[:,1])[:20].sum()/20
    max_z = np.sort(points[:,2])[-20:].sum()/20
    min_z = np.sort(points[:,2])[:20].sum()/20
    maxp = np.array([max_x,max_y,max_z])
    minp = np.array([min_x,min_y,min_z])

    points = points - mean
    ori = [0,np.pi/4,0]
    ori_matrix = R.from_euler('zyx',ori,degrees=True).as_matrix()
    new_points = np.dot(ori_matrix,points.T).T
    particles = []
    for i in range(1000):
        particles.append(sphere_sample(minp,maxp))
    particles = np.array(particles)
    temp_points = torch.tensor(new_points).unsqueeze(0).repeat(particles.shape[0],1,1).cuda()
    temp_particles = torch.tensor(particles).unsqueeze(1).repeat(1,temp_points.shape[1],1).cuda()
    temp = np.argsort(np.array(torch.pow(temp_points - temp_particles,2).cpu()).sum(axis=2))[:,:2].reshape(-1,)
    show = np.zeros((new_points.shape[0]))
    show[temp]=10
    return particles,new_points,show,mean

def gaussian_grasp(mean):
    #gaussian_point
    gaussian_point = np.random.randn(1,3)*0.01 + mean
    #axis-angle-->quat
    value = gaussian_point - mean
    new_value = np.insert(value / np.sqrt(sum(sum(value**2))),0,0)

    ori = R.from_quat(new_value).as_euler('zyx',degrees=True)
    return ori,mean
def contract_point_distance(pos,ori,row_data,new_points,show):
    key_points = generate_gripper(pos,ori,row_data)

    left_point,right_point = key_points[-2:]

    boundary_point = new_points[show==10]+pos
    distance1 = np.sqrt(np.sum((boundary_point - left_point)**2,axis=1))
    point1 = np.argsort(distance1)[0]

    distance2 = np.sqrt(np.sum((boundary_point - right_point)**2,axis=1))
    point2 = np.argsort(distance2)[0]

    distance3 = np.sqrt(np.sum((boundary_point - boundary_point[point1])**2,axis=1))
    distance4 = np.sqrt(np.sum((boundary_point - boundary_point[point2])**2,axis=1))
    return key_points,boundary_point,distance3,distance4
def contact_vec(boundary_point,distance):
    contact_region = np.argsort(distance)[:8]
    show_boundary = np.zeros(boundary_point.shape[0])
    show_boundary[contact_region]=10
    h_sample = []
    for i in range(50):
        sample = np.random.choice(contact_region,3,replace=False)

        s1 = boundary_point[sample[0]]
        s2 = boundary_point[sample[1]]
        s3 = boundary_point[sample[2]]

        s21 = s2 - s1
        s31 = s3 -s1
        h_sample.append(X_times(s21,s31))
    h = np.sum(np.array(h_sample),axis=0)/50
    print(h)
    normalization_h = h/np.sqrt(np.sum(h**2))

    return normalization_h,show_boundary
if __name__ == "__main__":
    env = gym.make('env_gym:ur5_env-v0', object=0)
    obs = env.reset()
    data = np.loadtxt('3/points4.txt')
    with open("/media/zcl/file2/particle_main_v1/kernel/grasp_planning/gripper.json", 'r') as f:
        row_data = json.load(f)
    particles,new_points,show,mean= grasp_points(data,3)

    while 1:
        ori,pos = gaussian_grasp(mean)
        key_points,boundary_point,distance3,distance4 = contract_point_distance(pos,ori,row_data,new_points,show)
        print(sum(key_points[:,2]>0))
        if sum(key_points[:,2]>0) != 6:
            continue
        normalization_h1,show_boundary1 = contact_vec(boundary_point,distance3)
        normalization_h2,show_boundary2 = contact_vec(boundary_point,distance4)
        cos_theta1 = normalization_h2.dot(normalization_h1)/np.sum(normalization_h2**2)/np.sum(normalization_h1**2)
        contract_point1 = np.sum(boundary_point[show_boundary1==10],axis=0)/8
        contract_point2 = np.sum(boundary_point[show_boundary2==10],axis=0)/8
        point1topoint2 = contract_point1 - contract_point2
        point1topoint2 = point1topoint2 / np.sqrt(np.sum(point1topoint2**2))
        cos_theta2 = normalization_h2.dot(point1topoint2) / np.sum(normalization_h2 ** 2) / np.sum(point1topoint2 ** 2)
        cos_theta3 = normalization_h1.dot(point1topoint2) / np.sum(normalization_h1 ** 2) / np.sum(point1topoint2 ** 2)
        if cos_theta1**2 > 0.85 and cos_theta2**2 > 0.85 and cos_theta3**2 > 0.85:
            break


    res = env.show_part_particle(boundary_point,show_boundary1+show_boundary2)
    # env.show_particle(points+mean)
    draw_gripper(env,key_points)
    env.draw_line([0,0,0],normalization_h1,[1,0,0])
    env.draw_line([0,0,0],normalization_h2,[0,1,0])
    env.draw_line([0, 0, 0],point1topoint2, [0, 0, 1])
    #time.sleep(5)
    while 1:
        pass
    #env.hide_particle(res)