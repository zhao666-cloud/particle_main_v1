import os
import cv2
import numpy as np
ROOT_PATH = os.path.dirname(__file__)

def save_data(rgb,seg,m_r_list,m_t_list,intrinsic):
    file_root_path = f'{ROOT_PATH}/generate_particle/data'
    # rgb_path = f'{file_root_path}/rgb'
    # if not os.path.exists(rgb_path):
    #     os.makedirs(rgb_path)
    # for i,r in enumerate(rgb):
    #     cv2.imwrite(os.path.join(rgb_path,f'{i}.jpg'),r*255)
    seg_path = f'{file_root_path}/seg'
    if not os.path.exists(seg_path):
        os.makedirs(seg_path)
    for i,s in enumerate(seg):
        cv2.imwrite(os.path.join(seg_path,f'{i}_seg.jpg'),s*255)
def save_particles(particles):
    file_root_path = f'{ROOT_PATH}/generate_particle/data'
    particles_path = f'{file_root_path}/particles'
    if not os.path.exists(particles_path):
        os.makedirs(particles_path)
def get_position(mask,bound_x,bound_y):
    roate_mask = mask.copy().astype(np.uint8)
    M = cv2.getRotationMatrix2D((128,128),45,1.0)
    roate_mask = cv2.warpAffine(roate_mask,M,(256,256))*255
    binary = cv2.threshold(roate_mask,0,255,cv2.THRESH_BINARY)[1]
    kernel = np.ones((10, 10), np.uint8)
    close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('mm',close)
    cv2.waitKey()
    contours,hierarchy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    center_x = int(M['m10']/M['m00'])
    center_y = int(M['m01']/M['m00'])
    position_x = bound_x * center_x/255
    position_y = bound_y * center_y/255
    print(position_x,position_y)
    #TODO mutil objects
    return position_x,position_y





