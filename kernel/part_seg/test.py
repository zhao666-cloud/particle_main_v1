import sys
import gym
import torch
import numpy as np
import importlib
from tqdm import tqdm
from ShapeNetDataLoader import PartNormalDataset

# def to_categorical(y, num_classes):
#     """ 1-hot encodes a tensor """
#     new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
#     if (y.is_cuda):
#         return new_y.cuda()
#     return new_y
# root = '../part_seg_data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
# sys.path.append('pointnet2_part_seg_msg')
#
# seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
#                'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
#                'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
#
#
# DATASET = PartNormalDataset(root=root,npoints=2048,split='test',normal_channel=True)
# DataLoader = torch.utils.data.DataLoader(DATASET,batch_size=24,shuffle=False,num_workers=4)
# num_classes = 16
# num_part = 50
#
# model_name = 'pointnet2_part_seg_msg'
# MODEL = importlib.import_module(model_name)
# classifier = MODEL.get_model(num_part,normal_channel=True).cuda()
# checkpoint = torch.load('pointnet2_part_seg_msg/checkpoints/best_model.pth')
# classifier.load_state_dict(checkpoint['model_state_dict'])
#
# with torch.no_grad():
#     seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
#
#     for cat in seg_classes.keys():
#         for label in seg_classes[cat]:
#             seg_label_to_cat[label] = cat
#     classifier = classifier.eval()
#     for batch_id,(points,label,target) in tqdm(enumerate(DataLoader),total=(len(DataLoader)),smoothing=0.9):
#         batchsize,num_point,_ = points.size()
#         cur_batch_size, NUM_POINT, _ = points.size()
#         points,label,target = points.float().cuda(),label.long().cuda(),target.long().cuda()
#         points = points.transpose(2,1)
#         vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()  # B N 50
#         for _ in range(3):
#             seg_pred, _ = classifier(points, to_categorical(label, num_classes))
#             vote_pool += seg_pred
#         seg_pred = vote_pool / 3
#         cur_pred = seg_pred.cpu().data.numpy()
#         cur_pred_val_logits = cur_pred
#         cur_pred = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
#         target = target.cpu().data.numpy()
#         for i in range(cur_batch_size):
#             cat = seg_label_to_cat[target[i, 0]]
#             logits = cur_pred_val_logits[i, :, :]
#             cur_pred[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
#             seg_sign = cur_pred[i,:]
#             prec_points = points.transpose(2,1)[i,:,:].cpu().numpy()
#             data = np.c_[prec_points,seg_sign]
#             np.savetxt('cat'+str(i)+'.txt',data)

data = np.loadtxt('cat0.txt')
particle = data[:,:3]/8
particle[:,0] += 0.5
particle[:,1] += 0.5
particle[:,2] += 0.3
particles = torch.tensor(particle)
sign = data[:,6]
env = gym.make('env_gym:ur5_env-v0')
obs = env.reset()
env.show_part_particle(particles,sign)
while 1:
    pass








