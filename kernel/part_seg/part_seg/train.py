import argparse
import importlib
import os
import numpy as np
import shutil
from tqdm import tqdm
import provider
from ShapeNetDataLoader import PartNormalDataset
import torch
seg_classes = {'cone':[0],'cylinder':[1],'part_ring':[2],'part_sphere':[3]}
seg_label_to_cat = {}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--npoints',type=int,default=2048,help='point Number')
    parser.add_argument('--epoch', default=512, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    return parser.parse_args()
def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] =args.gpu

    root = 'data'
    TRAIN_DATASET = PartNormalDataset(root=root,npoints=args.npoints,normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET,batch_size=args.batch_size,shuffle=True,num_workers=10,drop_last=True)
    num_classes = 4
    num_part = 7

    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(num_part,normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('No existing model, starting training from scratch...')

    start_epoch = 0
    classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        print('Learning rate:%f' % lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoints))
            loss = criterion(seg_pred,target,trans_feat)
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        state = {
            'epoch': epoch,
            'train_acc': train_instance_acc,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, 'best_model.pth')
        print(train_instance_acc)
        global_epoch += 1

if __name__ == '__main__':
    args = parse_args()
    main(args)