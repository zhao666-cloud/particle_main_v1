import torch
import numpy as np
from kernel.generate_particles.fc_gen import FCGenerator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from kernel.generate_particles.recon_dataset import reconDataset
from kernel.generate_particles.recon_loss import mask_loss_batch
from kernel.args import get_args
def optimize_cloud(cloud):
    # remove outliner
    gt2p_pos = cloud.clone()
    gt2p_pos = gt2p_pos.unsqueeze(0).repeat((cloud.size(0), 1, 1))
    gt2p_gtpos = cloud.clone()
    gt2p_gtpos = gt2p_gtpos.unsqueeze(1).repeat((1, cloud.size(0), 1))
    gt2p_dis = torch.pow(gt2p_gtpos - gt2p_pos, 2).sum(dim=2).sqrt()
    index_x = torch.arange(cloud.size(0)).cuda()
    gt2p_dis[index_x, index_x] = 100
    min_k = 1.0 / ((1.0 / gt2p_dis).topk(15)[0].mean(1))

    avg = min_k.mean()
    pos = cloud[min_k < avg * 1.5]

    return pos
def underground_loss(particles):
    loss = particles[:,2].clamp_max(0).abs().sum()
    return loss

def generate_point_cloud(env,m_r_list,m_t_list,intrinsic,mask_list,rgb_list,x,y):
    args = get_args()
    np.random.seed(1)
    torch.manual_seed(1)
    train_set = reconDataset(m_r_list,m_t_list,rgb_list,mask_list,intrinsic)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    gen_net = FCGenerator()
    optimizer = torch.optim.AdamW(gen_net.parameters(),lr = 2e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=15,
                                  threshold=2e-1, verbose=True, min_lr=2e-7)
    gen_net=gen_net.cuda().float()

    for epoch in range(15):
        for i,data in enumerate(train_loader):
            optimizer.zero_grad()
            particles = gen_net().reshape((args.num_particles,3))

            particles[:,0] += x
            particles[:,1] += y
            particles[:,2] += 0.05
            intrinsic = data['intrinsic'].cuda().float()
            m_r_list = data['m_r_list'].cuda().float()
            m_t_list = data['m_t_list'].cuda().float()
            mask_list = data['mask_list'].cuda().float()

            mask_loss = mask_loss_batch(particles,mask_list,intrinsic,m_r_list,m_t_list)
            under_loss = underground_loss(particles)
            loss = mask_loss+under_loss
            #print(under_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_net.parameters(),0.005)
            optimizer.step()
            scheduler.step(loss)
        particles_list = env.show_particle(particles)
        env.hide_particle(particles_list)


    particles = optimize_cloud(particles)
    particles = optimize_cloud(particles)
    print(particles.shape)
    return particles

