import torch.nn as nn
import models.color_transform_network as col
import models.temporal_constraint_network as tem
import yaml
from torch.utils.data import DataLoader
import os
import torch
from functions.total_loss import AdversarialLoss, TotalLoss
from models.dataset import Dataset
import logging.config

# load config
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# load logger
lc = config['environment']['log_config']
logging.config.fileConfig(lc)
logs = logging.getLogger()

# load snapshots folder
sf = config['train']['snapshots_folder']
if not os.path.exists(sf):
    os.makedirs(sf)

# load device config
cuda = config['environment']['cuda']
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load dataloader
it = config['train']['image_root']
iz = config['train']['image_size']
bs = config['train']['batch_size']
data = Dataset(it, iz, cuda)
loader = DataLoader(data, bs, cuda)

# load color transform network
net_col_g = col.Generator(2)
net_col_g = nn.DataParallel(net_col_g)
net_col_g = net_col_g.cuda() if cuda else net_col_g

net_col_d = col.Discriminator()
net_col_d = nn.DataParallel(net_col_d)
net_col_d = net_col_d.cuda() if cuda else net_col_d

# load temporal constraint network
net_tem_g = tem.Generator(64)
net_tem_g = nn.DataParallel(net_tem_g)
net_tem_g = net_tem_g.cuda() if cuda else net_tem_g

net_tem_d = tem.Discriminator(64)
net_tem_d = nn.DataParallel(net_tem_d)
net_tem_d = net_tem_d.cuda() if cuda else net_tem_d

# load loss
col_loss = AdversarialLoss()
tem_loss = AdversarialLoss()
tot_loss = TotalLoss()

# load optimizer
lr = config['train']['learning_rate']
wd = config['train']['weight_decay']
opt_col_g = torch.optim.Adam(net_col_g.parameters(), lr, weight_decay=wd)
opt_col_d = torch.optim.Adam(net_col_d.parameters(), lr, weight_decay=wd)
opt_tem_g = torch.optim.Adam(net_tem_g.parameters(), lr, weight_decay=wd)
opt_tem_d = torch.optim.Adam(net_tem_d.parameters(), lr, weight_decay=wd)

if config['train']['resume']:
    net_col_g.load_state_dict(torch.load(config['train']['load_pretrain_model'][0], map_location='cpu'))
    net_tem_g.load_state_dict(torch.load(config['train']['load_pretrain_model'][1], map_location='cpu'))
    net_col_d.load_state_dict(torch.load(config['train']['load_pretrain_model'][2], map_location='cpu'))
    net_tem_d.load_state_dict(torch.load(config['train']['load_pretrain_model'][3], map_location='cpu'))

# load cuda
if cuda:
    tem_loss = tem_loss.cuda()
    tot_loss = tot_loss.cuda()

# start training...
print("Starting Training Loop...")
max_epoch = config['train']['max_epoch']
for epoch in range(max_epoch):
    for count, [[y_s, y_e, y], [l_s, l_e, x], [d_s, d_e, d_x], n] in enumerate(loader):
        net_col_g.train()
        net_col_d.train()
        net_tem_g.train()
        net_tem_d.train()

        net_col_d.zero_grad()
        net_tem_d.zero_grad()

        y_trans, _, _ = net_col_g(x, d_x, [d_s, d_e], [y_s, y_e])
        y_trans.detach()
        input_tem = torch.cat(
            (torch.cat((l_s, y_s), dim=1).unsqueeze(1), torch.cat(
                (x, y_trans), dim=1).unsqueeze(1), torch.cat((l_e, y_e), dim=1).unsqueeze(1)),
            dim=1,
        )
        pre_tem = net_tem_g(input_tem)

        real_y = torch.cat((y_trans.unsqueeze(1), y_s.unsqueeze(1), y_e.unsqueeze(1)), dim=1)
        real_x = torch.cat((x.unsqueeze(1), l_s.unsqueeze(1), l_e.unsqueeze(1)), dim=1)
        real = torch.cat((real_x, real_y), dim=2)

        col_d_pre = net_col_d(x, y_trans)
        col_d_real = net_col_d(x, y)
        col_adv_loss = col_loss(col_d_real, True) + col_loss(col_d_pre, False)

        tem_d_pre = net_tem_d(torch.cat((real_x, pre_tem), dim=2))
        tem_d_real = net_tem_d(real)
        tem_adv_loss = tem_loss(tem_d_real, True) + tem_loss(tem_d_pre, False)

        loss_d = col_adv_loss + tem_adv_loss
        loss_d.backward()

        opt_col_d.step()
        opt_tem_d.step()

        net_col_g.zero_grad()
        net_tem_g.zero_grad()

        pre_tem = net_tem_g(input_tem)
        input_tem_dis = torch.cat((real_x, pre_tem), dim=2)
        y_trans, y_sim, y_mid = net_col_g(x, d_x, [d_s, d_e], [y_s, y_e])

        col_d_pre = net_col_d(d_x, y_trans)
        tem_d_pre = net_tem_d(input_tem_dis)
        col_d_pre.detach()
        tem_d_pre.detach()

        loss_adv = tem_loss(tem_d_pre, True) + col_loss(col_d_pre, True)

        y_pre = torch.split(pre_tem, 1, dim=1)[1].squeeze(dim=1)
        loss_tot = tot_loss(y_pre, y_sim, y_mid, y)

        loss_g = loss_tot + loss_adv
        loss_g.backward()

        opt_col_g.step()
        opt_tem_d.step()

    t = os.path.join(sf, 'epoch_col_g_{}.pth'.format(str(epoch).zfill(3)))
    torch.save(net_col_g.state_dict(), t)
    t = os.path.join(sf, 'epoch_col_d_{}.pth'.format(str(epoch).zfill(3)))
    torch.save(net_col_d.state_dict(), t)
    t = os.path.join(sf, 'epoch_tem_g_{}.pth'.format(str(epoch).zfill(3)))
    torch.save(net_tem_g.state_dict(), t)
    t = os.path.join(sf, 'epoch_tem_d_{}.pth'.format(str(epoch).zfill(3)))
    torch.save(net_tem_d.state_dict(), t)
