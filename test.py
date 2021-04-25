import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.dataset import Dataset
import models.color_transform_network as col
import models.temporal_constraint_network as tem
from tqdm import tqdm
import os
import time
import statistics
from tools.imsave import imsave
import yaml
import logging.config

# load config
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# load logger
lc = config['environment']['log_config']
logging.config.fileConfig(lc)
logs = logging.getLogger()

# load device config
cuda = config['environment']['cuda']
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load dataloader
it = config['test']['image_root']
bs = 1
iz = None
data = Dataset(it, iz, cuda)
loader = DataLoader(data, bs, cuda)

# load color transform network
net_col = col.Generator(2)
net_col = nn.DataParallel(net_col)
net_col = net_col.cuda() if cuda else net_col

# load temporal constraint network
net_tem = tem.Generator(64)
net_tem = nn.DataParallel(net_tem)
net_tem = net_tem.cuda() if cuda else net_tem

# load pretrained models
# col_gen.load_state_dict(torch.load(test['load_pretrain_model'][0], map_location='cpu'))
# tem_gen.load_state_dict(torch.load(test['load_pretrain_model'][1], map_location='cpu'))

# load cuda
if cuda:
    net_tem = net_tem.cuda()
    net_col = net_col.cuda()

# start testing...
rf = config['test']['result_folder']
tr = []
for [y_s, y_e, y], [l_s, l_e, x], [d_s, d_e, d_x], n in tqdm(loader):
    st = time.time()
    y_trans, y_sim, y_mid = net_col(x, d_x, [d_s, d_e], [y_s, y_e])
    input_tem = torch.cat(
        (torch.cat((l_s, y_s), dim=1).unsqueeze(1), torch.cat(
            (x, y_trans), dim=1).unsqueeze(1), torch.cat((d_e, y_e), dim=1).unsqueeze(1)),
        dim=1,
    )
    pre_tem = net_tem(input_tem)

    et = time.time()
    tr.append(et - st)
    imsave(os.path.join(rf, n), pre_tem[0][1])

# time record
s = statistics.stdev(tr[1:])
m = statistics.mean(tr[1:])
print('std time: {} \t mean time: {}'.format(s, m))
