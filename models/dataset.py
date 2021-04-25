import torch
import os

import torch.utils.data as data
import torchvision

from tools.imread import imread


def get_all_images(path: str):
    li = []
    for par, _, names in os.walk(path):
        for name in names:
            if name.lower().endswith(('.bmp', '.jpg', '.png')):
                li.append(os.path.join(par, name))
    li.sort()
    return li


class Dataset(data.Dataset):
    def __init__(self, image_root: str, image_size: list = None, cuda: bool = True):
        super(Dataset, self).__init__()

        r_path = image_root
        gt_path = os.path.join(r_path, 'GT')

        video_list = []
        for root, dirs, files in os.walk(gt_path):
            video_list += dirs

        meta_list = []
        for v in video_list:
            p = os.path.join(gt_path, v)
            fs = get_all_images(p)
            for f in fs[1:-1]:
                m = {'video': v, 'start': fs[0], 'end': fs[-1], 'cur': f}
                meta_list.append(m)

        self.meta_list = meta_list
        self.resize = torchvision.transforms.Resize(image_size) if image_size else None
        self.cuda = cuda

    def get_group_images(self, s, e, c):

        cuda = self.cuda
        r = self.resize

        y_s, y_e, y = imread(s, cuda), imread(e, cuda), imread(c, cuda)
        t = torch.stack([y_s, y_e, y], dim=0)
        t = r(t) if r else t
        return t[0], t[1], t[2]

    def __getitem__(self, index: int):

        m = self.meta_list[index]
        s, e, c = m['start'], m['end'], m['cur']
        y_s, y_e, y = self.get_group_images(s, e, c)

        s, e, c = [x.replace('GT', 'LI') for x in [s, e, c]]
        l_s, l_e, x = self.get_group_images(s, e, c)

        s, e, c = [x.replace('LI', 'DI') for x in [s, e, c]]
        d_s, d_e, d_x = self.get_group_images(s, e, c)

        v = m['video']
        n = c.split('/')[-1]
        n = os.path.join(v, n)

        return [y_s, y_e, y], [l_s, l_e, x], [d_s, d_e, d_x], n

    def __len__(self):
        return len(self.meta_list)
