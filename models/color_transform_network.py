import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from models.basic_block import ConvBnReLU2d, AdaINResBlocks


class Encoder(nn.Module):
    def __init__(self, in_channels: int):
        super(Encoder, self).__init__()

        self.convs = nn.Sequential(
            ConvBnReLU2d(in_channels, 64, mode="same"),
            ConvBnReLU2d(64, 128),
            ConvBnReLU2d(128, 256),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.convs(x)
        return x


class Embedder(nn.Module):
    def __init__(self, ref_num: int):
        super(Embedder, self).__init__()

        self.convs = nn.Sequential(
            ConvBnReLU2d(4, 64, 'same'),
            nn.ReLU(),
            ConvBnReLU2d(64, 128),
            nn.ReLU(),
            ConvBnReLU2d(128, 256),
            nn.ReLU(),
            ConvBnReLU2d(256, 512),
            nn.ReLU(),
            ConvBnReLU2d(512, 512),
            nn.ReLU(),
            nn.AvgPool2d(16),
        )

        self.avg = nn.AvgPool2d(16)
        self.fc = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 512))

        self.ref_num = ref_num

    def forward(self, lines: Tensor, refs: Tensor) -> Tensor:
        x = [self.convs(torch.cat((lines[i], refs[i]), dim=1)) for i in range(self.ref_num)]  # 2 * [1, 512, 32, 32]
        x = torch.mean(torch.stack(x), 0).squeeze()  # [512, 32, 32]
        x = self.fc(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels: int):
        super(Decoder, self).__init__()

        self.convs = nn.Sequential(
            ConvBnReLU2d(in_channels, 256, 'same'),
            ConvBnReLU2d(256, 128, 'up'),
            ConvBnReLU2d(128, 64, 'up'),
            ConvBnReLU2d(64, 3, 'same'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.convs(x)
        return x


class SBCTL(nn.Module):
    def __init__(self, ref_num: int):
        super(SBCTL, self).__init__()

        self.conv_m = ConvBnReLU2d(512, 256, 'same', True)
        self.conv_n = ConvBnReLU2d(512, 256, 'same', True)
        self.sig = nn.Sigmoid()

        self.l_conv = nn.ModuleList()
        for i in range(ref_num):
            self.l_conv.append(nn.ModuleList([
                nn.Conv2d(256, 32, (1, 1)),
                nn.Conv2d(256, 32, (1, 1)),
            ]))
        self.r_conv = nn.ModuleList()
        for i in range(ref_num):
            self.r_conv.append(nn.Conv2d(256, 32, (1, 1)))
        self.c_conv = nn.Conv2d(32, 256, (1, 1))

        self.ref_num = ref_num

    def forward(self, f_d: Tensor, f_ds: Tensor, f_ys: Tensor) -> Tensor:
        m = []
        bs = f_d.shape[0]
        for i in range(self.ref_num):
            i_t = f_ds[i]  # [bs, 256, 64]
            i_t = self.l_conv[i][0](i_t)  # [bs, 32, 64, 64]
            i_t = torch.reshape(i_t, (bs, i_t.shape[1], -1))  # [bs, 32, 4096]

            t = self.l_conv[i][1](f_d)  # [bs, 32, 64, 64]
            t = torch.reshape(t, (bs, t.shape[1], -1))  # [bs, 32, 4096]
            t = t.permute(0, 2, 1)  # [bs, 4096, 32]

            m.append(torch.matmul(t, i_t))

        m = torch.cat(m, dim=1)
        m_i = []
        n_i = []
        for i in range(self.ref_num):
            x = torch.cat((f_d, f_ds[i]), dim=1)
            m_i.append(self.sig(self.conv_m(x)))
            n_i.append(self.sig(self.conv_n(x)))

        c_i = []
        for i in range(self.ref_num):
            i_t = f_ys[i]
            i_t = self.r_conv[i](torch.mul(i_t, m_i[i]))
            i_t = torch.reshape(i_t, (bs, i_t.shape[1], -1))
            c_i.append(i_t)

        c = torch.cat(c_i, dim=2)
        f_m = torch.matmul(c, m)
        f_m = torch.reshape(f_m, (bs, 32, 64, 64))
        f_m = self.c_conv(f_m)
        f_sim = [torch.mul(f_m, 1 - n_i[i]) + torch.mul(n_i[i], f_ys[i]) for i in range(self.ref_num)]
        f_sim = torch.mean(torch.stack(f_sim), 0)

        return f_sim


class LatentDecoder(nn.Module):
    def __init__(self, in_channels: int):
        super(LatentDecoder, self).__init__()
        self.conv = ConvBnReLU2d(in_channels, 3, 'same')

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Generator(nn.Module):
    def __init__(self, ref_num: int):
        super(Generator, self).__init__()
        self.encoder_c = Encoder(3)
        self.encoder_l = Encoder(1)
        self.encoder_d = Encoder(1)
        self.sbctl = SBCTL(ref_num)
        self.embedder = Embedder(ref_num)
        self.adain_res = AdaINResBlocks(256)
        self.decoder = Decoder(256)
        self.sim_decoder = LatentDecoder(256)
        self.mid_decoder = LatentDecoder(256)

    @staticmethod
    def assign_adain_params(adain_params, model):
        for m in model.modules():
            if m.__class__.__name__ == 'AdaptiveInstanceNorm2d':
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def forward(self, x: Tensor, d: Tensor, d_rs: Tensor, y_rs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        f_d = self.encoder_d(d)  # [256, 64, 64] distance field map of original image
        f_l = self.encoder_l(x)
        f_ds = [self.encoder_d(d_r) for d_r in d_rs]  # [2, 256, 64, 64]
        f_ys = [self.encoder_c(y_r) for y_r in y_rs]  # [2, 256, 64, 64]
        f_sim = self.sbctl(f_d, f_ds, f_ys)
        p_em = self.embedder(d_rs, y_rs)
        p_em = torch.reshape(p_em, (-1, 512))
        self.assign_adain_params(p_em, self.adain_res)
        f_mid = self.adain_res(torch.add(f_l, f_sim))
        y_sim = self.sim_decoder(f_sim)
        y_mid = self.mid_decoder(f_mid)
        y_trans = self.decoder(f_mid)
        return y_trans, y_sim, y_mid


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 4):
        super(Discriminator, self).__init__()

        self.convs = nn.Sequential(
            ConvBnReLU2d(in_channels, 64, 'same'),
            ConvBnReLU2d(64, 128),
            ConvBnReLU2d(128, 256),
            ConvBnReLU2d(256, 512),
            ConvBnReLU2d(512, 512),
        )

    def forward(self, x: Tensor, y_trans: Tensor) -> Tensor:
        x = torch.cat((x, y_trans), dim=1)
        return self.convs(x)
