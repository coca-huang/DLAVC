import torch
import torch.nn as nn
from torch import Tensor

from models.tsm_utils import GatedConv, GatedDeconv


class Generator(nn.Module):
    def __init__(self, feature_num: int, in_channels: int = 4, out_channels: int = 3):
        super(Generator, self).__init__()

        self.conv_1 = GatedConv(in_channels, feature_num, 3, 1, 1)
        self.conv_2 = GatedConv(feature_num * 1, feature_num * 2, 3, (1, 2, 2), 1)
        self.conv_3 = GatedConv(feature_num * 2, feature_num * 4, 3, (1, 2, 2), 1)
        self.conv_4 = GatedConv(feature_num * 2, out_channels, 3, 1, 1)

        self.dilated_conv_1 = GatedConv(feature_num * 4, feature_num * 4, 3, 1, -1, (1, 2, 2))
        self.dilated_conv_2 = GatedConv(feature_num * 4, feature_num * 4, 3, 1, -1, (1, 4, 4))
        self.dilated_conv_3 = GatedConv(feature_num * 4, feature_num * 4, 3, 1, -1, (1, 8, 8))
        self.dilated_conv_4 = GatedConv(feature_num * 4, feature_num * 4, 3, 1, -1, (1, 16, 16))

        self.deconv_1 = GatedDeconv(feature_num * 4 * 2, feature_num * 2, 3, 1, 1)
        self.deconv_2 = GatedDeconv(feature_num * 2 * 2, feature_num, 3, 1, 1)

    @staticmethod
    def preprocess(x: Tensor) -> Tensor:
        return torch.transpose(x, 1, 2)

    @staticmethod
    def postprocess(x: Tensor) -> Tensor:
        return torch.transpose(x, 1, 2)

    def forward(self, x: Tensor) -> Tensor:

        x = self.preprocess(x)
        d_1 = self.conv_1(x)
        d_2 = self.conv_2(d_1)
        d_3 = self.conv_3(d_2)

        c_1 = self.dilated_conv_1(d_3)
        c_2 = self.dilated_conv_2(c_1)
        c_3 = self.dilated_conv_3(c_2)
        c_4 = self.dilated_conv_4(c_3)
        u_1 = self.deconv_1(torch.cat((c_4, d_3), 1))
        u_2 = self.deconv_2(torch.cat((u_1, d_2), 1))
        x = self.conv_4(torch.cat((u_2, d_1), 1))

        return self.postprocess(x)


class Discriminator(nn.Module):
    def __init__(self, feature_num: int, in_channels: int = 4):
        super(Discriminator, self).__init__()

        self.convs = nn.Sequential(
            GatedConv(in_channels, feature_num * 1, (3, 5, 5), (1, 2, 2), 1),
            GatedConv(feature_num * 1, feature_num * 2, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
            GatedConv(feature_num * 2, feature_num * 4, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
            GatedConv(feature_num * 4, feature_num * 4, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
            GatedConv(feature_num * 4, feature_num * 4, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
        )

        self.sig = nn.Sigmoid()

    @staticmethod
    def preprocess(x: Tensor) -> Tensor:
        return torch.transpose(x, 1, 2)

    @staticmethod
    def postprocess(x: Tensor) -> Tensor:
        return torch.transpose(x, 1, 2)

    def forward(self, x: Tensor) -> Tensor:

        x = self.preprocess(x)
        x = self.convs(x)
        x = self.sig(x)

        return self.postprocess(x)
