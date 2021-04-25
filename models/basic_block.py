import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor


class ConvBnReLU2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str = 'half',
        pure: bool = False,
        norm: str = 'bn',
        act: str = 'relu',
    ):
        super(ConvBnReLU2d, self).__init__()
        k = (4, 4) if mode == 'half' else (3, 3)
        s = (2, 2) if mode == 'half' else (1, 1)
        p = (1, 1)
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p)
        self.bn = nn.BatchNorm2d(out_channels) if norm == 'bn' else AdaptiveInstanceNorm2d(out_channels)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        if act == 'relu':
            self.activate = nn.ReLU()
        elif act == 'sig':
            self.activate = nn.Sigmoid()
        self.mode = mode
        self.pure = pure

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x) if not self.pure else x
        x = self.up(x) if self.mode == 'up' else x
        x = self.activate(x) if not self.pure else x
        return x


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(ConvBnReLU2d(dim, dim, 'same', norm='adapt'))

    def forward(self, x: Tensor) -> Tensor:
        t = self.model(x)
        t += x
        return t


class AdaINResBlocks(nn.Module):
    def __init__(self, dim: int, block_nums: int = 8):
        super(AdaINResBlocks, self).__init__()

        self.model = []
        for i in range(block_nums):
            self.model += [ResBlock(dim)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = functional.batch_norm(
            x_reshaped,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )

        return out.view(b, c, *x.size()[2:])
