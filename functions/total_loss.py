import torch
import torch.nn as nn

from functions.vgg19 import VGG19


class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()

        self.adv_loss = AdversarialLoss()
        self.l1_loss = nn.L1Loss()
        self.per_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.latent_loss = LatentLoss()


def __call__(self, y_pre, y_sim, y_mid, y):
    al = 1 * self.adv_loss(y_pre, True)
    l1 = 10 * self.l1_loss(y_pre, y)
    pl = self.per_loss(y_pre, y)
    sl = 1000 * self.style_loss(y_pre, y)
    ll = 1 * self.latent_loss(y_sim, y_mid, y)
    return al + l1 + pl + sl + ll


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def __call__(self, l_pre, l):
        label = torch.ones_like(l_pre) if l else torch.zeros_like(l_pre)
        loss = self.criterion(l_pre, label)
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self, weights=None):
        super(PerceptualLoss, self).__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def forward(self, y_pre, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(y_pre), self.vgg(y)

        loss = 0.0
        loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return loss


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    @staticmethod
    def compute_gram(x):
        bt, c, h, w = x.size()
        f = x.view(bt, c, w * h)
        f_tran = f.transpose(1, 2)
        gram = f.bmm(f_tran) / (h * w * c)

        return gram

    def forward(self, y_pre, y):

        # Compute features
        x_vgg, y_vgg = self.vgg(y_pre), self.vgg(y)

        # Compute loss
        loss = 0.0
        loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return loss


class LatentLoss(nn.Module):
    def __init__(self):
        super(LatentLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.max_pool = nn.MaxPool2d(4)

    def forward(self, y_sim, y_mid, y):
        y = self.max_pool(y)
        l_1 = self.loss(y_sim, y)
        l_2 = self.loss(y_mid, y)
        return l_1 + l_2
