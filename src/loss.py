"""Loss functions
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import grad
from torch.cuda.amp import GradScaler, autocast


class NonSaturating:
    def real_loss(self, real_logits):
        return F.softplus(-real_logits).mean()

    def fake_loss(self, fake_logits):
        return F.softplus(fake_logits).mean()

    def d_loss(self, real_logits, fake_logits, return_all=False):
        real_loss = self.real_loss(real_logits)
        fake_loss = self.fake_loss(fake_logits)
        loss = real_loss + fake_loss
        if return_all:
            return loss, real_loss, fake_loss
        return loss

    def g_loss(self, fake_logits):
        return self.real_loss(fake_logits)


class LeastSquare:
    def real_loss(self, real_logits):
        target = torch.ones(real_logits.size(), device=real_logits.device)
        return F.mse_loss(real_logits, target)

    def fake_loss(self, fake_logits):
        target = torch.zeros(fake_logits.size(), device=fake_logits.device)
        return F.mse_loss(fake_logits, target)

    def d_loss(self, real_logits, fake_logits, return_all=False):
        real_loss = self.real_loss(real_logits)
        fake_loss = self.fake_loss(fake_logits)
        loss = real_loss + fake_loss
        if return_all:
            return loss, real_loss, fake_loss
        return loss

    def g_loss(self, fake_logits):
        return self.real_loss(fake_logits)


@autocast(enabled=False)
def calc_grad(inputs, outputs, scaler: Optional[GradScaler] = None):
    """Calc gradient of inputs.
    Works with native pytorch AMP.
    """
    if isinstance(scaler, GradScaler):
        outputs = scaler.scale(outputs)
    ones = torch.ones(outputs.size(), device=outputs.device)
    gradients = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    if isinstance(scaler, GradScaler):
        gradients = gradients / scaler.get_scale()
    return gradients


def r1_regularize(real, D, scaler: Optional[GradScaler] = None):
    """R1 regularizer"""
    real = real.clone()
    real.requires_grad_(True)

    output = D(real).sum()
    gradients = calc_grad(real, output, scaler)
    gradients = gradients.reshape(gradients.size(0), -1)

    penalty = gradients.norm(2, dim=1).pow(2).mean() / 2.0
    return penalty


class VGG(nn.Module):
    """VGG with only feature layers"""

    def __init__(self, layers: int = 16, pretrained: bool = True):
        super().__init__()
        assert layers in [16, 19], "only supports VGG16 and VGG19"
        if layers == 16:
            vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
            slices = [4, 9, 16, 23, 30]
        if layers == 19:
            vgg_pretrained_features = models.vgg19(pretrained=pretrained).features
            slices = [4, 9, 18, 27, 36]

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(slices[0]):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slices[0], slices[1]):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slices[1], slices[2]):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slices[2], slices[3]):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slices[3], slices[4]):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        return h_relu1, h_relu2, h_relu3, h_relu4, h_relu5


def gram_matrix(x):
    B, C, H, W = x.size()
    feat = x.reshape(B, C, H * W)
    G = torch.bmm(feat, feat.permute(0, 2, 1))
    return G.div(C * H * W)


class VGGLoss:
    """loss using vgg"""

    def __init__(
        self,
        device,
        vgg: int = 16,
    ) -> None:
        self.vgg = VGG(vgg, pretrained=True)
        self.vgg.to(device)

    @autocast(enabled=False)
    def style_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """style loss introduced in
        "Perceptual Losses for Real-Time Style Transfer and Super-Resolution",
        Justin Johnson, Alexandre Alahi, and Li Fei-Fei
        """
        loss = 0
        real_acts = self.vgg(real)
        fake_acts = self.vgg(fake)
        for index in [0, 1, 2, 3]:
            loss = loss + F.l1_loss(
                gram_matrix(fake_acts[index]), gram_matrix(real_acts[index])
            )

        return loss
