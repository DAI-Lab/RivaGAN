import torch
import torch.nn as nn
import torch_dct as dct
from random import random, randint

class Crop(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, min_pct=0.7, max_pct=1.0):
        super(Crop, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct

    def forward(self, frames):
        pct = random() * (self.max_pct - self.min_pct) + self.min_pct

        _, _, _, old_H, old_W = frames.size()
        H, W = int(pct * old_H), int(pct * old_W)

        y = randint(0, old_H - H - 1)
        x = randint(0, old_W - W - 1)
        return frames[:,:,:,y:y+H,x:x+W]

class Scale(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, min_pct=0.7, max_pct=1.0):
        super(Scale, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct

    def forward(self, frames):
        pct = random() * (self.max_pct - self.min_pct) + self.min_pct
        _, _, old_D, old_H, old_W = frames.size()
        H, W = int(pct * old_H), int(pct * old_W)
        return nn.AdaptiveAvgPool3d((old_D, H, W))(frames)
