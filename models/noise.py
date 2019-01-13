import torch
import torch.nn as nn
import torch_dct as dct
from random import random, randint

class Crop(nn.Module):
    """
    Input: (N, 3, W, H)
    Output: (N, 3, W, H)
    """

    def __init__(self, min_pct=0.5, max_pct=1.0):
        super(Scale, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct

    def forward(self, frames):
        pct = random() * (self.max_pct - self.min_pct) + self.min_pct
        n, _, w, h = frames.size()
        crop_w, crop_h = int(pct * w), int(pct * h)
        x = randint(0, w - crop_w - 1)
        y = randint(0, h - crop_h - 1)
        return frames[:,3,x:x+crop_w,y:y+crop_h]

class Scale(nn.Module):
    """
    Input: (N, 3, W, H)
    Output: (N, 3, W, H)
    """

    def __init__(self, min_pct=0.5, max_pct=1.0):
        super(Scale, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct

    def forward(self, frames):
        pct = random() * (self.max_pct - self.min_pct) + self.min_pct
        n, _, w, h = frames.size()
        w, h = int(pct * w), int(pct * h)
        return nn.AdaptiveAvgPool2d((w, h))(frames)
