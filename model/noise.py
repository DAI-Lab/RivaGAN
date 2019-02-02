import torch
import torch.nn as nn
import torch_dct as dct
from random import random, randint

class Crop(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, min_pct=0.9, max_pct=1.0):
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

    def __init__(self, min_pct=0.9, max_pct=1.0):
        super(Scale, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct

    def forward(self, frames):
        pct = random() * (self.max_pct - self.min_pct) + self.min_pct
        _, _, old_D, old_H, old_W = frames.size()
        H, W = int(pct * old_H), int(pct * old_W)
        return nn.AdaptiveAvgPool3d((old_D, H, W))(frames)

class Compression(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, min_pct=0.0, max_pct=0.1):
        super(Compression, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct

    def forward(self, frames):
        N, _, L, H, W = frames.size()
        
        L = int(frames.size(2) * (random() * (self.max_pct - self.min_pct) + self.min_pct))
        H = int(frames.size(3) * (random() * (self.max_pct - self.min_pct) + self.min_pct))
        W = int(frames.size(4) * (random() * (self.max_pct - self.min_pct) + self.min_pct))
        
        y = dct.dct_3d(frames)
        if L > 0: y[:,:,-L:,:,:] = 0.0
        if H > 0: y[:,:,:,-H:,:] = 0.0
        if W > 0: y[:,:,:,:,-W:] = 0.0
        y = dct.idct_3d(y)

        return y
