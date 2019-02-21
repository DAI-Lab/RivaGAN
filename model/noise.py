import torch
import torch.nn as nn
import torch_dct as dct
from random import random, randint

class Crop(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, min_pct=0.8, max_pct=1.0):
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

    def __init__(self, min_pct=0.8, max_pct=1.0):
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

    def __init__(self, yuv=True, min_pct=0.0, max_pct=0.5):
        super(Compression, self).__init__()
        self.yuv = yuv
        self.min_pct = min_pct
        self.max_pct = max_pct

    def forward(self, y):
        N, _, L, H, W = y.size()
        
        L = int(y.size(2) * (random() * (self.max_pct - self.min_pct) + self.min_pct))
        H = int(y.size(3) * (random() * (self.max_pct - self.min_pct) + self.min_pct))
        W = int(y.size(4) * (random() * (self.max_pct - self.min_pct) + self.min_pct))

        if self.yuv:
            y = torch.stack([
                0.299    * y[:,2,:,:,:] + 0.587    * y[:,1,:,:,:] + 0.114    * y[:,0,:,:,:],
                -0.168736 * y[:,2,:,:,:] - 0.331264 * y[:,1,:,:,:] + 0.500    * y[:,0,:,:,:],
                0.500    * y[:,2,:,:,:] - 0.418688 * y[:,1,:,:,:] - 0.081312 * y[:,0,:,:,:],
            ], dim=1)
        
        y = dct.dct_3d(y)
        if L > 0: y[:,:,-L:,:,:] = 0.0
        if H > 0: y[:,:,:,-H:,:] = 0.0
        if W > 0: y[:,:,:,:,-W:] = 0.0
        y = dct.idct_3d(y)

        if self.yuv:
            y = torch.stack([
                1.000 * y[:,0,:,:,:] + 1.772    * y[:,1,:,:,:] + 0.000    * y[:,2,:,:,:],
                1.000 * y[:,0,:,:,:] - 0.344136 * y[:,1,:,:,:] - 0.714136 * y[:,2,:,:,:],
                1.000 * y[:,0,:,:,:] + 0.000    * y[:,1,:,:,:] + 1.402    * y[:,2,:,:,:],
            ], dim=1)

        return y

if __name__ == "__main__":
    N, L, H, W = 100, 1, 100, 100
    model = Compression()
    x = torch.randn((N, 3, L, H, W))
    y = model(x)
    print(torch.mean(torch.abs(x - y)))
    pass
