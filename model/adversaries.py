import torch
import torch.nn as nn
import torch.nn.functional as F

class Adversary(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self):
        super(Adversary, self).__init__()
        self._conv = nn.Sequential(
            nn.Conv3d(3, 3, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
        )

    def forward(self, frames):
        x = frames
        x = torch.cat([
            x[:,:,:1],
            x,
            x[:,:,-1:],
        ], dim=2)
        x = self._conv(x)
        return frames + x[:,:,1:-1,:,:]
