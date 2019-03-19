import torch
import torch.nn as nn
import torch.nn.functional as F

class Adversary(nn.Module):
    """
    The Adversary module maps a sequence of frames to another sequence of frames
    with a constraint on the maximum distortion of each individual pixel.

    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, l1_max=0.016):
        super(Adversary, self).__init__()
        self.l1_max = l1_max
        self._conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(1,7,7), padding=(0,3,3)),
            nn.ELU(inplace=True),
            nn.InstanceNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=(1,7,7), padding=(0,3,3)),
            nn.ELU(inplace=True),
            nn.InstanceNorm3d(64),
            nn.Conv3d(64, 3, kernel_size=(1,7,7), padding=(0,3,3)),
            nn.Tanh(),
        )

    def forward(self, frames):
        x = frames
        x = self._conv(x)
        return frames + self.l1_max * x
