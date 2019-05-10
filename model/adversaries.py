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

    def __init__(self, l1_max=0.05, kernel_size=(1,3,3), padding=(0,1,1)):
        super(Adversary, self).__init__()
        self.l1_max = l1_max
        self._conv = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 3, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
        )

    def forward(self, frames):
        x = frames
        x = self._conv(x)
        return frames + self.l1_max * x
