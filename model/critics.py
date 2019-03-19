import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """
    The Critic module maps a sequence of frames to a scalar score.

    Input: (N, 3, L, H, W)
    Output: (N, 1)
    """

    def __init__(self):
        super(Critic, self).__init__()
        self._conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(1,7,7)),
            nn.ELU(inplace=True),
            nn.InstanceNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=(1,7,7)),
            nn.ELU(inplace=True),
            nn.InstanceNorm3d(64),
            nn.Conv3d(64, 128, kernel_size=(1,7,7)),
            nn.ELU(inplace=True),
            nn.InstanceNorm3d(128),
        )
        self._linear = nn.Linear(128, 1)

    def forward(self, frames):
        frames = self._conv(frames)
        N, _, L, H, W = frames.size()
        return self._linear(torch.mean(frames.view(N, -1, L*H*W), dim=2))
