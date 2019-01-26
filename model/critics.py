import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, 1)
    """

    def __init__(self):
        super(Critic, self).__init__()
        self._conv = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(16),
            nn.Conv3d(16, 32, kernel_size=3, padding=(1, 1, 1), stride=(1, 2, 2)),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(32),
            nn.Conv3d(32, 48, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(48),
            nn.Conv3d(48, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(64),
        )
        self._linear = nn.Linear(64, 1)

    def forward(self, frames):
        frames = self._conv(frames)
        N, _, L, H, W = frames.size()
        return self._linear(torch.mean(frames.view(N, -1, L*H*W), dim=2))
