import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """
    The Critic module maps a sequence of frames to a scalar score.
    Input: (N, 3, L, H, W)
    Output: (N, 1)
    """

    def __init__(self, kernel_size=(1,3,3), padding=(0,0,0)):
        super(Critic, self).__init__()
        self._conv = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=kernel_size, padding=padding, stride=2),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 32, kernel_size=kernel_size, padding=padding, stride=2),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=kernel_size, padding=padding, stride=2),
        )
        self._linear = nn.Linear(64, 1)

    def forward(self, frames):
        frames = self._conv(frames)
        N, _, L, H, W = frames.size()
        return self._linear(torch.mean(frames.view(N, -1, L*H*W), dim=2))
