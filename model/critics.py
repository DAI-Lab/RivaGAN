import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """
    The Critic module maps a sequence of frames to a scalar score.
    Input: (N, 3, L, H, W)
    Output: (N, 1)
    """

    def __init__(self, kernel_size=(1,11,11), padding=(0,5,5)):
        super(Critic, self).__init__()
        self._conv = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
        )
        self._linear = nn.Linear(32, 1)

    def forward(self, frames):
        frames = self._conv(frames)
        N, _, L, H, W = frames.size()
        return self._linear(torch.mean(frames.view(N, -1, L*H*W), dim=2))
