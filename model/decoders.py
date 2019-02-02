import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, D)
    """

    def __init__(self, data_dim):
        super(Decoder, self).__init__()
        self.data_dim = data_dim
        self._conv = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(1,5,5), padding=(0,2,2), stride=1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(16),
            nn.Conv3d(16, 32, kernel_size=(1,5,5), padding=(0,2,2), stride=(1, 2, 2)),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(32),
            nn.Conv3d(32, 48, kernel_size=(1,5,5), padding=(0,2,2), stride=1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(48),
            nn.Conv3d(48, 64, kernel_size=(3,5,5), padding=(1,2,2), stride=1),
        )
        self._linear = nn.Linear(64, data_dim)

    def forward(self, frames):
        frames = self._conv(frames)
        N, _, L, H, W = frames.size()
        return self._linear(torch.mean(frames.view(N, -1, L*H*W), dim=2))
