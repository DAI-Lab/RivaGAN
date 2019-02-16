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
            nn.Conv3d(3, 32, kernel_size=(1,7,7), padding=(0,3,3), stride=1),
            nn.Tanh(),
            nn.InstanceNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=(1,7,7), padding=(0,3,3), stride=(1, 2, 2)),
            nn.Tanh(),
            nn.InstanceNorm3d(64),
            nn.Conv3d(64, 128, kernel_size=(1,7,7), padding=(0,3,3), stride=(1, 2, 2)),
            nn.Tanh(),
            nn.InstanceNorm3d(128),
        )
        self._linear = nn.Linear(256, self.data_dim)

    def forward(self, frames):
        frames = self._conv(frames)
        N, _, L, H, W = frames.size()
        return self._linear(torch.cat([
            torch.max(frames.view(N, -1, L*H*W), dim=2)[0],
            torch.min(frames.view(N, -1, L*H*W), dim=2)[0],
        ], dim=1))
