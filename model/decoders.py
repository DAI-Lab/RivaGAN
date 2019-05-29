import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, D)
    """

    def __init__(self, data_dim, kernel_size=(1,11,11)):
        super(Decoder, self).__init__()
        self.data_dim = data_dim
        self._conv = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=kernel_size),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=kernel_size),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=kernel_size),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=kernel_size),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=kernel_size),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=kernel_size),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=kernel_size),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, self.data_dim, kernel_size=kernel_size),
        )
        self._linear = nn.Linear(self.data_dim, self.data_dim)

    def forward(self, frames):
        frames = self._conv(frames)
        N, C, L, H, W = frames.size()
        data = torch.mean(frames.view(N, self.data_dim, -1), dim=2)
        return self._linear(data)
