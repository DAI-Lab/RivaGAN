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
        self._conv1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(1,7,7), padding=(0,3,3), stride=1),
            nn.Tanh(),
            nn.InstanceNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=(1,7,7), padding=(0,3,3), stride=(1, 2, 2)),
            nn.Tanh(),
            nn.InstanceNorm3d(64),
        )
        self._conv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(1,7,7), padding=(0,3,3), stride=(1, 2, 2), output_padding=(0, 1, 1)),
            nn.Tanh(),
            nn.InstanceNorm3d(32),
            nn.Conv3d(32, 3, kernel_size=(1,7,7), padding=(0,3,3), stride=1),
            nn.Tanh(),
        )

    def forward(self, frames):
        x = frames
        x = self._conv1(x)
        x = self._conv2(x)
        return frames + 0.01 * x
