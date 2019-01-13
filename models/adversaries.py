import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import _conv_3d

class Adversary(nn.Module):
    """
    Given a sequence of watermarked frames, try to strip the watermark out.

    Input: (N, 3, W, H)
    Output: (N, 3, W, H)
    """

    def __init__(self, hidden_size=32):
        super(Adversary, self).__init__()
        self._preprocess = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(hidden_size),
        )
        self._conv1 = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(hidden_size),
        )
        self._conv2 = nn.Sequential(
            nn.Conv2d(hidden_size*2, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(hidden_size),
        )
        self._conv3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(hidden_size*3, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, frames):
        a = self._preprocess(frames)
        b = _conv_3d(self._conv1, a)
        c = self._conv2(torch.cat([a, b], dim=1))
        d = self._conv3(torch.cat([a, b, c], dim=1))
        return frames + d / 10.0
