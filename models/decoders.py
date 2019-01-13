import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import _conv_3d

class Decoder(nn.Module):
    """
    Given a sequence of watermarked frames, try to decode the watermark.

    Input: (N, 3, W, H)
    Output: (D,)
    """

    def __init__(self, data_dim, hidden_size=32):
        super(Decoder, self).__init__()
        self.data_dim = data_dim
        self._preprocess = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=4, padding=2, stride=2),
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
            nn.Conv2d(hidden_size*3, data_dim, kernel_size=3, padding=1),
        )

    def forward(self, frames):
        a = self._preprocess(frames)
        b = _conv_3d(self._conv1, a)
        c = self._conv2(torch.cat([a, b], dim=1))
        d = self._conv3(torch.cat([a, b, c], dim=1))
        data = torch.mean(torch.mean(torch.mean(d[1:-1,:,:,:], dim=3), dim=2), dim=0)
        return data
