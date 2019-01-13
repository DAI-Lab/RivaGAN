import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import _conv_3d

class Encoder(nn.Module):
    """
    Given a sequence of frames and a watermark, try to embed watermark.

    Input: (N, 3, W, H), (D,)
    Output: (N, 3, W, H)
    """

    def __init__(self, data_dim, hidden_size=32):
        super(Encoder, self).__init__()
        self.data_dim = data_dim
        self._preprocess = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(hidden_size),
        )
        self._conv1 = nn.Sequential(
            nn.Conv3d(hidden_size+data_dim, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(hidden_size),
        )
        self._conv2 = nn.Sequential(
            nn.Conv2d(hidden_size*2+data_dim, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(hidden_size),
        )
        self._conv3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(hidden_size*3+data_dim, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, frames, data):
        frames = torch.cat([ # copy first and last frames as padding
            frames[:1],
            frames,
            frames[-1:],
        ], dim=0)

        a = self._preprocess(frames)
        N, _, H, W = a.size()
        data = data.unsqueeze(1).unsqueeze(1).unsqueeze(0).expand(N, -1, H, W)
        b = _conv_3d(self._conv1, torch.cat([a, data], dim=1))
        c = self._conv2(torch.cat([a, b, data], dim=1))
        d = self._conv3(torch.cat([a, b, c, data], dim=1))
        return frames[1:-1,:,:,:] + d[1:-1,:,:,:] / 10.0
