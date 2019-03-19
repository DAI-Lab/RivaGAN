import torch
import torch.nn as nn
import torch.nn.functional as F

def spatial_repeat(x, data):
    """
    This function takes a 5d tensor (with the same shape and dimension order
    as the input to Conv3d) and a 2d data tensor. For each element in the 
    batch, the data vector is replicated spatially/temporally and concatenated
    to the channel dimension.
    
    Input: (N, C_{in}, L, H, W), (N, D)
    Output: (N, C_{in} + D, L, H, W)
    """
    N, D = data.size()
    N, _, L, H, W = x.size()
    x = torch.cat([
        x,
        data.view(N, D, 1, 1, 1).expand(N, D, L, H, W)
    ], dim=1)
    return x

class Encoder(nn.Module):
    """
    The Encoder module maps a sequence of frames and a fixed-length bit 
    vector to another sequence of frames with a constraint on the maximum 
    distortion of each individual pixel.

    Input: (N, 3, L, H, W), (N, D,)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, data_dim, l1_max=0.016):
        super(Encoder, self).__init__()
        self.l1_max = l1_max
        self.data_dim = data_dim
        self._conv1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(1,15,15), padding=(0,7,7)),
            nn.ELU(inplace=True),
            nn.InstanceNorm3d(32),
        )
        self._conv2 = nn.Sequential(
            nn.Conv3d(32+data_dim, 64, kernel_size=(1,15,15), padding=(0,7,7)),
            nn.ELU(inplace=True),
            nn.InstanceNorm3d(64),
        )
        self._conv3 = nn.Sequential(
            nn.Conv3d(64+data_dim, 128, kernel_size=(1,15,15), padding=(0,7,7)),
            nn.ELU(inplace=True),
            nn.InstanceNorm3d(128),
        )
        self._conv4 = nn.Sequential(
            nn.Conv3d(128+data_dim, 3, kernel_size=(1,15,15), padding=(0,7,7)),
            nn.Tanh(),
        )

    def forward(self, frames, data):
        data = data * 2.0 - 1.0
        x = self._conv1(frames)
        x = self._conv2(spatial_repeat(x, data))
        x = self._conv3(spatial_repeat(x, data))
        x = self._conv4(spatial_repeat(x, data))
        return frames + self.l1_max * x
