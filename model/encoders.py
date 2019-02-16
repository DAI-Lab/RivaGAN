import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Input: (N, 3, L, H, W), (N, D,)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, data_dim):
        super(Encoder, self).__init__()
        self.data_dim = data_dim
        self._conv1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(1,7,7), padding=(0,3,3), stride=1),
            nn.Tanh(),
            nn.InstanceNorm3d(32),
        )
        self._conv2 = nn.Sequential(
            nn.Conv3d(32+data_dim, 64, kernel_size=(1,7,7), padding=(0,3,3), stride=(1, 2, 2)),
            nn.Tanh(),
            nn.InstanceNorm3d(64),
        )
        self._conv3 = nn.Sequential(
            nn.ConvTranspose3d(64+data_dim, 32, kernel_size=(1,7,7), padding=(0,3,3), stride=(1, 2, 2), output_padding=(0, 1, 1)),
            nn.Tanh(),
            nn.InstanceNorm3d(32),
        )
        self._conv4 = nn.Sequential(
            nn.Conv3d(32+data_dim, 3, kernel_size=(1,1,1), padding=(0,0,0), stride=1),
            nn.Tanh(),
        )

    def forward(self, frames, data):
        x = frames
        data = data * 2.0 - 1.0
        x = self._conv1(x)
        
        N, _, L, H, W = x.size()
        x = torch.cat([
            x,
            data.view(N, self.data_dim, 1, 1, 1).expand(N, self.data_dim, L, H, W)
        ], dim=1)
        x = self._conv2(x)

        N, _, L, H, W = x.size()
        x = torch.cat([
            x,
            data.view(N, self.data_dim, 1, 1, 1).expand(N, self.data_dim, L, H, W)
        ], dim=1)
        x = self._conv3(x)

        N, _, L, H, W = x.size()
        x = torch.cat([
            x,
            data.view(N, self.data_dim, 1, 1, 1).expand(N, self.data_dim, L, H, W)
        ], dim=1)
        x = self._conv4(x)

        return frames + 0.0157 * x
