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
            nn.Conv3d(3, 16, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(16),
            nn.Conv3d(16, 32, kernel_size=3, padding=(1, 1, 1), stride=(1, 2, 2)),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(32),
        )
        self._conv2 = nn.Sequential(
            nn.ConvTranspose3d(32+data_dim, 16, kernel_size=3, padding=(1, 1, 1), stride=(1, 2, 2), output_padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(16),
            nn.Conv3d(16, 3, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
        )

    def forward(self, frames, data):
        x = frames

        # Pad and process
        x = torch.cat([
            x[:,:,:1],
            x,
            x[:,:,-1:],
        ], dim=2)
        x = self._conv1(x)

        # Concat and process
        N, _, L, H, W = x.size()
        x = torch.cat([
            x,
            data.view(N, self.data_dim, 1, 1, 1).expand(N, self.data_dim, L, H, W)
        ], dim=1)
        x = self._conv2(x)

        return frames + x[:,:,1:-1,:,:]
