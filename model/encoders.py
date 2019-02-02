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
            nn.Conv3d(3, 16, kernel_size=(1,5,5), padding=(0,2,2), stride=1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(16),
            nn.Conv3d(16, 32, kernel_size=(1,5,5), padding=(0,2,2), stride=(1, 2, 2)),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(32),
        )
        self._conv2 = nn.Sequential(
            nn.ConvTranspose3d(32+data_dim, 48, kernel_size=(1,5,5), padding=(0,2,2), stride=(1, 2, 2), output_padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(48),
            nn.Conv3d(48, 3, kernel_size=(3,5,5), padding=(1,2,2), stride=1),
            nn.Tanh(),
        )

    def forward(self, frames, data):
        x = frames
        data = data * 2.0 - 1.0

        # Pad and process
        x = self._conv1(x)

        # Concat and process
        N, _, L, H, W = x.size()
        x = torch.cat([
            x,
            data.view(N, self.data_dim, 1, 1, 1).expand(N, self.data_dim, L, H, W)
        ], dim=1)
        x = self._conv2(x)

        return frames + x

class ShortcutEncoder(nn.Module):
    """
    Input: (N, 3, L, H, W), (N, D,)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, data_dim):
        super(ShortcutEncoder, self).__init__()
        self.data_dim = data_dim
        self._conv1 = nn.Sequential(
            nn.Conv3d(3+12, 16, kernel_size=3, padding=1, stride=1),
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
        data = data * 2.0 - 1.0

        # Pad and process
        N, _, L, H, W = x.size()
        x = torch.cat([
            x,
            # Absolute
            torch.sin(torch.arange(float(L)).view(1, 1, L, 1, 1).expand(N, 1, L, H, W).cuda() * 2.0 * 3.14 / 8.0),
            torch.sin(torch.arange(float(H)).view(1, 1, 1, H, 1).expand(N, 1, L, H, W).cuda() * 2.0 * 3.14 / 8.0),
            torch.sin(torch.arange(float(W)).view(1, 1, 1, 1, W).expand(N, 1, L, H, W).cuda() * 2.0 * 3.14 / 8.0),
            torch.sin(torch.arange(float(L)).view(1, 1, L, 1, 1).expand(N, 1, L, H, W).cuda() * 2.0 * 3.14 / 16.0),
            torch.sin(torch.arange(float(H)).view(1, 1, 1, H, 1).expand(N, 1, L, H, W).cuda() * 2.0 * 3.14 / 16.0),
            torch.sin(torch.arange(float(W)).view(1, 1, 1, 1, W).expand(N, 1, L, H, W).cuda() * 2.0 * 3.14 / 16.0),
            # Relative
            torch.sin(8.0 * 3.14 * torch.arange(float(L)).view(1, 1, L, 1, 1).expand(N, 1, L, H, W).cuda() / L),
            torch.sin(8.0 * 3.14 * torch.arange(float(H)).view(1, 1, 1, H, 1).expand(N, 1, L, H, W).cuda() / H),
            torch.sin(8.0 * 3.14 * torch.arange(float(W)).view(1, 1, 1, 1, W).expand(N, 1, L, H, W).cuda() / W),
            torch.sin(16.0 * 3.14 * torch.arange(float(L)).view(1, 1, L, 1, 1).expand(N, 1, L, H, W).cuda() / L),
            torch.sin(16.0 * 3.14 * torch.arange(float(H)).view(1, 1, 1, H, 1).expand(N, 1, L, H, W).cuda() / H),
            torch.sin(16.0 * 3.14 * torch.arange(float(W)).view(1, 1, 1, 1, W).expand(N, 1, L, H, W).cuda() / W),
        ], dim=1)
        x = self._conv1(x)

        # Concat and process
        N, _, L, H, W = x.size()
        x = torch.cat([
            x,
            data.view(N, self.data_dim, 1, 1, 1).expand(N, self.data_dim, L, H, W)
        ], dim=1)
        x = self._conv2(x)

        return frames + x

if __name__ == "__main__":
    import numpy as np
    from time import time

    N = 16
    frames = torch.randn(N, 3, 10, 50, 50).cuda()
    data = torch.randn(N, 32).cuda()

    start = time()
    model = Encoder(32).cuda()
    print(sum(np.prod(p.size()) for p in model.parameters()))
    print(model(frames, data).size())
    print(time() - start)

    start = time()
    model = ShortcutEncoder(32).cuda()
    print(sum(np.prod(p.size()) for p in model.parameters()))
    print(model(frames, data).size())
    print(time() - start)
