import torch
import torch.nn as nn
import torch.nn.functional as F

def spatial_repeat(x, data, concat=True):
    N, D = data.size()
    N, _, L, H, W = x.size()
    data = data.view(N, D, 1, 1, 1).expand(N, D, L, H, W)
    if concat:
        x = torch.cat([
            x,
            data
        ], dim=1)
    return data

position_embedding_dims = 6
def position_embedding(frames):
    N, _, L, H, W = frames.size()
    offsetH, offsetW = torch.randn(1).item() * H, torch.randn(1).item() * H
    embedding = torch.cat([
        torch.sin(2 * 2 * 3.14 * torch.arange(0, H).view(1, 1, 1, H, 1).expand(N, 1, L, H, W).float() / H + offsetH),
        torch.sin(2 * 2 * 3.14 * torch.arange(0, W).view(1, 1, 1, 1, W).expand(N, 1, L, H, W).float() / W + offsetW),
        torch.sin(4 * 2 * 3.14 * torch.arange(0, H).view(1, 1, 1, H, 1).expand(N, 1, L, H, W).float() / H + offsetH),
        torch.sin(4 * 2 * 3.14 * torch.arange(0, W).view(1, 1, 1, 1, W).expand(N, 1, L, H, W).float() / W + offsetW),
        torch.sin(8 * 2 * 3.14 * torch.arange(0, H).view(1, 1, 1, H, 1).expand(N, 1, L, H, W).float() / H + offsetH),
        torch.sin(8 * 2 * 3.14 * torch.arange(0, W).view(1, 1, 1, 1, W).expand(N, 1, L, H, W).float() / W + offsetW),
    ], dim=1).to(frames.device)
    return torch.cat([frames, embedding], dim=1)

class Encoder(nn.Module):

    def __init__(self, data_dim=32, lmax_norm=0.016, kernel_size=(1, 11, 11), use_position_embedding=False):
        super(Encoder, self).__init__()
        self._data_dim = data_dim
        self._lmax_norm = lmax_norm
        self._kernel_size = kernel_size
        self._padding = tuple(x//2 for x in self._kernel_size)
        self._use_position_embedding = use_position_embedding

        input_dims = 3 + self._data_dim
        if self._use_position_embedding:
            input_dims += position_embedding_dims
        self._conv1 = nn.Sequential(
            nn.Conv3d(input_dims, 16, kernel_size=self._kernel_size, padding=self._padding),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(16),
        )
        self._conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=self._kernel_size, padding=self._padding),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(32),
        )
        self._conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=self._kernel_size, padding=self._padding),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(64),
        )
        self._conv4 = nn.Sequential(
            nn.Conv3d(64, 3, kernel_size=self._kernel_size, padding=self._padding),
            nn.Tanh(),
        )

    def forward(self, frames, data):
        data = data * 2.0 - 1.0
        x = spatial_repeat(frames, data)
        if self._use_position_embedding:
            x = position_embedding(x)
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x)
        return frames + self._lmax_norm * x

class MultiplicativeEncoder(nn.Module):

    def __init__(self, data_dim=32, lmax_norm=0.016, kernel_size=(1, 11, 11), use_position_embedding=False):
        super(MultiplicativeEncoder, self).__init__()
        self._data_dim = data_dim
        self._lmax_norm = lmax_norm
        self._kernel_size = kernel_size
        self._padding = tuple(x//2 for x in self._kernel_size)
        self._use_position_embedding = use_position_embedding

        input_dims = 3
        if self._use_position_embedding:
            input_dims += position_embedding_dims
        self._conv1 = nn.Sequential(
            nn.Conv3d(input_dims, 32, kernel_size=self._kernel_size, padding=self._padding),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(32),
        )
        self._conv2 = nn.Sequential(
            nn.Conv3d(input_dims+32, data_dim, kernel_size=self._kernel_size, padding=self._padding),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(data_dim),
        )
        self._conv3 = nn.Sequential(
            nn.Conv3d(input_dims+32+data_dim, 3, kernel_size=self._kernel_size, padding=self._padding),
            nn.Tanh(),
        )

    def forward(self, frames, data):
        x = frames
        data = data * 2.0 - 1.0
        if self._use_position_embedding:
            x = position_embedding(x)
        x1 = x
        x2 = self._conv1(x1)
        x3 = self._conv2(torch.cat([x1,x2], dim=1)) * spatial_repeat(frames, data, concat=False)
        x4 = self._conv3(torch.cat([x1,x2,x3], dim=1))
        return frames + self._lmax_norm * x4
