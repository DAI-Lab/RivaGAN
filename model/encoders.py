import torch
import torch.nn as nn
import torch.nn.functional as F

def spatial_repeat(x, data):
    N, D = data.size()
    N, _, L, H, W = x.size()
    x = torch.cat([
        x,
        data.view(N, D, 1, 1, 1).expand(N, D, L, H, W)
    ], dim=1)
    return x

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

    def __init__(self, data_dim=32, lmax_norm=0.024, kernel_size=(1, 11, 11), use_position_embedding=False):
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
            nn.Conv3d(input_dims, 32, kernel_size=self._kernel_size, padding=self._padding),
            nn.Tanh(),
            nn.InstanceNorm3d(32),
        )
        self._conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=self._kernel_size, padding=self._padding),
            nn.Tanh(),
            nn.InstanceNorm3d(64),
        )
        self._conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=self._kernel_size, padding=self._padding),
            nn.Tanh(),
            nn.InstanceNorm3d(128),
        )
        self._conv4 = nn.Sequential(
            nn.Conv3d(128, 3, kernel_size=self._kernel_size, padding=self._padding),
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

if __name__ == "__main__":
    import torch
    enc = Encoder(data_dim=32, kernel_size=(1,11,11), use_position_embedding=False).cuda()
    N, _, L, H, W = 16, 3, 1, 100, 100
    frames = torch.randn(N, 3, L, H, W).cuda()
    data = torch.randn(N, 32).cuda()
    print(enc(frames, data).size())
