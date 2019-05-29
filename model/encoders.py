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

def multiplicative(x, data):
    """
    This function takes a 5d tensor (with the same shape and dimension order
    as the input to Conv3d) and a 2d data tensor. For each element in the 
    batch, the data vector is combined with the first D dimensions of the 5d
    tensor through an elementwise product.
    
    Input: (N, C_{in}, L, H, W), (N, D)
    Output: (N, C_{in}, L, H, W)
    """
    N, D = data.size()
    N, C, L, H, W = x.size()
    assert D <= C, "data dims must be less than channel dims"
    x = torch.cat([
        x[:,:D,:,:,:] * data.view(N, D, 1, 1, 1).expand(N, D, L, H, W),
        x[:,D:,:,:,:]
    ], dim=1)
    return x

class Encoder(nn.Module):
    """
    Input: (N, 3, L, H, W), (N, D,)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, data_dim, linf_max=0.016, combiner="spatial_repeat", kernel_size=(1,3,3), padding=(0,1,1)):
        super(Encoder, self).__init__()
        
        self.linf_max = linf_max
        self.data_dim = data_dim
        self.combiner = {
            "spatial_repeat": spatial_repeat,
            "multiplicative": multiplicative
        }[combiner]

        self._conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=kernel_size, padding=padding, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=kernel_size, padding=padding, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=kernel_size, padding=padding, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=kernel_size, padding=padding, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64),
        )
        if self.combiner == spatial_repeat:
            self._conv2 = nn.Sequential(
                nn.Conv3d(64+data_dim, 64, kernel_size=kernel_size, padding=padding, stride=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(64),
            )
        else:
            self._conv2 = nn.Sequential(
                nn.Conv3d(64, 64, kernel_size=kernel_size, padding=padding, stride=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(64),
            )
        self._conv3 = nn.Sequential(
            nn.Conv3d(64, 3, kernel_size=(1,1,1), padding=(0,0,0), stride=1),
            nn.Tanh(),
        )

    def forward(self, frames, data):
        data = data * 2.0 - 1.0
        x = self._conv1(frames)
        x = self._conv2(self.combiner(x, data))
        x = self._conv3(x)
        return frames + self.linf_max * x

class StridedEncoder(nn.Module):
    """
    Input: (N, 3, L, H, W), (N, D,)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, data_dim, linf_max=0.016, combiner="spatial_repeat", kernel_size=(1,11,11), padding=(0,5,5)):
        super(StridedEncoder, self).__init__()
        
        self.linf_max = linf_max
        self.data_dim = data_dim
        self.combiner = {
            "spatial_repeat": spatial_repeat,
            "multiplicative": multiplicative
        }[combiner]

        self._conv1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=kernel_size, padding=padding, stride=1),
            nn.Tanh(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=kernel_size, padding=padding, stride=(1,2,2)),
            nn.Tanh(),
            nn.BatchNorm3d(64),
        )
        if self.combiner == spatial_repeat:
            self._conv2 = nn.Sequential(
                nn.ConvTranspose3d(64+data_dim, 128, kernel_size=kernel_size, padding=padding, stride=(1,2,2), output_padding=(0,1,1)),
                nn.Tanh(),
                nn.BatchNorm3d(128),
            )
        else:
            self._conv2 = nn.Sequential(
                nn.ConvTranspose3d(64, 128, kernel_size=kernel_size, padding=padding, stride=(1,2,2), output_padding=(0,1,1)),
                nn.Tanh(),
                nn.BatchNorm3d(128),
            )
        self._conv3 = nn.Sequential(
            nn.Conv3d(128, 3, kernel_size=kernel_size, padding=padding, stride=1),
            nn.Tanh(),
        )

    def forward(self, frames, data):
        data = data * 2.0 - 1.0
        x = self._conv1(frames)
        x = self._conv2(self.combiner(x, data))
        x = self._conv3(x)
        return frames + self.linf_max * x
