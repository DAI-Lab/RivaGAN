import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoders import *

class Decoder(nn.Module):

    def __init__(self, data_dim=32, kernel_size=(1, 11, 11), use_position_embedding=False):
        super(Decoder, self).__init__()
        self._data_dim = data_dim
        self._kernel_size = kernel_size
        self._use_position_embedding = use_position_embedding

        input_dims = 3
        if self._use_position_embedding:
            input_dims += position_embedding_dims
        self._conv1 = nn.Sequential(
            nn.Conv3d(input_dims, 16, kernel_size=self._kernel_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(16),
        )
        self._conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=self._kernel_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(32),
        )
        self._conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=self._kernel_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(64),
        )
        self._conv4 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=self._kernel_size),
            nn.Tanh(),
        )
        self._pool = nn.MaxPool3d(kernel_size=self._kernel_size, stride=(1, 1, 1))
        self._1x1 = nn.Conv3d(256, self._data_dim, kernel_size=(1,1,1))

    def forward(self, frames):
        x = frames
        if self._use_position_embedding:
            x = position_embedding(x)
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x)
        x = torch.cat([self._pool(x), -self._pool(-x)], dim=1)
        return torch.mean(self._1x1(x).view(x.size(0), self._data_dim, -1), dim=2)

class MultiplicativeDecoder(nn.Module):

    def __init__(self, data_dim=32, kernel_size=(1, 11, 11), use_position_embedding=False):
        super(MultiplicativeDecoder, self).__init__()
        self._data_dim = data_dim
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
            nn.Conv3d(input_dims+32+data_dim, 64, kernel_size=self._kernel_size, padding=self._padding),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(64),
        )
        self._pool = nn.MaxPool3d(kernel_size=self._kernel_size, stride=(1, 1, 1))
        self._1x1 = nn.Conv3d(128, self._data_dim, kernel_size=(1,1,1))

    def forward(self, frames):
        x = frames
        if self._use_position_embedding:
            x = position_embedding(x)
        x1 = x
        x2 = self._conv1(x1)
        x3 = self._conv2(torch.cat([x1,x2], dim=1))
        x4 = self._conv3(torch.cat([x1,x2,x3], dim=1))
        sx, ex = 2*self._padding[1], -1-2*self._padding[1]
        x = x4[:,:,:,sx:ex,sx:ex]
        x = torch.cat([self._pool(x), -self._pool(-x)], dim=1)
        return torch.mean(self._1x1(x).view(x.size(0), self._data_dim, -1), dim=2)
