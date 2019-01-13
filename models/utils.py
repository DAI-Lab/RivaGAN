import torch
import torch.nn as nn
import torch.nn.functional as F

def _conv_3d(conv_3d, x):
    """
    Apply a 3d convolution layer to a batch of 2d images by treating the batch 
    dimension as the time dimension.

    Input: (N, C_{in}, W, H)
    Output: (N, C_{out}, W, H)
    """
    # (N, C_{in}, W, H) -> (1, C_{in}, N, W, H)
    x = x.permute(1, 0, 2, 3).unsqueeze(0)
    # (1, C_{in}, N, W, H) -> (1, C_{out}, N, W, H)
    x = conv_3d(x)
    # (1, C_{out}, N, W, H) -> (N, C_{out}, W, H)
    x = x[0].permute(1, 0, 2, 3)
    return x
