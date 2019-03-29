import os
import cv2
import torch
import torch.nn as nn
import torch_dct as dct
import torch.optim as optim
import torch.nn.functional as F
from random import random, randint

job_id = randint(0, 1000)

def mjpeg(x):
    """
    Write each video to disk and re-read it from disk.

    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """
    y = torch.zeros(x.size())
    _, _, _, height, width = x.size()
    for n in range(x.size(0)):
        vout = cv2.VideoWriter("/tmp/%s.avi" % job_id, cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (width, height))
        for l in range(x.size(2)):
            image = x[n,:,l,:,:] # (3, H, W)
            image = torch.clamp(image.permute(1,2,0), min=-1.0, max=1.0)
            vout.write(((image + 1.0) * 127.5).detach().cpu().numpy().astype("uint8"))
        vout.release()

        vin = cv2.VideoCapture("/tmp/%s.avi" % job_id)
        for l in range(x.size(2)):
            _, frame = vin.read() # (H, W, 3)
            frame = torch.tensor(frame / 127.5 - 1.0)
            y[n,:,l,:,:] = frame.permute(2,0,1)
    return y.to(x.device)

class Crop(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H', W')
    """

    def __init__(self, min_pct=0.8, max_pct=1.0):
        super(Crop, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct

    def forward(self, frames):
        pct = random() * (self.max_pct - self.min_pct) + self.min_pct

        _, _, _, old_H, old_W = frames.size()
        H, W = int(pct * old_H), int(pct * old_W)

        y = randint(0, old_H - H - 1)
        x = randint(0, old_W - W - 1)
        return frames[:,:,:,y:y+H,x:x+W]

class Scale(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H', W')
    """

    def __init__(self, min_pct=0.9, max_pct=1.0):
        super(Scale, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct

    def forward(self, frames):
        pct = random() * (self.max_pct - self.min_pct) + self.min_pct
        _, _, old_D, old_H, old_W = frames.size()
        H, W = int(pct * old_H), int(pct * old_W)
        return nn.AdaptiveAvgPool3d((old_D, H, W))(frames)

class Quantization(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, frames):
        one_bit_change = 2.0 * (1.0 / 255.0)
        noise = 2.0 * one_bit_change * torch.randn_like(frames)
        return frames + noise
