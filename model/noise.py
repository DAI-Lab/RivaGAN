import os
import cv2
import torch
import torch.nn as nn
import torch_dct as dct
import torch.optim as optim
import torch.nn.functional as F
from random import random, randint

class Crop(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, min_pct=0.9, max_pct=1.0):
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
    Output: (N, 3, L, H, W)
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

class Compression(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, yuv=False, min_pct=0.0, max_pct=0.5):
        super(Compression, self).__init__()
        self.yuv = yuv
        self.min_pct = min_pct
        self.max_pct = max_pct

    def forward(self, y):
        N, _, L, H, W = y.size()
        
        L = int(y.size(2) * (random() * (self.max_pct - self.min_pct) + self.min_pct))
        H = int(y.size(3) * (random() * (self.max_pct - self.min_pct) + self.min_pct))
        W = int(y.size(4) * (random() * (self.max_pct - self.min_pct) + self.min_pct))

        if self.yuv:
            y = torch.stack([
                0.299    * y[:,2,:,:,:] + 0.587    * y[:,1,:,:,:] + 0.114    * y[:,0,:,:,:],
                -0.168736 * y[:,2,:,:,:] - 0.331264 * y[:,1,:,:,:] + 0.500    * y[:,0,:,:,:],
                0.500    * y[:,2,:,:,:] - 0.418688 * y[:,1,:,:,:] - 0.081312 * y[:,0,:,:,:],
            ], dim=1)
        
        y = dct.dct_3d(y)
        if L > 0: y[:,:,-L:,:,:] = 0.0
        if H > 0: y[:,:,:,-H:,:] = 0.0
        if W > 0: y[:,:,:,:,-W:] = 0.0
        y = dct.idct_3d(y)

        if self.yuv:
            y = torch.stack([
                1.000 * y[:,0,:,:,:] + 1.772    * y[:,1,:,:,:] + 0.000    * y[:,2,:,:,:],
                1.000 * y[:,0,:,:,:] - 0.344136 * y[:,1,:,:,:] - 0.714136 * y[:,2,:,:,:],
                1.000 * y[:,0,:,:,:] + 0.000    * y[:,1,:,:,:] + 1.402    * y[:,2,:,:,:],
            ], dim=1)

        return y

job_id = randint(0, 100)

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

class AdaptiveCompression(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self):
        super(AdaptiveCompression, self).__init__()
        self.compress = nn.Sequential(
            nn.Conv3d(3, 3, kernel_size=(3,15,15), padding=(1,7,7)),
            nn.Tanh(),
        ).cuda()
        self.optimizer = optim.SGD(self.compress.parameters(), lr=1e-6)

    def adapt(self, frames, steps=2):
        self.optimizer.zero_grad()
        y = frames.detach()
        y = torch.clamp(y, min=-1.0, max=1.0)
        loss = F.mse_loss(self.forward(y, adapt=False, require_grad=True), mjpeg(y))
        loss.backward(retain_graph=True)
        self.optimizer.step()

        print()
        print("-"*100)
        print("Adaptive ", loss.item())
        print("DCT (YUV)", F.mse_loss(Compression(yuv=True)(frames), mjpeg(frames)).item())
        print("DCT (RGB)", F.mse_loss(Compression(yuv=False)(frames), mjpeg(frames)).item())
        print("-"*100)

    def forward(self, frames, adapt=True, require_grad=False):
        # Update learned compression
        #if adapt:
        #    self.adapt(frames)

        # Randomly return the actual compression result. This will stop the gradients from 
        # flowing to the encoder but will help the decoder by providing a more accurate signal.
        if not require_grad and random() < 0.5:
            return mjpeg(frames)

        # Use learned compression
        x = frames
        x = torch.clamp(x, min=-1.0, max=1.0)
        x = x + 0.0157 * (2.0*torch.rand(x.size()).to(x.device)-1.0) #self.compress(x)
        x = torch.clamp(x, min=-1.0, max=1.0)
        return x

if __name__ == "__main__":
    model = AdaptiveCompression().cuda()
    for epoch in range(100):
        N, L, H, W = 32, randint(1, 5), 2*randint(64, 128), 2*randint(64, 128)
        x = torch.rand((N, 3, L, H, W)).cuda() * 2.0 - 1.0
        y = model(x, adapt=True)
        print(epoch, torch.mean(torch.abs(x - y)).item())
