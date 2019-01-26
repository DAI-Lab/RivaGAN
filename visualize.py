import os
import cv2
import torch
import numpy as np
from model import *
from glob import glob

def handle(path_to_model):
    path_to_output = os.path.join(os.path.dirname(path_to_model), "output.avi")

    data = torch.FloatTensor([[0]*10 + [1] * 22]).cuda()
    encoder, decoder, _, _ = torch.load(path_to_model)
    encoder, decoder = map(lambda x: x.cuda(), (encoder, decoder))

    vin = cv2.VideoCapture("data/hollywood2/val/actioncliptest00015.avi")
    width = int(vin.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vin.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vout = cv2.VideoWriter(path_to_output, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))

    i = 0
    frames = []
    slices = []
    ok, frame = vin.read()
    while ok:
        i += 1
        print(path_to_model, i)
        frames.append(frame)
        if len(frames) < 3:
            continue
        frames = frames[-3:]

        x = torch.FloatTensor(frames) / 127.5 - 1.0 # (L, H, W, 3)
        x = x.permute(3, 0, 1, 2).unsqueeze(0) # (1, 3, L, H, W)]
        x = encoder(x.cuda(), data.cuda())[0,:,1,:,:].permute(1,2,0) # (H, W, 3)
        x = torch.clamp(x, min=-1.0, max=1.0)
        x = ((x + 1.0) * 127.5).detach().cpu().numpy().astype("uint8")
        vout.write(x)
        prev_frame = frame
        ok, frame = vin.read()

        if len(slices) < x.shape[0]:
            slices.append(x[x.shape[0]//2,:])
            cv2.imwrite(path_to_output.replace("output.avi", "st-slice.png"), np.stack(slices, axis=0))
        if len(slices) == 1:
            x = x.copy()
            x[x.shape[0]//2,:,0] = 0
            x[x.shape[0]//2,:,1] = 255
            x[x.shape[0]//2,:,2] = 0
            cv2.imwrite(path_to_output.replace("output.avi", "st-source.png"), x)

    # read it back
    y_out = None
    frames = []
    vin = cv2.VideoCapture(path_to_output)
    ok, frame = vin.read()
    while ok:
        i += 1
        print(i)
        frames.append(frame)
        if len(frames) < 3:
            continue
        frames = frames[-3:]

        x = torch.FloatTensor(frames) / 127.5 - 1.0 # (L, H, W, 3)
        x = x.permute(3, 0, 1, 2).unsqueeze(0) # (1, 3, L, H, W)]
        if type(y_out) == type(None):
            y_out = decoder(x.cuda()).cpu().detach()
        else:
            y_out += decoder(x.cuda()).cpu().detach()
        print(y_out > 0.0)
        ok, frame = vin.read()

for path_to_model in glob("results/**/model.pt"):
    handle(path_to_model)
