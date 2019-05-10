import sys; sys.path.append('../')
import cv2
import os
import torch
import imageio
import numpy as np
from glob import glob
from tqdm import tqdm
from data import load_train_val
from model import AttentiveEncoder, AttentiveDecoder

video_paths = glob("../data/hollywood2/val/*.avi")
video_paths = list(sorted(video_paths))[:100]

def make_mturk(dir_name, path_to_model):
    encoder, decoder, _, _ = torch.load(path_to_model)
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        data = torch.tensor([[0.0]*10 + [1.0] * (encoder.data_dim - 10)]).cuda()
        for path_to_video in video_paths:
            vin = cv2.VideoCapture(path_to_video)
            width = int(vin.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vin.get(cv2.CAP_PROP_FRAME_HEIGHT))
            length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))

            video_name = os.path.basename(path_to_video)
            source = cv2.VideoWriter("/home/kevz/mturk/source/%s" % video_name, cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (width, height))
            watermarked = cv2.VideoWriter("/home/kevz/mturk/%s/%s" % (dir_name, video_name), cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (width, height))

            for _ in tqdm(range(min(length, 100)), path_to_video):
                ok, frame = vin.read()

                frame = torch.FloatTensor([frame]) / 127.5 - 1.0      # (L, H, W, 3)
                frame = frame.permute(3, 0, 1, 2).unsqueeze(0).cuda() # (1, 3, L, H, W)
                wm_frame = encoder(frame, data)                       # (1, 3, L, H, W)
                wm_frame = torch.clamp(wm_frame, min=-1.0, max=1.0)

                frame = ((frame[0,:,0,:,:].permute(1,2,0) + 1.0) * 127.5).detach().cpu().numpy().astype("uint8")
                wm_frame = ((wm_frame[0,:,0,:,:].permute(1,2,0) + 1.0) * 127.5).detach().cpu().numpy().astype("uint8")

                source.write(frame)
                watermarked.write(wm_frame)

            source.release()
            watermarked.release()

make_mturk("c0.a0", "../results/1557517365/model.pt")
make_mturk("c1.a0", "../results/1557517373/model.pt")
make_mturk("c1.a1", "../results/1557517388/model.pt")
