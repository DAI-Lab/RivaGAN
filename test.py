import gc
import cv2
import json
import torch
import random
import imageio
import numpy as np
from model import *
from glob import glob
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

SEQ_LEN = 3
SEQ_LEN_MIDDLE = SEQ_LEN//2

def create_video(path_to_model):
    # Create an example video
    gc.collect()
    data = torch.tensor([[0.0]*10 + [1.0] * 22]).cuda()
    encoder, decoder, _, _ = torch.load(path_to_model)
    encoder, decoder = map(lambda x: x.cuda(), (encoder, decoder))
    
    for index, path in enumerate([
            "data/hollywood2/val/actioncliptest00001.avi",
            "data/hollywood2/val/actioncliptest00003.avi",
            "data/hollywood2/val/actioncliptest00005.avi",
            "data/hollywood2/val/actioncliptest00007.avi",
            "data/hollywood2/val/actioncliptest00009.avi",
            "data/hollywood2/val/actioncliptest00011.avi",
            "data/hollywood2/val/actioncliptest00013.avi",
            "data/hollywood2/val/actioncliptest00015.avi",
        ]):
        frames = []
        vin = cv2.VideoCapture(path)
        width = int(vin.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vin.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nb_frames = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
        vout1 = cv2.VideoWriter(path_to_model.replace("model.pt", "source.%s.avi" % index), cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))
        vout2 = cv2.VideoWriter(path_to_model.replace("model.pt", "watermarked.%s.avi" % index), cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))
        for _ in tqdm(range(min(nb_frames, 100)), "create_video"):
            ok, frame = vin.read()
            frames.append(frame)
            if len(frames) < SEQ_LEN:
                continue
            frames = frames[-SEQ_LEN:]

            x = torch.FloatTensor(frames) / 127.5 - 1.0   # (L, H, W, 3)
            x = x.permute(3, 0, 1, 2).unsqueeze(0).cuda() # (1, 3, L, H, W)
            y = encoder(x, data)                          # (1, 3, L, H, W)

            vout1.write(((x[0,:,SEQ_LEN_MIDDLE,:,:].permute(1,2,0) + 1.0) * 127.5).detach().cpu().numpy().astype("uint8"))
            image = torch.clamp(y[0,:,SEQ_LEN_MIDDLE,:,:].permute(1,2,0), min=-1.0, max=1.0)
            vout2.write(((image + 1.0) * 127.5).detach().cpu().numpy().astype("uint8"))

def generate_metrics(path_to_model, path_to_data="data/hollywood2/val/*.avi", key="val"):
    # Generate metrics
    gc.collect()
    metrics = {
        "ssim": [],
        "psnr": [],
        "acc.identity": [],
        "acc.aspect_ratio": [],
        "acc.frame_rate": [],
        "acc.transcoded": [],
    }
    data = torch.tensor([[0.0]*10 + [1.0] * 22]).cuda()
    encoder, decoder, _, _ = torch.load(path_to_model)
    encoder, decoder = map(lambda x: x.cuda(), (encoder, decoder))

    videos = list(sorted(glob(path_to_data)))
    random.Random(4).shuffle(videos)
    for video_path in tqdm(videos[:32], "generate_metrics"):
        gc.collect()
        vin = cv2.VideoCapture(video_path)
        width = int(vin.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vin.get(cv2.CAP_PROP_FRAME_HEIGHT))
        height, width = 4*(height//8), 4*(width//8)
        vout = cv2.VideoWriter("/tmp/output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (width, height))

        nb_frames = min(int(vin.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
        start_idx = random.randint(1, int(vin.get(cv2.CAP_PROP_FRAME_COUNT)) - nb_frames - 1)
        vin.set(cv2.CAP_PROP_POS_FRAMES, start_idx-1)

        frames = []
        for _ in range(nb_frames):
            gc.collect()
            ok, frame = vin.read()
            frames.append(frame[:height,:width])
            if len(frames) < SEQ_LEN:
                continue
            frames = frames[-SEQ_LEN:]

            x = torch.FloatTensor(frames) / 127.5 - 1.0   # (L, H, W, 3)
            x = x.permute(3, 0, 1, 2).unsqueeze(0).cuda() # (1, 3, L, H, W)
            y = encoder(x, data)                          # (1, 3, L, H, W)

            image = torch.clamp(y[0,:,SEQ_LEN_MIDDLE,:,:].permute(1,2,0), min=-1.0, max=1.0)
            image = ((image + 1.0) * 127.5).int().float() / 127.5 - 1.0 # quantize
            vout.write(((image + 1.0) * 127.5).detach().cpu().numpy().astype("uint8"))

            metrics["ssim"].append(ssim(x[:,:,SEQ_LEN_MIDDLE,:,:], y[:,:,SEQ_LEN_MIDDLE,:,:]).item())
            metrics["psnr"].append(psnr(x[:,:,SEQ_LEN_MIDDLE,:,:], y[:,:,SEQ_LEN_MIDDLE,:,:]).item())

            y_out = decoder(torch.clamp(y.detach(), min=-1.0, max=1.0))
            acc = (y_out >= 0.0).eq(data >= 0.5).sum().float().item() / data.numel()
            metrics["acc.identity"].append(acc)

            y_out = decoder(torch.clamp(y.detach()[:,:,:,:100,:100], min=-1.0, max=1.0))
            acc = (y_out >= 0.0).eq(data >= 0.5).sum().float().item() / data.numel()
            metrics["acc.aspect_ratio"].append(acc)

            y_out = decoder(torch.clamp(torch.stack([
                y.detach()[:,:,SEQ_LEN_MIDDLE,:,:],
            ], dim=2), min=-1.0, max=1.0))
            acc = (y_out >= 0.0).eq(data >= 0.5).sum().float().item() / data.numel()
            metrics["acc.frame_rate"].append(acc)
        vout.release()

        frames = []
        vin = cv2.VideoCapture("/tmp/output.avi")
        for _ in range(int(vin.get(cv2.CAP_PROP_FRAME_COUNT))):
            ok, frame = vin.read()
            frames.append(frame)
            if len(frames) < SEQ_LEN:
                continue
            frames = frames[-SEQ_LEN:]

            x = torch.FloatTensor(frames) / 127.5 - 1.0   # (L, H, W, 3)
            x = x.permute(3, 0, 1, 2).unsqueeze(0).cuda() # (1, 3, L, H, W)
            y_out = decoder(x)
            acc = (y_out >= 0.0).eq(data >= 0.5).sum().float().item() / data.numel()
            metrics["acc.transcoded"].append(acc)

    imageio.imwrite(path_to_model.replace("model.pt", "example.png"), frame)
    with open(path_to_model.replace("model.pt", "%s.json" % key), "wt") as fout:
        json.dump({k: np.mean(v) for k, v in metrics.items()}, fout, indent=2)

with torch.no_grad():
    for path_to_model in sorted(glob("results/*/model.pt")):
        print(path_to_model)
        create_video(path_to_model)
        generate_metrics(path_to_model, path_to_data="data/hollywood2/train/*.avi", key="train")
        generate_metrics(path_to_model, path_to_data="data/hollywood2/val/*.avi", key="val")
