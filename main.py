import gc
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm
from models import *
from datasets import *

data_dim = 32
num_samples = 6400
use_noise = False
use_critic = False
use_adversarial = False

train, val = load_train_val()
noise = lambda x: Scale()(x) if use_noise else x
enc = Encoder(data_dim=data_dim).cuda()
dec = Decoder(data_dim=data_dim).cuda()
critic, adversary = Critic().cuda(), Adversary().cuda()

opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-4)
adv_opt = torch.optim.Adam(list(critic.parameters()) + list(adversary.parameters()), lr=1e-4)

rows = []
for epoch in range(1, 128):
    metrics = {
        "train.loss": [], 
        "train.acc": [], 
        "train.mse": [], 
        "val.acc": [], 
        "val.mse": [],
    }

    losses = []
    iterator = tqdm(train)
    for frames in map(lambda x: x[0], iterator):
        gc.collect()
        frames = frames.cuda()
        data = torch.zeros(data_dim).random_(0, 2).cuda()
        wm_frames = enc(frames, data)
        wm_data = dec(noise(wm_frames))

        # Optimize critic
        if use_adversarial or use_critic:
            adv_opt.zero_grad()
            loss = 0.0
            if use_adversarial:
                loss += F.mse_loss(adversary(wm_frames), frames) # Train the adversary to remove watermarks
            if use_critic:
                loss += critic(frames) - critic(wm_frames)       # Train the critic to detect watermarks
            loss.backward(retain_graph=True)
            for p in critic.parameters():
                p.data.clamp_(-0.1, 0.1)
            adv_opt.step()

        # Optimize encoder and decoder
        loss = 0.0
        loss += F.mse_loss(wm_frames, frames)                     # Train the encoder/decoder to reproduce the input
        loss += F.binary_cross_entropy_with_logits(wm_data, data) # Train the decoder to recover watermarks
        if use_critic:
            loss += critic(wm_frames)                             # Train the encoder/decoder to fool the critic
        if use_adversarial:
            loss -= F.mse_loss(adversary(wm_frames), frames)      # Train the encoder/decoder to mess up the adversary
        losses.append(loss)
        if len(losses) == 16:
            (sum(losses) / len(losses)).backward()
            opt.step()
            opt.zero_grad()
            losses = []

        metrics["train.loss"].append(loss.item())
        metrics["train.acc"].append((wm_data >= 0.0).eq(data >= 0.5).sum().float().item() / data.numel())
        metrics["train.mse"].append(F.mse_loss(wm_frames, frames).item())
        if len(metrics["train.acc"]) > num_samples:
            break
        iterator.set_description("%s" % ({k: round(sum(v) / len(v), 3) for k, v in metrics.items() if len(v) > 0}))

    iterator = tqdm(val)
    for frames in map(lambda x: x[0], iterator):
        gc.collect()
        frames = frames.cuda()
        data = torch.zeros(data_dim).random_(0, 2).cuda()
        wm_frames = enc(frames, data)
        wm_data = dec(wm_frames)
        metrics["val.acc"].append((wm_data >= 0.0).eq(data >= 0.5).sum().float().item() / data.numel())
        metrics["val.mse"].append(F.mse_loss(wm_frames, frames).item())
        if len(metrics["val.acc"]) > num_samples:
            break
        iterator.set_description("%s" % ({k: round(sum(v) / len(v), 3) for k, v in metrics.items() if len(v) > 0}))

        if len(metrics["val.acc"]) == 2:
            _, _, H, W = wm_frames.size()
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('logs/example.avi', fourcc, 20.0, (H,W))
            for i in range(wm_frames.size(0)):
                frame = wm_frames[i].permute(1, 2, 0).clamp(-1.0, 1.0)
                frame = 255.0 * (frame.cpu().detach().numpy() + 1.0) / 2.0
                out.write(frame.astype(np.uint8))
                cv2.imwrite("logs/example.%s.png" % i, frame.astype(np.uint8))
            out.release()

    metrics = {k: round(sum(v) / len(v), 3) for k, v in metrics.items()}
    metrics["epoch"] = epoch
    rows.append(metrics)

    df = pd.DataFrame(rows)
    df.to_csv("logs/metrics.csv", index=False)
    torch.save((enc, dec, critic, adversary), "logs/model.pt")
