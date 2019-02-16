import gc
import os
import json
import torch
import numpy as np
import pandas as pd
import argparse
import random
from time import time
from tqdm import tqdm
from itertools import chain
from torch.optim.lr_scheduler import ReduceLROnPlateau

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from data import *
from model import *

def run(args):
    log_dir = os.path.join("results/", str(int(time())))
    os.makedirs(log_dir, exist_ok=False)
    with open(os.path.join(log_dir, "config.json"), "wt") as fout:
        fout.write(json.dumps(args.__dict__, indent=2, default=lambda o: str(o)))

    encoder = Encoder(data_dim=args.data_dim).cuda()
    decoder = Decoder(data_dim=args.data_dim).cuda()
    critic = Critic().cuda()
    adversary = Adversary().cuda()

    noise_layers = []
    if args.use_crop:
        noise_layers.append(Crop())
    if args.use_scale:
        noise_layers.append(Scale())
    if args.use_compression:
        if args.use_yuv:
            noise_layers.append(YUVCompression())
        else:
            noise_layers.append(Compression())
    def noise(x):
        for layer in noise_layers:
            if random() < 0.5:
                x = layer(x)
        return x
    
    train, val = load_train_val(args.seq_len, args.batch_size)
    g_opt = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=args.lr)
    d_opt = torch.optim.Adam(chain(critic.parameters(), adversary.parameters()), lr=args.lr)
    g_scheduler = ReduceLROnPlateau(g_opt)
    d_scheduler = ReduceLROnPlateau(d_opt)

    # Training
    history = []
    for epoch in range(1, args.epochs + 1):
        metrics = {
            "train.loss": [], 
            "train.acc": [], 
            "train.mse": [], 
            "val.acc": [], 
            "val.mse": [],
        }

        gc.collect()
        iterator = tqdm(train, ncols=0)
        for frames in iterator:
            frames = torch.cat([frames, frames, frames, frames, frames, frames], dim=0).cuda()
            data = torch.zeros((frames.size(0), args.data_dim)).random_(0, 2).cuda()
            frames = torch.cat([frames, frames], dim=0).cuda()
            data = torch.cat([data, 1.0 - data], dim=0).cuda()

            if args.use_critic or args.use_adversary:
                loss = 0.0
                wm_frames = encoder(frames, data)
                if args.use_critic:
                    loss += torch.mean(critic(frames) - critic(wm_frames))
                if args.use_adversary:
                    loss -= F.binary_cross_entropy_with_logits(decoder(adversary(wm_frames)), data)
                d_opt.zero_grad()
                loss.backward()
                d_opt.step()
                for p in critic.parameters():
                    p.data.clamp_(-0.01, 0.01)

            loss = 0.0
            wm_frames = encoder(frames, data)
            wm_data = decoder(noise(wm_frames))
            loss += F.binary_cross_entropy_with_logits(wm_data, data)
            if args.use_critic:
                loss += torch.mean(critic(wm_frames))
            if args.use_adversary:
                loss += F.binary_cross_entropy_with_logits(decoder(adversary(wm_frames)), data)
            g_opt.zero_grad()
            loss.backward()
            g_opt.step()

            metrics["train.loss"].append(loss.item())
            metrics["train.acc"].append((wm_data >= 0.0).eq(data >= 0.5).sum().float().item() / data.numel())
            metrics["train.mse"].append(F.mse_loss(wm_frames, frames).item())
            iterator.set_description("Epoch %s | Loss %s | MSE %s | Acc %s" % (
                epoch, 
                np.mean(metrics["train.loss"]), 
                np.mean(metrics["train.mse"]), 
                np.mean(metrics["train.acc"])
            ))

        gc.collect()
        iterator = tqdm(val, ncols=0)
        for frames in iterator:
            frames = frames.cuda()
            data = torch.zeros((frames.size(0), args.data_dim)).random_(0, 2).cuda()
            wm_frames = encoder(frames, data)
            wm_data = decoder(wm_frames)
            metrics["val.acc"].append((wm_data >= 0.0).eq(data >= 0.5).sum().float().item() / data.numel())
            metrics["val.mse"].append(F.mse_loss(wm_frames, frames).item())
            iterator.set_description("Epoch %s | Val MSE %s | Val Acc %s" % (
                epoch,
                np.mean(metrics["val.mse"]),
                np.mean(metrics["val.acc"])
            ))

        metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
        metrics["epoch"] = epoch
        history.append(metrics)
        pd.DataFrame(history).to_csv(os.path.join(log_dir, "metrics.tsv"), index=False, sep="\t")
        torch.save((encoder, decoder, critic, adversary), os.path.join(log_dir, "model.pt"))
        g_scheduler.step(metrics["train.loss"])
        d_scheduler.step(metrics["train.loss"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_dim', type=int, default=32)
    parser.add_argument('--use_crop', type=int, default=0)
    parser.add_argument('--use_scale', type=int, default=0)
    parser.add_argument('--use_compression', type=int, default=0)
    parser.add_argument('--use_yuv', type=int, default=0)
    parser.add_argument('--use_critic', type=int, default=0)
    parser.add_argument('--use_adversary', type=int, default=0)
    run(parser.parse_args())
