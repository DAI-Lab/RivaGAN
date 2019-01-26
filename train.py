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

    encoder = Encoder(data_dim=args.data_dim).cuda()
    decoder = Decoder(data_dim=args.data_dim).cuda()
    critic = Critic().cuda()
    adversary = Adversary().cuda()
    noise = lambda x: Scale()(Crop()(x)) if args.use_noise else x
    
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

        iterator = tqdm(train, ncols=0)
        for frames in iterator:
            frames = frames.cuda()
            data = torch.zeros((frames.size(0), args.data_dim)).random_(0, 2).cuda()
            wm_frames = encoder(frames, data)
            wm_data = decoder(noise(wm_frames))

            if args.use_critic or args.use_adversary:
                d_opt.zero_grad()
                loss = 0.0
                if args.use_critic:
                    loss += torch.mean(critic(frames) - critic(wm_frames))
                if args.use_adversary:
                    adv_frames = adversary(wm_frames)
                    loss += F.mse_loss(adv_frames, frames)
                    loss -= F.binary_cross_entropy_with_logits(decoder(adv_frames), data)
                loss.backward(retain_graph=True)
                for p in critic.parameters():
                    p.data.clamp_(-0.1, 0.1)
                d_opt.step()

            loss = 0.0
            loss += F.mse_loss(wm_frames, frames)
            loss += F.binary_cross_entropy_with_logits(wm_data, data)
            if args.use_critic:
                loss += torch.mean(critic(wm_frames))
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

        iterator = tqdm(val, ncols=0)
        for frames in iterator:
            frames = frames.cuda()
            data = torch.zeros((frames.size(0), args.data_dim)).random_(0, 2).cuda()
            wm_frames = encoder(frames, data)
            wm_data = decoder(noise(wm_frames))
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
        with open(os.path.join(log_dir, "config.json"), "wt") as fout:
            fout.write(json.dumps(args.__dict__, indent=2, default=lambda o: str(o)))
        g_scheduler.step(metrics["train.loss"])
        d_scheduler.step(metrics["train.loss"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_dim', type=int, default=32)
    parser.add_argument('--use_noise', type=int, default=1)
    parser.add_argument('--use_critic', type=int, default=1)
    parser.add_argument('--use_adversary', type=int, default=1)
    run(parser.parse_args())
