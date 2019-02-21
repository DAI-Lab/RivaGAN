import gc
import os
import json
import numpy as np
import pandas as pd
import argparse
import random
from time import time
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from data import *
from model import *

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def run(args):
    # Load our datasets
    train, val = load_train_val(args.seq_len, args.batch_size)

    # Initialize our noise layers
    _noise = []
    if args.use_crop: _noise.append(Crop())
    if args.use_scale: _noise.append(Scale())
    if args.use_compression: _noise.append(Compression())
    def noise(x):
        for layer in _noise:
            if random.random() < 0.5:
                x = layer(x)
        return x
    
    # Initialize our modules and optimizers
    encoder = Encoder(data_dim=args.data_dim).cuda()
    decoder = Decoder(data_dim=args.data_dim).cuda()
    critic, adversary = Critic().cuda(), Adversary().cuda()

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    optimizer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    xoptimizer = optim.Adam(list(critic.parameters()) + list(adversary.parameters()), lr=args.lr)
    xoptimizer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(xoptimizer)

    # Set up the log directory
    log_dir = os.path.join("results/", "n%s%s%s-m%s%s-%s" % (
        args.use_crop,
        args.use_scale,
        args.use_compression,
        args.use_critic,
        args.use_adversary,
        str(int(time()))
    ))
    os.makedirs(log_dir, exist_ok=False)
    with open(os.path.join(log_dir, "config.json"), "wt") as fout:
        fout.write(json.dumps(args.__dict__, indent=2, default=lambda o: str(o)))

    # Generate metrics
    history = []
    for epoch in range(1, args.epochs + 1):
        metrics = {
            "train.loss": [], 
            "train.acc": [], 
            "val.acc": [], 
        }

        # Train
        gc.collect()
        iterator = tqdm(train, ncols=0)
        for frames in iterator:
            # generate data vectors
            frames = torch.cat([frames] * args.multiplicity, dim=0).cuda()
            data = torch.zeros((frames.size(0), args.data_dim)).random_(0, 2).cuda()
            frames = torch.cat([frames, frames], dim=0).cuda()
            data = torch.cat([data, 1.0 - data], dim=0).cuda()

            # random cropping
            w = random.choice(list(range(64, 128, 4)))
            h = random.choice(list(range(64, 128, 4)))
            frames = frames[:,:,:,:w,:h].cuda()
            data = data.cuda()
            
            # adversarial training
            if args.use_critic or args.use_adversary:
                loss = 0.0
                wm_frames = encoder(frames, data)
                if args.use_critic:
                    loss += torch.mean(critic(frames) - critic(wm_frames))
                if args.use_adversary:
                    loss -= F.binary_cross_entropy_with_logits(decoder(adversary(wm_frames)), data)
                xoptimizer.zero_grad()
                loss.backward()
                xoptimizer.step()
                for p in critic.parameters():
                    p.data.clamp_(-0.01, 0.01)

            # regular training
            loss = 0.0
            wm_frames = encoder(frames, data)
            wm_data = decoder(noise(wm_frames))
            loss += F.binary_cross_entropy_with_logits(wm_data, data)
            if args.use_critic:
                loss += torch.mean(critic(wm_frames))
            if args.use_adversary:
                loss += F.binary_cross_entropy_with_logits(decoder(adversary(wm_frames)), data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics["train.loss"].append(loss.item())
            metrics["train.acc"].append((wm_data >= 0.0).eq(data >= 0.5).sum().float().item() / data.numel())
            iterator.set_description("Epoch %s | Loss %s | Acc %s" % (
                epoch, 
                np.mean(metrics["train.loss"]), 
                np.mean(metrics["train.acc"])
            ))

        # Validate
        gc.collect()
        iterator = tqdm(val, ncols=0)
        with torch.no_grad():
            for frames in iterator:
                frames = frames.cuda()
                data = torch.zeros((frames.size(0), args.data_dim)).random_(0, 2).cuda()
                wm_frames = encoder(frames, data)
                wm_data = decoder(wm_frames)
                metrics["val.acc"].append((wm_data >= 0.0).eq(data >= 0.5).sum().float().item() / data.numel())
                iterator.set_description("Epoch %s | Loss %s | Acc %s | Val Acc %s" % (
                    epoch, 
                    np.mean(metrics["train.loss"]), 
                    np.mean(metrics["train.acc"]),
                    np.mean(metrics["val.acc"])
                ))

        metrics = {k: np.mean(v) for k, v in metrics.items()}
        metrics["epoch"] = epoch
        history.append(metrics)
        pd.DataFrame(history).to_csv(os.path.join(log_dir, "metrics.tsv"), index=False, sep="\t")
        
        torch.save((encoder, decoder, critic, adversary), os.path.join(log_dir, "model.pt"))
        xoptimizer_scheduler.step(metrics["train.loss"])
        optimizer_scheduler.step(metrics["train.loss"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=128)

    parser.add_argument('--seq_len', type=int, default=1)
    parser.add_argument('--data_dim', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--multiplicity', type=int, default=4)
    
    parser.add_argument('--use_crop', type=int, default=0)
    parser.add_argument('--use_scale', type=int, default=0)
    parser.add_argument('--use_compression', type=int, default=0)

    parser.add_argument('--use_critic', type=int, default=0)
    parser.add_argument('--use_adversary', type=int, default=0)

    run(parser.parse_args())
