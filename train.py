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
torch.backends.cudnn.benchmark = True

def make_pair(frames, args):
    # Add multiplicity to stabilize training.
    frames = torch.cat([frames] * args.multiplicity, dim=0).cuda()
    data = torch.zeros((frames.size(0), args.data_dim)).random_(0, 2).cuda()

    # Add the bit-inverse to further stabilize training.
    frames = torch.cat([frames, frames], dim=0).cuda()
    data = torch.cat([data, 1.0 - data], dim=0).cuda()

    return frames, data

def run(args):
    # Load our datasets
    train, val = load_train_val(args.seq_len, args.batch_size, args.dataset)

    # Initialize our noise layers
    _noise = [Quantization()]
    if args.use_crop: _noise.append(Crop())
    if args.use_scale: _noise.append(Scale())
    if args.use_jpeg: _noise.append(JpegCompression(torch.device("cuda")))
        
    def noise(x):
        for layer in _noise:
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
    log_dir = os.path.join("results/", "m%s%s-n%s%s%s-%s" % (
        args.use_critic,
        args.use_adversary,
        args.use_jpeg,
        args.use_crop,
        args.use_scale,
        str(int(time()))
    ))
    os.makedirs(log_dir, exist_ok=False)
    with open(os.path.join(log_dir, "config.json"), "wt") as fout:
        fout.write(json.dumps(args.__dict__, indent=2, default=lambda o: str(o)))

    # Optimize the model
    history = []
    for epoch in range(1, args.epochs + 1):
        metrics = {
            "train.loss": [], 
            "train.raw_acc": [], 
            "train.mjpeg_acc": [], 
            "train.noise_acc": [], 
            "val.raw_acc": [], 
            "val.mjpeg_acc": [], 
            "val.noise_acc": [],
            "val.quantized_acc": [],
            "val.ssim": [],
            "val.psnr": []
        }

        # Train
        gc.collect()
        iterator = tqdm(train, ncols=0)
        for frames in iterator:
            # Pair each sequence of frames with a data vector
            frames, data = make_pair(frames, args)

            # Adversarial training to optimize quality / robustness.
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

            # Standard training to maximize accuracy.
            loss = 0.0
            wm_frames = encoder(frames, data)
            wm_data = decoder(wm_frames)
            wm_mjpeg_data = decoder(mjpeg(wm_frames))
            wm_noise_data = decoder(noise(wm_frames))
            loss += F.binary_cross_entropy_with_logits(wm_data, data)
            if args.use_jpeg:
                loss += F.binary_cross_entropy_with_logits(wm_mjpeg_data, data)
            loss += F.binary_cross_entropy_with_logits(wm_noise_data, data)
            if args.use_critic:
                loss += torch.mean(critic(wm_frames))
            if args.use_adversary:
                loss += F.binary_cross_entropy_with_logits(decoder(adversary(wm_frames)), data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics["train.loss"].append(loss.item())
            metrics["train.raw_acc"].append((wm_data >= 0.0).eq(data >= 0.5).sum().float().item() / data.numel())
            metrics["train.mjpeg_acc"].append((wm_mjpeg_data >= 0.0).eq(data >= 0.5).sum().float().item() / data.numel())
            metrics["train.noise_acc"].append((wm_noise_data >= 0.0).eq(data >= 0.5).sum().float().item() / data.numel())

            iterator.set_description("%s | Loss %.3f | Raw %.3f | MJPEG %.3f | Noise %.3f" % (
                epoch, 
                np.mean(metrics["train.loss"]), 
                np.mean(metrics["train.raw_acc"]),
                np.mean(metrics["train.mjpeg_acc"]),
                np.mean(metrics["train.noise_acc"]),
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
                wm_mjpeg_data = decoder(mjpeg(wm_frames))
                wm_noise_data = decoder(noise(wm_frames))
                wm_quantized_data = decoder(((wm_frames + 1.0) * 127.5).int().float() / 127.5 - 1.0)

                metrics["val.raw_acc"].append((wm_data >= 0.0).eq(data >= 0.5).sum().float().item() / data.numel())
                metrics["val.mjpeg_acc"].append((wm_mjpeg_data >= 0.0).eq(data >= 0.5).sum().float().item() / data.numel())
                metrics["val.noise_acc"].append((wm_noise_data >= 0.0).eq(data >= 0.5).sum().float().item() / data.numel())
                metrics["val.quantized_acc"].append((wm_quantized_data >= 0.0).eq(data >= 0.5).sum().float().item() / data.numel())

                metrics["val.ssim"].append(ssim(frames[:,:,0,:,:], wm_frames[:,:,0,:,:]).item())
                metrics["val.psnr"].append(psnr(frames[:,:,0,:,:], wm_frames[:,:,0,:,:]).item())

                iterator.set_description("%s | Raw %.3f | MJPEG %.3f | Noise %.3f | Quantized %.3f | SSIM %.3f | PSNR %.3f" % (
                    epoch, 
                    np.mean(metrics["val.raw_acc"]),
                    np.mean(metrics["val.mjpeg_acc"]),
                    np.mean(metrics["val.noise_acc"]),
                    np.mean(metrics["val.quantized_acc"]),
                    np.mean(metrics["val.ssim"]),
                    np.mean(metrics["val.psnr"]),
                ))

        metrics = {k: round(np.mean(v), 3) for k, v in metrics.items()}
        metrics["epoch"] = epoch
        history.append(metrics)
        pd.DataFrame(history).to_csv(os.path.join(log_dir, "metrics.tsv"), index=False, sep="\t")
        
        torch.save((encoder, decoder, critic, adversary), os.path.join(log_dir, "model.pt"))
        xoptimizer_scheduler.step(metrics["train.loss"])
        optimizer_scheduler.step(metrics["train.loss"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--dataset', type=str, default="hollywood2")

    parser.add_argument('--seq_len', type=int, default=1)
    parser.add_argument('--data_dim', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--multiplicity', type=int, default=1)
    
    parser.add_argument('--use_critic', type=int, default=0)
    parser.add_argument('--use_adversary', type=int, default=0)

    parser.add_argument('--use_jpeg', type=int, default=0)
    parser.add_argument('--use_crop', type=int, default=0)
    parser.add_argument('--use_scale', type=int, default=0)

    run(parser.parse_args())
