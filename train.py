import gc
import os
import json
import numpy as np
import pandas as pd
import argparse
import random
from time import time
from tqdm import tqdm
from itertools import chain

import torch
import torch.optim as optim
import torch.nn.functional as F
from data import load_train_val
from model import Encoder, Decoder
from model.utils import ssim, psnr, mjpeg
from model.attention import AttentiveEncoder, AttentiveDecoder

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def get_acc(y_true, y_pred):
    assert y_true.size() == y_pred.size()
    return (y_pred >= 0.0).eq(y_true >= 0.5).sum().float().item() / y_pred.numel()

def quantize(frames):
    return ((frames + 1.0) * 127.5).int().float() / 127.5 - 1.0

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

    # Initialize our modules and optimizers
    encoder = Encoder(data_dim=args.data_dim, combiner=args.combiner).cuda()
    decoder = Decoder(data_dim=args.data_dim).cuda()
    if args.attention:
        encoder = AttentiveEncoder(data_dim=args.data_dim).cuda()
        decoder = AttentiveDecoder(encoder).cuda()

    optimizer = optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Set up the log directory
    log_dir = os.path.join("results/", "%s" % (
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
            "val.ssim": [],
            "val.psnr": [],
            "val.mjpeg_acc": [], 
        }

        # Train
        gc.collect()
        encoder.train()
        decoder.train()
        iterator = tqdm(train, ncols=0)
        for frames in iterator:
            frames, data = make_pair(frames, args)

            wm_frames = encoder(frames, data)
            wm_raw_data = decoder(wm_frames)
            wm_mjpeg_data = decoder(mjpeg(wm_frames))

            loss = 0.0
            loss += F.binary_cross_entropy_with_logits(wm_raw_data, data)
            loss += F.binary_cross_entropy_with_logits(wm_mjpeg_data, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics["train.loss"].append(loss.item())
            metrics["train.raw_acc"].append(get_acc(data, wm_raw_data))
            metrics["train.mjpeg_acc"].append(get_acc(data, wm_mjpeg_data))
            iterator.set_description("%s | Loss %.3f | Raw %.3f | MJPEG %.3f" % (
                epoch, 
                np.mean(metrics["train.loss"]), 
                np.mean(metrics["train.raw_acc"]),
                np.mean(metrics["train.mjpeg_acc"]),
            ))

        # Validate
        gc.collect()
        encoder.eval()
        decoder.eval()
        iterator = tqdm(val, ncols=0)
        with torch.no_grad():
            for frames in iterator:
                frames = frames.cuda()
                data = torch.zeros((frames.size(0), args.data_dim)).random_(0, 2).cuda()

                wm_frames = encoder(frames, data)
                wm_mjpeg_data = decoder(mjpeg(wm_frames))

                metrics["val.ssim"].append(ssim(frames[:,:,0,:,:], wm_frames[:,:,0,:,:]).item())
                metrics["val.psnr"].append(psnr(frames[:,:,0,:,:], wm_frames[:,:,0,:,:]).item())
                metrics["val.mjpeg_acc"].append(get_acc(data, wm_mjpeg_data))

                iterator.set_description("%s | SSIM %.3f | PSNR %.3f | MJPEG %.3f" % (
                    epoch, 
                    np.mean(metrics["val.ssim"]),
                    np.mean(metrics["val.psnr"]),
                    np.mean(metrics["val.mjpeg_acc"]),
                ))

        metrics = {k: round(np.mean(v), 3) for k, v in metrics.items()}
        metrics["epoch"] = epoch
        history.append(metrics)
        pd.DataFrame(history).to_csv(os.path.join(log_dir, "metrics.tsv"), index=False, sep="\t")
        
        torch.save((encoder, decoder), os.path.join(log_dir, "model.pt"))
        scheduler.step(metrics["train.loss"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--dataset', type=str, default="hollywood2")

    parser.add_argument('--seq_len', type=int, default=1)
    parser.add_argument('--data_dim', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--multiplicity', type=int, default=1)
    parser.add_argument('--attention', type=int, default=0)
    parser.add_argument('--combiner', type=str, default="spatial_repeat", help="spatial_repeat | multiplicative")

    run(parser.parse_args())
