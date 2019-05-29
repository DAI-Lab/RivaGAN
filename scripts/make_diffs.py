import sys; sys.path.append('../')
import torch
import imageio
import numpy as np
from glob import glob
from data import load_train_val
from model.utils import ssim, psnr, mjpeg
from model.noise import Crop, Scale, Compression
from model import AttentiveEncoder, AttentiveDecoder

seq_len = 24
batch_size = 1
dataset = "hollywood2"
_, val = load_train_val(seq_len, batch_size, dataset)

crop = Crop()
scale = Scale()
compress = Compression()

def get_acc(y_true, y_pred):
    assert y_true.size() == y_pred.size()
    return (y_pred >= 0.0).eq(y_true >= 0.5).sum().float().item() / y_pred.numel()

def tensor_to_image(x, i=0):
    x = torch.clamp(x[0,:,i,:,:].permute(1,2,0), min=-1.0, max=1.0)
    x = ((x + 1.0) * 127.5).cpu().int().numpy()
    return x[:,:,::-1] # BGR -> RGB

def make_examples(path_to_model):
    encoder, decoder, _, _ = torch.load(path_to_model)

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for frames in val:
            frames = frames.cuda()

            data = torch.zeros((frames.size(0), encoder.data_dim)).random_(0, 2).cuda()
            wm_frames = encoder(frames, data)

            imageio.imwrite(path_to_model.replace("model.pt", "0_source.png"), tensor_to_image(frames, i=0))
            imageio.imwrite(path_to_model.replace("model.pt", "0_diff.png"), tensor_to_image(torch.abs(frames - wm_frames), i=0))

            imageio.imwrite(path_to_model.replace("model.pt", "1_source.png"), tensor_to_image(frames, i=5))
            imageio.imwrite(path_to_model.replace("model.pt", "1_diff.png"), tensor_to_image(torch.abs(frames - wm_frames), i=5))

            imageio.imwrite(path_to_model.replace("model.pt", "2_source.png"), tensor_to_image(frames, i=10))
            imageio.imwrite(path_to_model.replace("model.pt", "2_diff.png"), tensor_to_image(torch.abs(frames - wm_frames), i=10))

            imageio.imwrite(path_to_model.replace("model.pt", "3_source.png"), tensor_to_image(frames, i=15))
            imageio.imwrite(path_to_model.replace("model.pt", "3_diff.png"), tensor_to_image(torch.abs(frames - wm_frames), i=15))

            imageio.imwrite(path_to_model.replace("model.pt", "4_source.png"), tensor_to_image(frames, i=20))
            imageio.imwrite(path_to_model.replace("model.pt", "4_diff.png"), tensor_to_image(torch.abs(frames - wm_frames), i=20))

            break

for path_to_model in glob("../results/1558293718/*.pt"):
    make_examples(path_to_model)
