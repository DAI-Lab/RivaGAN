import sys; sys.path.append('../')
import torch
import imageio
import numpy as np
from glob import glob
from data import load_train_val
from model.utils import ssim, psnr, mjpeg
from model.noise import Crop, Scale, Compression
from model import AttentiveEncoder, AttentiveDecoder

seq_len = 1
batch_size = 1
dataset = "hollywood2"
_, val = load_train_val(seq_len, batch_size, dataset)

crop = Crop()
scale = Scale()
compress = Compression()

def get_acc(y_true, y_pred):
    assert y_true.size() == y_pred.size()
    return (y_pred >= 0.0).eq(y_true >= 0.5).sum().float().item() / y_pred.numel()

def tensor_to_image(x):
    x = torch.clamp(x[0,:,0,:,:].permute(1,2,0), min=-1.0, max=1.0)
    x = ((x + 1.0) * 127.5).cpu().int().numpy()
    return x[:,:,::-1] # BGR -> RGB

def make_examples(path_to_model):
    encoder, decoder, _, _ = torch.load(path_to_model)

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for frames in val:
            source = frames.clone()
            frames = frames.cuda()

            data = torch.zeros((frames.size(0), encoder.data_dim)).random_(0, 2).cuda()
            wm_frames_1 = encoder(frames, data)
            print(get_acc(data, decoder(wm_frames_1)))

            data[:,0] = 1 - data[:,0] # Flip the first bit
            wm_frames_2 = encoder(frames, data)
            print(get_acc(data, decoder(wm_frames_2)))

            imageio.imwrite(path_to_model.replace("model.pt", "image_in.png"), tensor_to_image(source))
            imageio.imwrite(path_to_model.replace("model.pt", "image_out.png"), tensor_to_image(wm_frames_1))
            imageio.imwrite(path_to_model.replace("model.pt", "image_out_alt.png"), tensor_to_image(wm_frames_2))

            diff_1 = tensor_to_image(wm_frames_1) - tensor_to_image(frames)
            diff_2 = tensor_to_image(wm_frames_2) - tensor_to_image(frames)
            diff = 5 + (diff_1 - diff_2)
            imageio.imwrite(path_to_model.replace("model.pt", "image_out_diff.png"), diff[:,:,0])

            # ----
            data[:,0] = 1 - data[:,0] # Flip the first bit
            data[:,1] = 1 - data[:,1] # Flip the second bit
            wm_frames_2 = encoder(frames, data)
            diff_1 = tensor_to_image(wm_frames_1) - tensor_to_image(frames)
            diff_2 = tensor_to_image(wm_frames_2) - tensor_to_image(frames)
            diff = 5 + (diff_1 - diff_2)
            imageio.imwrite(path_to_model.replace("model.pt", "image_out_diff_alt.png"), diff[:,:,0])
            # ----

            np.savez("examples.npz", source=tensor_to_image(frames), wm_1=tensor_to_image(wm_frames_1), wm_2=tensor_to_image(wm_frames_2))
            break

for path_to_model in glob("../results/1558293718/*.pt"):
    make_examples(path_to_model)
