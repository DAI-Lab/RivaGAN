import os
import cv2
import torch
from glob import glob
from random import randint
from torch.utils.data import Dataset, DataLoader

class VideoDataset(Dataset):
    """
    Given a folder of *.avi video files organized as shown below, this dataset
    selects randomly crops the video to `crop_size` and returns a random 
    continuous sequence of `seq_len` frames of shape.

        /root_dir
            1.avi
            2.avi

    The output has shape (3, seq_len, crop_size[0], crop_size[1]).
    """

    def __init__(self, root_dir, seq_len=10, crop_size=(128, 128)):
        self.seq_len = seq_len
        self.crop_size = crop_size
        
        self.videos = []
        for ext in ["avi", "mp4"]:
            for path in glob(os.path.join(root_dir, "**/*.%s" % ext), recursive=True):
                cap = cv2.VideoCapture(path)
                nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.videos.append((path, nb_frames))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        # Select time index
        path, nb_frames = self.videos[idx]
        start_idx = randint(0, nb_frames - self.seq_len - 1)
        
        # Select space index
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx-1)
        ok, frame = cap.read()
        H, W, D = frame.shape
        dx, dy = self.crop_size
        x = randint(0, W-dx-1)
        y = randint(0, H-dy-1)
        
        # Read frames and normalize to [-1.0, 1.0]
        frames = []
        for _ in range(self.seq_len):
            ok, frame = cap.read()
            frame = frame[y:y+dy,x:x+dx]
            frames.append(frame / 127.5 - 1.0)
        x = torch.FloatTensor(frames)
        x = x.permute(3, 0, 1, 2)
        return x

def load_train_val(seq_len, batch_size, dataset="hollywood2"):
    train = DataLoader(VideoDataset(
        "data/%s/train" % dataset, 
        crop_size=(128, 128), 
        seq_len=seq_len,
    ), shuffle=True, num_workers=8, batch_size=batch_size)
    val = DataLoader(VideoDataset(
        "data/%s/val" % dataset, 
        crop_size=(128, 128),
        seq_len=seq_len,
    ), shuffle=False, num_workers=8, batch_size=batch_size)
    return train, val
