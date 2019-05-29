import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

bit_inverse = pd.read_csv("../results/1556823778/metrics.tsv", sep="\t")
no_bit_inverse = pd.read_csv("../results/1557020399/metrics.tsv", sep="\t")
bit_inverse_lr = pd.read_csv("../results/1557330872/metrics.tsv", sep="\t")
no_bit_inverse_lr = pd.read_csv("../results/1557103447/metrics.tsv", sep="\t")

bit_inverse = bit_inverse.iloc[:min(100, len(bit_inverse))]
no_bit_inverse = no_bit_inverse.iloc[:min(200, len(no_bit_inverse))]
bit_inverse_lr = bit_inverse_lr.iloc[:min(100, len(bit_inverse_lr))]
no_bit_inverse_lr = no_bit_inverse_lr.iloc[:min(200, len(no_bit_inverse_lr))]

fig, axes = plt.subplots(1, 2, figsize=(10,3))

axes[0].plot(np.arange(0, 2*len(bit_inverse), 2), bit_inverse[["train.loss"]], label="Bit Inverse (LR = 1e-4)")
axes[0].plot(np.arange(0, 2*len(bit_inverse_lr), 2), bit_inverse_lr[["train.loss"]], label="Bit Inverse (LR = 1e-3)")
axes[0].plot(no_bit_inverse[["train.loss"]], label="No Bit Inverse (LR = 1e-4)")
axes[0].plot(no_bit_inverse_lr[["train.loss"]], label="No Bit Inverse (LR = 1e-3)")
axes[0].set_xlabel("Wall Clock Time")
axes[0].set_ylabel("Train Loss")
axes[0].legend(loc="lower left")

axes[1].plot(np.arange(0, 2*len(bit_inverse), 2), bit_inverse[["val.mjpeg_acc"]], label="Bit Inverse (LR = 1e-4)")
axes[1].plot(np.arange(0, 2*len(bit_inverse_lr), 2), bit_inverse_lr[["val.mjpeg_acc"]], label="Bit Inverse (LR = 1e-3)")
axes[1].plot(no_bit_inverse[["val.mjpeg_acc"]], label="No Bit Inverse (LR = 1e-4)")
axes[1].plot(no_bit_inverse_lr[["val.mjpeg_acc"]], label="No Bit Inverse (LR = 1e-3)")
axes[1].set_xlabel("Wall Clock Time")
axes[1].set_ylabel("Test Accuracy")
#axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig("make_bit_inverse.png")
