import sys
import json
import pandas as pd
from glob import glob

rows = []
for path_to_output in glob("results/**/output.json"):
    output = json.load(open(path_to_output, "rt"))
    config = json.load(open(path_to_output.replace("output.json", "config.json"), "rt"))
    rows.append({**output, **config})
df = pd.DataFrame(rows)
del df["seq_len"], df["lr"], df["epochs"], df["data_dim"], df["batch_size"]
df = df.sort_values(["use_shortcut", "use_adversary", "use_critic", "use_crop", "use_scale", "use_compression"])
df["acc.aspect_ratio"] = round(df["acc.aspect_ratio"], 2)
df["acc.frame_rate"] = round(df["acc.frame_rate"], 2)
df["acc.identity"] = round(df["acc.identity"], 2)
df["acc.transcoded"] = round(df["acc.transcoded"], 2)
df["psnr"] = round(df["psnr"], 2)
df["ssim"] = round(df["ssim"], 2)
df.to_csv("results/summary.csv", index=False)
