import sys
import json
import pandas as pd
from glob import glob

rows = []
for path_to_output in sorted(glob("results/**/val.json")):
    output = json.load(open(path_to_output, "rt"))
    config = json.load(open(path_to_output.replace("val.json", "config.json"), "rt"))
    rows.append({**output, **config})
df = pd.DataFrame(rows)
df.to_csv("results/summary.csv", index=False)
print(df)
