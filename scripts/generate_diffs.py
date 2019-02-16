"""
Usage:

 - Place `watermarked.avi` and `source.avi` in this directory.
 - Make sure you have `ffmpeg` installed.
 - Run `python_generate_difsf.py.`
 - The image diffs will be generated into `diff/*`.

"""
import subprocess
import numpy as np
from imageio import imread, imwrite

subprocess.call("""
mkdir diff
mkdir source
mkdir watermarked
ffmpeg -i source.avi source/%d.png
ffmpeg -i watermarked.avi watermarked/%d.png
""", shell=True)

for i, j in enumerate([10, 30, 50, 70, 90]):
    source = imread("source/%s.png" % j)
    watermark = imread("watermarked/%s.png" % j)
    imwrite("diff/%s_source.png" % i, source)
    imwrite("diff/%s_watermark.png" % i, watermark)
    imwrite("diff/%s_diff.png" % i, np.abs(source.astype(float) - watermark.astype(float)).astype(int))

subprocess.call("""
rm -r source
rm -r watermarked
""", shell=True)
