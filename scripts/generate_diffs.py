import numpy as np
from imageio import imread, imwrite

for i, j in enumerate([210, 220, 230, 240, 250]):
    source = imread("source/%s.png" % j)
    watermark = imread("watermarked/%s.png" % j)
    imwrite("diff/%s_source.png" % i, source)
    imwrite("diff/%s_watermark.png" % i, watermark)
    imwrite("diff/%s_diff.png" % i, np.abs(source.astype(float) - watermark.astype(float)).astype(int))
