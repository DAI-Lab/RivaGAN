ffmpeg -i source.avi source/%d.png
ffmpeg -i watermarked.avi watermarked/%d.png
python generate_diffs.py
