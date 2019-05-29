[![PyPI Shield](https://img.shields.io/pypi/v/rivagan.svg)](https://pypi.python.org/pypi/rivagan)
[![Travis CI Shield](https://travis-ci.org/DAI-Lab/rivagan.svg?branch=master)](https://travis-ci.org/DAI-Lab/rivagan)

# RivaGAN

Robust Video Watermarking with Attention

- Free software: MIT license
- Documentation: https://DAI-Lab.github.io/rivagan
- Homepage: https://github.com/DAI-Lab/rivagan

## Training
Start by running the following commands to automatically download the Hollywood2 and Moments in 
Time datasets. Depending on the speed of your internet connection, this may take up to an hour.

```
cd data
bash download.sh
```

Set up a new conda environment and install RivaGAN in development mode by running:

```
conda create -n pytorch python=3.6 anaconda
source activate pytorch
make install-develop
```

Now you're ready to train a model.

```
from rivagan import RivaGAN
model = RivaGAN()
model.fit("data/hollywood2", epochs=300)
model.save("/path/to/model.pt")
```

You can load the trained model and use it as follows:

```
data = tuple([0] * 32)
model = RivaGAN.load("/path/to/model.pt")
model.encode("/path/to/video.avi", data, "/path/to/output.avi")
for recovered_data in model.decode("/path/to/output.avi"):
    assert recovered_data == data
```
