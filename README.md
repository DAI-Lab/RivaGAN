<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<!-- Uncomment these lines after releasing the package to PyPI for version and downloads badges -->
<!--[![PyPI Shield](https://img.shields.io/pypi/v/rivagan.svg)](https://pypi.python.org/pypi/rivagan)-->
<!--[![Downloads](https://pepy.tech/badge/rivagan)](https://pepy.tech/project/rivagan)-->
<!--[![Coverage Status](https://codecov.io/gh/DAI-Lab/RivaGAN/branch/master/graph/badge.svg)](https://codecov.io/gh/DAI-Lab/RivaGAN)-->

[![Travis CI Shield](https://travis-ci.org/DAI-Lab/RivaGAN.svg?branch=master)](https://travis-ci.org/DAI-Lab/RivaGAN)


# RivaGAN

Robust Video Watermarking with Attention

- Free software: MIT license
- Documentation: https://DAI-Lab.github.io/RivaGAN
- Homepage: https://github.com/DAI-Lab/RivaGAN

# Overview

The goal of video watermarking is to embed a message within a video file in a
way such that it minimally impacts the viewing experience but can be recovered
even if the video is redistributed and modified, allowing media producers to assert
ownership over their content.

RivaGAN implements a novel architecture for robust video watermarking which features a
custom attention-based mechanism for embedding arbitrary data as well as two independent
adversarial networks which critique the video quality and optimize for robustness.

Using this technique, we are able to achieve state-of-the-art results in deep learning-based
video watermarking and produce watermarked videos which have minimal visual distortion and are
robust against common video processing operations.

# Install

## Requirements

**RivaGAN** has been developed and tested on [Python3.4, 3.5, 3.6 and 3.7](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system
in which **RivaGAN** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **RivaGAN**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) RivaGAN-venv
```

Afterwards, you have to execute this command to activate the virtualenv:

```bash
source RivaGAN-venv/bin/activate
```

Remember to execute it every time you start a new console to work on **RivaGAN**!

<!-- Uncomment this section after releasing the package to PyPI for installation instructions
## Install from PyPI

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **RivaGAN**:

```bash
pip install rivagan
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/).
-->

## Install from source

With your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:DAI-Lab/RivaGAN.git
cd RivaGAN
git checkout stable
make install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

Please head to the [Contributing Guide](https://DAI-Lab.github.io/RivaGAN/contributing.html#get-started)
for more details about this process.

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started training your own instance of **RivaGAN**.

## Download the training data

Start by running the following commands to automatically download the Hollywood2 and Moments in
Time datasets. Depending on the speed of your internet connection, this may take up to an hour.

```
cd data
bash download.sh
```

## Train a model

Now you're ready to train a model.

Make sure to having activated your virtualenv and installed the project, and then
execute the following python commands:

```
from rivagan import RivaGAN

model = RivaGAN()
model.fit("data/hollywood2", epochs=300)
model.save("/path/to/model.pt")
```

Make sure to replace the `/path/to/model.pt` string with an appropiate save path.

## Encode data in a video

You can now load the trained model and use it as follows:

```
data = tuple([0] * 32)
model = RivaGAN.load("/path/to/model.pt")
model.encode("/path/to/video.avi", data, "/path/to/output.avi")
```

## Decode data from the video

After the data is encoded in the video, it can be recovered as follows:

```
recovered_data = model.decode("/path/to/output.avi"):
```


# Citing RivaGAN

If you use RivaGAN for your research, please consider citing the following work:

Zhang, Kevin Alex and Xu, Lei and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan. Robust
Invisible Video Watermarking with Attention. MIT EECS, September 2019. ([PDF](https://arxiv.org/abs/1909.01285))

```
@article{zhang2019robust,
    author={Kevin Alex Zhang and Lei Xu and Alfredo Cuesta-Infante and Kalyan Veeramachaneni},
    title={Robust Invisible Video Watermarking with Attention},
    year={2019},
    eprint={1909.01285},
    archivePrefix={arXiv},
    primaryClass={cs.MM}
}
```

# What's next?

For more details about **RivaGAN** and all its possibilities
and features, please check the [documentation site](
https://DAI-Lab.github.io/RivaGAN/).
