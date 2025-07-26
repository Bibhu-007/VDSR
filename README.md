# VDSR-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation of [Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://arxiv.org/abs/1511.04587).

### Table of contents

- [VDSR-PyTorch](#vdsr-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About Accurate Image Super-Resolution Using Very Deep Convolutional Networks](#about-accelerating-the-super-resolution-convolutional-neural-network)
    - [Training dataset](#test)
    - [Testing dataset](#test)
    - [Test](#test)
    - [Train](#train)
    - [Result](#result)
    - [Credit](#credit)
        - [Accelerating the Super-Resolution Convolutional Neural Network](#accurate-image-super-resolution-using-very-deep-convolutional-networks)

## About Accelerating the Super-Resolution Convolutional Neural Network

If you're new to VDSR, here's an abstract straight from the paper:

We present a highly accurate single-image superresolution (SR) method. Our method uses a very deep convolutional network inspired by VGG-net used for
ImageNet classification. We find increasing our network depth shows a significant improvement in accuracy. Our finalmodel uses 20 weight layers. By
cascading small filters many times in a deep network structure, contextual information over large image regions is exploited in an efficient way. With
very deep networks, however, convergence speed becomes a critical issue during training. We propose a simple yet effective training procedure. We
learn residuals onlyb and use extremely high learning rates
(104 times higher than SRCNN) enabled by adjustable gradient clipping. Our proposed method performs better than existing methods in accuracy and
visual improvements in our results are easily noticeable.

## Training dataset
For training and validation a combination of T-91. BSD 200, and Technik datasets have been used.
Modify config.py as follows:
- line 40: Update directory as per the desired folder structure.
- line 41: Update directory as per the desired folder structure.


## Testing dataset
Testing has been performed on Set 5 and Urban 100 dataset.
Modify config.py as follows:
- line 72: Update directory as per the desired folder structure.
- line 73: Update directory as per the desired folder structure.


## Test
Modify config.py as follows:

- line 31: `upscale_factor` change to the magnification you need to enlarge.
- line 33: `mode` change Set to "valid" mode.

## Train
Modify config.py as follows:

- line 31: `upscale_factor` change to the magnification you need to enlarge.
- line 33: `mode` change Set to "train" mode.

If you want to load weights that you've trained before, modify the contents of the file as follows.
Modify config.py as follows:
- line 49: `start_epoch` change number of training iterations in the previous round.
- line 50: `resume` change weight address that needs to be loaded.

## Result
| Dataset | Scale |     Avg.PSNR     |     Avg.SSIM(Grayscale)     |     Avg.SSIM(RGB)     |
|:-------:|:-----:|:----------------:|:---------------------------:|:---------------------:|
|  Set5   |   2   |       36.06      |             0.96            |          0.95         |
|  Set5   |   4   |       28.27      |             0.89            |          0.89         |
|Urban 100|   2   |       29.95      |             0.86            |          0.85         |
|Urban 100|   4   |       24.02      |             0.72            |          0.70         |

### Credit

#### Accurate Image Super-Resolution Using Very Deep Convolutional Networks

_Jiwon Kim, Jung Kwon Lee, Kyoung Mu Lee_ <br>

**Abstract** <br>
We present a highly accurate single-image superresolution (SR) method. Our method uses a very deep convolutional network inspired by VGG-net used for
ImageNet classification. We find increasing our network depth shows a significant improvement in accuracy. Our finalmodel uses 20 weight layers. By
cascading small filters many times in a deep network structure, contextual information over large image regions is exploited in an efficient way. With
very deep networks, however, convergence speed becomes a critical issue during training. We propose a simple yet effective training procedure. We
learn residuals onlyb and use extremely high learning rates
(104 times higher than SRCNN) enabled by adjustable gradient clipping. Our proposed method performs better than existing methods in accuracy and
visual improvements in our results are easily noticeable.

[[Paper]]([https://arxiv.org/pdf/1511.04587](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780551))
```
@InProceedings{Lee2016,
  author    = {Kim, Jiwon and Lee, Jung Kwon and Lee, Kyoung Mu},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  title     = {Accurate Image Super-Resolution Using Very Deep Convolutional Networks},
  year      = {2016},
  month     = dec,
  doi       = {10.1109/cvpr.2016.182},
}
```
[[Author's implements(MATLAB)]](https://cv.snu.ac.kr/research/VDSR/VDSR_code.zip)

```
@inproceedings{vedaldi15matconvnet,
  author    = {A. Vedaldi and K. Lenc},
  title     = {MatConvNet -- Convolutional Neural Networks for MATLAB},
  booktitle = {Proceeding of the {ACM} Int. Conf. on Multimedia},
  year      = {2015},
}
```

Referance of implementation: https://github.com/Lornatang/VDSR-PyTorch.git
