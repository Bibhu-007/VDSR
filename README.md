# VDSR-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation of [Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://arxiv.org/abs/1511.04587).

### Table of contents

- [VDSR-PyTorch](#vdsr-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About Accurate Image Super-Resolution Using Very Deep Convolutional Networks](#about-accelerating-the-super-resolution-convolutional-neural-network)
    - [File descriptions](#file-description)
    - [Folder structure](#folder-structure)
    - [Training dataset](#training-dataset)
    - [Testing dataset](#testing-dataset)
    - [Test](#test)
    - [Train](#train)
    - [Result](#result)
    - [Credit](#credit)
    - [Note](#note)
        - [Accurate Image Super-Resolution Using Very Deep Convolutional Neural Network](#accurate-image-super-resolution-using-very-deep-convolutional-networks)


## About Accelerating the Super-Resolution Convolutional Neural Network

Abstract straight from the paper:

We present a highly accurate single-image super-resolution (SR) method. Our method uses a very deep convolutional network inspired by VGG-net used for ImageNet classification. We find increasing our network depth shows a significant improvement in accuracy. Our final model uses 20 weight layers. By cascading small filters many times in a deep network structure, contextual information over large image regions is exploited in an efficient way. With very deep networks, however, convergence speed becomes a critical issue during training. We propose a simple yet effective training procedure. We learn residuals only and use extremely high learning rates (104 times higher than SRCNN) enabled by adjustable gradient clipping. Our proposed method performs better than existing methods in accuracy, and visual improvements in our results are easily noticeable.

## File descriptions

- `requirements.txt`: Contains the required packages. Install with: `pip install -r requirements.txt`.
- `train.py`: Main training script for the VDSR model. Initializes the dataset, model, and optimizer, performs training iterations, saves checkpoints, and logs loss and PSNR metrics.
- `validate.py`: Runs inference on benchmarking datasets using a trained model. Calculates PSNR and SSIM metrics and saves the super-resolved images.
- `dataset.py`: Defines the PyTorch `dataset` class for data loading, data preprocessing, and data augmentation.
- `model.py`: Contains the implementation of the VDSR architecture.
- `imgproc.py`: Includes image processing utilities such as bicubic upscaling, resizing, color conversions, etc.
- `Interpolation-based methods.ipynb`: Contains scripts for calculating PSNR and SSIM metrics on benchmark datasets using bicubic interpolation, nearest neighbor interpolation, and Lanczos resampling. Update the lr, hr, and sr directories as required.
- `scripts`: Contains `.py` files for generating the training and validation data.
- `DIV2K`: Contains `dataset_patches.py` and the modified `config.py` and `train.py` files to train the model without preloading the entire dataset into memory. To replicate this setup, copy the files into the main directory and run the train script as usual. (The modification was made to address long data preloading times and is optional.)


## Folder Structure
## Train dataset folder structure

```text
data/
└── Training and Validation/
    └── VDSR/
        ├── Dataset/
        ├── Scale 2/
        │   ├── train/
        │   │   ├── hr/
        │   │   └── lr/
        │   └── valid/
        │       ├── hr/
        │       └── lr/
        └── Scale 4/
            ├── train/
            │   ├── hr/
            │   └── lr/
            └── valid/
                ├── hr/
                └── lr/

```

## Test dataset folder structure

```text
data/
└── Testing/
    ├── Scale 2/
    │   ├── Set 5/
    │   │   ├── HR/
    │   │   └── LR/
    │   └── Urban 100/
    │       ├── HR/
    │       └── LR/
    └── Scale 4/
        ├── Set 5/
        │   ├── HR/
        │   └── LR/
        └── Urban 100/
            ├── HR/
            └── LR/

```


## Training dataset
For training and validation, a combination of T-91 and BSD200 datasets have been used.
Creating training dataset:
- Place that training HR images in "data/Training and Validation/VDSR/Dataset" & run "scripts/run.py".


## Testing dataset
Testing has been performed on the Set5 and Urban100 datasets.
Follow the folder structure as mentioned above and place the HR-LR images separately.
Modify config.py as follows:
- line 43: Update the directory as per the choice of test dataset.
- line 73: Update the directory as per the choice of test dataset.
- line 74: Update the directory as per the choice of test dataset.


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
- line 49: `start_epoch` change the number of training iterations in the previous round.
- line 50: `resume` change the weight address that needs to be loaded.

## Result
| Dataset  | Scale |     Avg.PSNR(Gray)     |     Avg.PSNR(RGB)     |     Avg.SSIM(Grayscale)     |     Avg.SSIM(RGB)     |
|:--------:|:-----:|:----------------------:|:---------------------:|:---------------------------:|:---------------------:|
|   Set5   |   2   |         37.16          |         35.73         |            0.9621           |         0.9600        |
|   Set5   |   4   |         30.84          |         29.49         |            0.8855           |         0.8768        |
| Urban100 |   2   |         29.78          |         28.46         |            0.9188           |         0.9127        |
| Urban100 |   4   |         24.80          |         23.44         |            0.7434           |         0.7412        |

## Note
Place the `best.pth.tar` file (pre-trained files) in `results/vdsr_baseline/Scale X` directory and run `validate.py` to reconfirm the results.

Pre-trained weights and sample super-resolved images: https://drive.google.com/drive/folders/1wUIY2U5XbdyLyl7Kq_y9IUwIYiOgMQS3?usp=sharing

### Credit

#### Accurate Image Super-Resolution Using Very Deep Convolutional Networks

_Jiwon Kim, Jung Kwon Lee, Kyoung Mu Lee_ <br>

[[Paper]](https://ieeexplore.ieee.org/document/7780551)
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

Reference of the PyTorch implementation: https://github.com/Lornatang/VDSR-PyTorch.git
