# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""File description: Realize the verification function after model training."""
import os
import cv2
import numpy as np
import torch
from natsort import natsorted

import config
import imgproc
from model import VDSR
from train import calculate_ssim
from skimage.metrics import structural_similarity as compare_ssim
from train import save_comparision_images


def main() -> None:
    # Initialize the super-resolution model
    model = VDSR().to(config.device)
    print("Build VDSR model successfully.")

    # Load the super-resolution model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load VDSR model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Start the verification mode of the model.
    model.eval()
    # Turn on half-precision inference.
    model.half()

    # Initialize the image evaluation index.
    total_psnr_grayscale = 0.0
    total_psnr_rgb = 0.0
    total_ssim = 0.0
    total_ssim_rgb = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.hr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        filename = file_names[index]
        hr_image_path = os.path.join(config.hr_dir, filename)
        sr_filename = filename.replace("_HR", "_SR")
        sr_image_path = os.path.join(config.sr_dir, sr_filename)
        lr_filename = filename.replace("_HR", "_LR")
        lr_image_path = os.path.join(config.lr_dir, lr_filename)

        print(f"Processing `{os.path.abspath(hr_image_path)}`...")
        # Make high-resolution image
        hr_image = cv2.imread(hr_image_path).astype(np.float32) / 255.0

        # Get low-resolution image
        lr_image = cv2.imread(lr_image_path).astype(np.float32) / 255.0
        lr_image = imgproc.imresize(lr_image, config.upscale_factor)

        # Convert BGR image to YCbCr image
        lr_ycbcr_image = imgproc.bgr2ycbcr(lr_image, use_y_channel=False)
        hr_ycbcr_image = imgproc.bgr2ycbcr(hr_image, use_y_channel=False)

        # Split YCbCr image data
        lr_y_image, lr_cb_image, lr_cr_image = cv2.split(lr_ycbcr_image)
        hr_y_image, hr_cb_image, hr_cr_image = cv2.split(hr_ycbcr_image)

        # Convert Y image data convert to Y tensor data
        lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)
        hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_y_tensor = model(lr_y_tensor).clamp_(0, 1.0)

        ## Grayscale SSIM Calculation 
        psnr_val_grayscale = 10. * torch.log10(1. / torch.mean((sr_y_tensor - hr_y_tensor) ** 2))
        total_psnr_grayscale += psnr_val_grayscale
        

        # Save image
        sr_y_image = imgproc.tensor2image(sr_y_tensor, range_norm=False, half=True)
        sr_y_image = sr_y_image.astype(np.float32) / 255.0
        sr_ycbcr_image = cv2.merge([sr_y_image, hr_cb_image, hr_cr_image])
        sr_image = imgproc.ycbcr2bgr(sr_ycbcr_image)
        cv2.imwrite(sr_image_path, sr_image * 255.0)
    
        ## Dimensioning
        if sr_y_image.ndim == 3 and sr_y_image.shape[2] == 1:
            sr_y_image = sr_y_image[:, :, 0]  # Or np.squeeze(sr_y_image, axis=2)
            
        ## Visualize Residuals
        residual = sr_y_image - lr_y_image
        vis_residual = cv2.normalize(residual, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        vis_residual = vis_residual.astype(np.uint8)
        save_path =  os.path.join(config.sr_dir, f"{filename}_residual.png")
        cv2.imwrite(save_path, vis_residual)

        
        ## SSIM Calculation
        ssim_val = compare_ssim(sr_y_image, hr_y_image, data_range=1.0)
        total_ssim += ssim_val
        ssim_val_rgb = compare_ssim(sr_image, hr_image, data_range=1.0, channel_axis = -1)
        total_ssim_rgb += ssim_val_rgb

        ## RGB SSIM Calculation
        mse = np.mean((sr_image - hr_image)**2)
        if mse == 0:
            return float("inf")
        psnr_rgb = 10 * np.log10(1.0 / mse)
        total_psnr_rgb += psnr_rgb
        

        save_path = os.path.join(config.sr_dir, "comparisions", f"compare_{index+1}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        save_comparision_images(lr_image, sr_image, hr_image, psnr_val_grayscale, ssim_val, save_path)
    

    print(f"PSNR(Grayscale): {total_psnr_grayscale / total_files:4.2f}dB.\n")
    print(f"PSNR(RGB): {total_psnr_rgb / total_files:4.2f}dB.\n")
    print(f"Average SSIM(Grayscale): {total_ssim / total_files:4.4f}")
    print(f"Average SSIM(RGB): {total_ssim_rgb / total_files:4.4f}")


if __name__ == "__main__":
    main()
