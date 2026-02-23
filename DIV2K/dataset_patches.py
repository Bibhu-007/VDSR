# dataset_patches.py
import os, cv2, numpy as np
from torch.utils.data import Dataset
import imgproc

class PatchFolderDataset(Dataset):
    """
    Loads pre-extracted HR/LR patch pairs from folders without preloading.
    Returns Y-channel tensors [1,H,W] in [0,1].
    """
    def __init__(self, hr_dir: str, lr_dir: str, mode: str = "Train", augment: bool = True):
        super().__init__()
        self.hr_dir, self.lr_dir = hr_dir, lr_dir
        self.mode = mode
        self.augment = augment and (mode == "Train")

        hr_files = {f for f in os.listdir(hr_dir) if f.lower().endswith((".png",".jpg",".jpeg"))}
        lr_files = {f for f in os.listdir(lr_dir) if f.lower().endswith((".png",".jpg",".jpeg"))}

        # pair by identical names existing in BOTH dirs
        self.files = sorted(hr_files & lr_files)
        if not self.files:
            raise RuntimeError(f"No matching HR/LR files.\nHR={hr_dir}\nLR={lr_dir}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        hr = cv2.imread(os.path.join(self.hr_dir, fname), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        lr = cv2.imread(os.path.join(self.lr_dir, fname), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

        # train VDSR on Y-channel
        hr_y = imgproc.bgr2ycbcr(hr, use_y_channel=True)
        lr_y = imgproc.bgr2ycbcr(lr, use_y_channel=True)

        # light aug (patches already cropped)
        if self.augment:
            if np.random.rand() < 0.5:
                hr_y = np.ascontiguousarray(np.flip(hr_y, 1)); lr_y = np.ascontiguousarray(np.flip(lr_y, 1))
            if np.random.rand() < 0.5:
                hr_y = np.ascontiguousarray(np.flip(hr_y, 0)); lr_y = np.ascontiguousarray(np.flip(lr_y, 0))
            k = np.random.randint(0, 4)
            if k:
                hr_y = np.ascontiguousarray(np.rot90(hr_y, k))
                lr_y = np.ascontiguousarray(np.rot90(lr_y, k))

        lr_t = imgproc.image2tensor(lr_y, range_norm=False, half=False)
        hr_t = imgproc.image2tensor(hr_y, range_norm=False, half=False)
        return {"lr": lr_t, "hr": hr_t}
