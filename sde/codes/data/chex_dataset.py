import os
import pathlib
from typing import Optional
from skimage.transform import radon, iradon, resize
from skimage.io import imread
import numpy as np
import polars as pl
import torch
import torchvision.transforms as transforms
import torch.utils.data as data


def min_max_slice_normalization(scan: torch.Tensor) -> torch.Tensor:
    scan_min = scan.min()
    scan_max = scan.max()
    if scan_max == scan_min:
        return scan
    normalized_scan = (scan - scan_min) / (scan_max - scan_min)
    return normalized_scan


class ChexDataset(data.Dataset):
    """Dataset for X-ray reconstruction using CycleGAN."""

    def __init__(self, opt, train=True, test=False):
        super().__init__()
        self.opt = opt
        self.data_root_LQ = pathlib.Path(opt["dataroot_LQ"])
        self.data_root_GT = pathlib.Path(opt["dataroot_GT"])
        self.csv_path_LQ = pathlib.Path(opt["csv_path_LQ"])
        self.csv_path_GT = pathlib.Path(opt["csv_path_GT"])
        self.number_of_samples = (
            opt["number_of_samples"] if "number_of_samples" in opt else None
        )
        self.seed = opt["seed"] if "seed" in opt else 31415
        self.train = train
        self.test = test

        # Load metadata
        self.metadata_LQ, self.metadata_GT = self._load_metadata()

    def _load_metadata(self):
        """Load metadata for the dataset from CSV file."""
        if not self.csv_path_LQ.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path_LQ}")
        if not self.csv_path_GT.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path_GT}")

        df_LQ = pl.read_csv(self.csv_path_LQ)
        df_GT = pl.read_csv(self.csv_path_GT)

        if self.train:
            df_LQ = df_LQ.filter(pl.col("split") == "train_recon")
            df_GT = df_GT.filter(pl.col("split") == "train_recon")
        elif not self.test:
            df_LQ = df_LQ.filter(pl.col("split") == "val_recon")
            df_GT = df_GT.filter(pl.col("split") == "val_recon")
        else:
            df_LQ = df_LQ.filter(pl.col("split") == "test")
            df_GT = df_GT.filter(pl.col("split") == "test")

        if self.number_of_samples is not None and self.number_of_samples > 0:
            df_LQ = df_LQ.sample(n=self.number_of_samples, seed=self.seed)
            df_GT = df_GT.sample(n=self.number_of_samples, seed=self.seed)

        return df_LQ, df_GT

    def __getitem__(self, index):
        row_LQ = self.metadata_LQ.row(index, named=True)
        row_GT = self.metadata_GT.row(index, named=True)

        # Load the original image
        image_path_LQ = os.path.join(self.data_root_LQ, row_LQ["Path"])
        image_LQ = imread(image_path_LQ, as_gray=True).astype(
            np.float32
        )  # Convert to float32
        image_LQ = min_max_slice_normalization(image_LQ)
        image_LQ = torch.from_numpy(image_LQ).float().unsqueeze(0)
        image_path_GT = os.path.join(self.data_root_GT, row_GT["Path"])
        image_GT = imread(image_path_GT, as_gray=True).astype(
            np.float32
        )  # Convert to float32
        image_GT = min_max_slice_normalization(image_GT)
        image_GT = resize(image_GT, (256, 256))
        image_GT = torch.from_numpy(image_GT).float().unsqueeze(0)

        return {
            "LQ": image_LQ,  # degraded image
            "GT": image_GT,  # original image
            "LQ_paths": image_path_LQ,
            "GT_paths": image_path_GT,
        }

    def __len__(self):
        return len(self.metadata_LQ)
