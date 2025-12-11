import os
import pathlib
from typing import Optional
from skimage.io import imread
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import polars as pl  # for metadata handling
import nibabel as nib  # for NIFTI file handling
import fastmri  # for k-space operations


def min_max_slice_normalization(scan: torch.Tensor) -> torch.Tensor:
    scan_min = scan.min()
    scan_max = scan.max()
    if scan_max == scan_min:
        return scan
    normalized_scan = (scan - scan_min) / (scan_max - scan_min)
    return normalized_scan


class ChexDataset(Dataset):
    """Dataset for X-ray reconstruction."""

    def __init__(self, opt, train=True):
        """Initialize this dataset class.

        Parameters:
            opt (dict) -- stores all the experiment flags
            train (bool) -- whether we are in training mode
        """
        self.data_root_A = pathlib.Path(opt["dataroot_A"])
        self.data_root_B = pathlib.Path(opt["dataroot_B"])
        self.csv_path_A = pathlib.Path(opt["csv_path_A"])
        self.csv_path_B = pathlib.Path(opt["csv_path_B"])
        self.number_of_samples = opt.get("number_of_samples", None)
        self.seed = opt.get("seed", 31415)
        self.split = "test"

        # Load metadata
        self.metadata_A, self.metadata_B = self._load_metadata()

        # Set up transforms
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256), antialias=True),
                min_max_slice_normalization,
            ]
        )

    def _load_metadata(self):
        """Load metadata for the dataset from CSV files."""
        if not self.csv_path_A.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path_A}")
        if not self.csv_path_B.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path_B}")

        df_A = pl.read_csv(self.csv_path_A)
        df_B = pl.read_csv(self.csv_path_B)

        # Filter by split
        df_A = df_A.filter(pl.col("split") == self.split)
        df_B = df_B.filter(pl.col("split") == self.split)

        # Sample if number_of_samples is specified
        if self.number_of_samples is not None and self.number_of_samples > 0:
            df_A = df_A.sample(n=self.number_of_samples, seed=self.seed)
            df_B = df_B.sample(n=self.number_of_samples, seed=self.seed)

        return df_A, df_B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int) -- a random integer for data indexing

        Returns:
            dict -- returns a dictionary containing the degraded and original images and their paths
        """
        row_A = self.metadata_A.row(index, named=True)
        row_B = self.metadata_B.row(index, named=True)

        # Load the original image
        image_path_A = os.path.join(self.data_root_A, row_A["Path"])
        image_path_B = os.path.join(self.data_root_B, row_B["Path"])

        image_A = imread(image_path_A, as_gray=True).astype(np.float32)
        image_B = imread(image_path_B, as_gray=True).astype(np.float32)

        # Apply transforms
        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        image_A = image_A.squeeze(0)
        image_B = image_B.squeeze(0)

        return {
            "A": image_A,  # degraded image
            "B": image_B,  # original image
            "A_paths": row_A["Path"],
            "B_paths": row_B["Path"],
        }

    def __len__(self):
        """Return the total number of images."""
        return len(self.metadata_A)
