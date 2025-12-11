import os
import pathlib
from typing import Optional
from skimage.transform import radon, iradon, resize
from skimage.io import imread
import numpy as np
import polars as pl
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset


def min_max_slice_normalization(scan: torch.Tensor) -> torch.Tensor:
    scan_min = scan.min()
    scan_max = scan.max()
    if scan_max == scan_min:
        return scan
    normalized_scan = (scan - scan_min) / (scan_max - scan_min)
    return normalized_scan


class ChexDataset(BaseDataset):
    """Dataset for X-ray reconstruction using CycleGAN."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--photon_count",
            type=float,
            default=1e5,
            help="photon count for Poisson noise",
        )
        parser.add_argument(
            "--number_of_samples",
            type=int,
            default=None,
            help="number of samples to use (0 for all)",
        )
        parser.add_argument("--seed", type=int, default=31415, help="random seed")
        parser.add_argument(
            "--csv_path_A",
            type=str,
            required=True,
            help="path to the metadata CSV file",
        )
        parser.add_argument(
            "--split", type=str, default="train_recon", help="train_recon or val_recon"
        )
        parser.add_argument(
            "--csv_path_B",
            type=str,
            required=True,
            help="path to the metadata CSV file",
        )
        parser.add_argument(
            "--dataroot_A", type=str, required=True, help="path to the data root"
        )
        parser.add_argument(
            "--dataroot_B", type=str, required=True, help="path to the data root"
        )
        return parser

    def __init__(self, opt, train=True):
        BaseDataset.__init__(self, opt)
        self.data_root_A = pathlib.Path(opt.dataroot_A)
        self.data_root_B = pathlib.Path(opt.dataroot_B)
        self.csv_path_A = pathlib.Path(opt.csv_path_A)
        self.csv_path_B = pathlib.Path(opt.csv_path_B)
        self.number_of_samples = (
            opt.number_of_samples if hasattr(opt, "number_of_samples") else None
        )
        self.seed = opt.seed if hasattr(opt, "seed") else 31415
        self.train = train

        # Load metadata
        self.metadata_A, self.metadata_B = self._load_metadata()

    def _load_metadata(self):
        """Load metadata for the dataset from CSV file."""
        if not self.csv_path_A.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path_A}")
        if not self.csv_path_B.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path_B}")

        df_A = pl.read_csv(self.csv_path_A)
        df_B = pl.read_csv(self.csv_path_B)

        if self.train:
            df_A = df_A.filter(pl.col("split") == "train_recon")
            df_B = df_B.filter(pl.col("split") == "train_recon")
        else:
            df_A = df_A.filter(pl.col("split") == "val_recon")
            df_B = df_B.filter(pl.col("split") == "val_recon")

        if self.number_of_samples is not None and self.number_of_samples > 0:
            df_A = df_A.sample(n=self.number_of_samples, seed=self.seed)
            df_B = df_B.sample(n=self.number_of_samples, seed=self.seed)

        return df_A, df_B

    def __getitem__(self, index):
        row_A = self.metadata_A.row(index, named=True)
        row_B = self.metadata_B.row(index, named=True)

        # Load the original image
        image_path_A = os.path.join(self.data_root_A, row_A["Path"])
        image_A = imread(image_path_A, as_gray=True).astype(
            np.float32
        )  # Convert to float32
        image_A = min_max_slice_normalization(image_A)
        image_A = torch.from_numpy(image_A).float().unsqueeze(0)
        image_path_B = os.path.join(self.data_root_B, row_B["Path"])
        image_B = imread(image_path_B, as_gray=True).astype(
            np.float32
        )  # Convert to float32
        image_B = min_max_slice_normalization(image_B)
        image_B = resize(image_B, (256, 256))
        image_B = torch.from_numpy(image_B).float().unsqueeze(0)

        return {
            "A": image_A,  # degraded image
            "B": image_B,  # original image
            "A_paths": image_path_A,
            "B_paths": image_path_B,
        }

    def __len__(self):
        return len(self.metadata_A)
