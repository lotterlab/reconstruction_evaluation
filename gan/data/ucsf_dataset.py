import os
import pathlib
from typing import Optional
from skimage.transform import resize

import fastmri
import nibabel as nib
import numpy as np
import polars as pl
import torch
from fastmri import fft2c, ifft2c
from fastmri.data.subsample import RandomMaskFunc
from data.base_dataset import BaseDataset, get_transform
import torchvision.transforms as transforms


def min_max_slice_normalization(scan: torch.Tensor) -> torch.Tensor:
    scan_min = scan.min()
    scan_max = scan.max()
    if scan_max == scan_min:
        return scan
    normalized_scan = (scan - scan_min) / (scan_max - scan_min)
    return normalized_scan


class UcsfDataset(BaseDataset):
    """Dataset for MRI reconstruction using CycleGAN.
    Domain A: Undersampled MRI images
    Domain B: Fully sampled MRI images
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options and rewrite default values for existing options."""
        parser.add_argument(
            "--sampling_mask",
            type=str,
            default="radial",
            help="type of k-space sampling mask (radial or linear)",
        )
        parser.add_argument(
            "--number_of_samples",
            type=int,
            default=None,
            help="number of samples to use (0 for all)",
        )
        parser.add_argument("--seed", type=int, default=31415, help="random seed")
        parser.add_argument(
            "--type", type=str, default="FLAIR", help="type of MRI scan"
        )
        parser.add_argument(
            "--pathology", type=str, nargs="+", help="list of pathologies to include"
        )
        parser.add_argument(
            "--lower_slice", type=int, default=60, help="lower slice index bound"
        )
        parser.add_argument(
            "--upper_slice", type=int, default=130, help="upper slice index bound"
        )
        parser.add_argument(
            "--age_bins",
            type=float,
            nargs="+",
            default=[0, 68, 100],
            help="age bins for stratification",
        )
        return parser

    def __init__(self, opt, train=True):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags
        """
        BaseDataset.__init__(self, opt)
        self.sampling_mask = (
            opt.sampling_mask if hasattr(opt, "sampling_mask") else "radial"
        )

        # Convert dataroot to pathlib.Path
        self.data_root = pathlib.Path(opt.dataroot)

        # Set up dataset parameters from options
        self.number_of_samples = (
            opt.number_of_samples if hasattr(opt, "number_of_samples") else None
        )
        self.seed = opt.seed if hasattr(opt, "seed") else 31415
        self.split = opt.phase  # use CycleGAN's phase (train/test) as split
        self.type = opt.type if hasattr(opt, "type") else "T2"
        self.pathology = opt.pathology if hasattr(opt, "pathology") else None
        self.lower_slice = opt.lower_slice if hasattr(opt, "lower_slice") else None
        self.upper_slice = opt.upper_slice if hasattr(opt, "upper_slice") else None
        self.train = train

        # Load metadata
        self.metadata = self._load_metadata()

        # Set up transforms
        self.transform = [
            min_max_slice_normalization,
            lambda x: transforms.functional.resize(x.unsqueeze(0), (256, 256)).squeeze(
                0
            ),
        ]
        self.transform = transforms.Compose(self.transform)

    def _load_metadata(self):
        """Load metadata for the dataset from CSV file."""
        # Load from CSV instead of parquet
        metadata_file = self.data_root / "metadata.csv"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        df = pl.read_csv(metadata_file)

        # Apply filters based on parameters
        if self.type:
            df = df.filter(pl.col("type") == self.type)
        if self.pathology:
            df = df.filter(pl.col("pathology").is_in(self.pathology))
        if self.lower_slice is not None:
            df = df.filter(pl.col("slice_id") >= self.lower_slice)
        if self.upper_slice is not None:
            df = df.filter(pl.col("slice_id") <= self.upper_slice)

        if self.train:
            df = df.filter(pl.col("split") == "train")
        else:
            df = df.filter(pl.col("split") == "val")

        # Sample if number_of_samples is specified
        if self.number_of_samples is not None and self.number_of_samples > 0:
            df = df.sample(n=self.number_of_samples, seed=self.seed)

        return df

    def convert_to_complex(self, image_slice):
        """Convert a real-valued 2D image slice to complex format."""
        complex_tensor = torch.stack(
            (image_slice, torch.zeros_like(image_slice)), dim=-1
        )
        return complex_tensor

    def create_radial_mask(self, shape, num_rays=140):
        """Create a radial mask for undersampling k-space."""
        H, W = shape
        center = (H // 2, W // 2)
        Y, X = np.ogrid[:H, :W]
        mask = np.zeros((H, W), dtype=np.float32)
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

        for angle in angles:
            line_x = np.cos(angle)
            line_y = np.sin(angle)
            for r in range(max(H, W) // 2):
                x = int(center[1] + r * line_x)
                y = int(center[0] + r * line_y)
                if 0 <= x < W and 0 <= y < H:
                    mask[y, x] = 1
        return mask

    def apply_radial_mask_to_kspace(self, kspace):
        """Apply a radial mask to the k-space data."""
        H, W, _ = kspace.shape
        radial_mask = self.create_radial_mask((H, W))
        radial_mask = torch.from_numpy(radial_mask).to(kspace.device).unsqueeze(-1)
        return kspace * radial_mask

    def apply_linear_mask_to_kspace(self, kspace):
        """Apply a linear mask to the k-space data."""
        mask_func = RandomMaskFunc(center_fractions=[0.1], accelerations=[6])
        mask = mask_func(kspace.shape, seed=None)[0]
        mask = mask.to(kspace.device).unsqueeze(-1)
        return kspace * mask

    def undersample_slice(self, slice_tensor):
        """Undersample an MRI slice using specified mask."""
        # Convert real slice to complex-valued tensor
        complex_slice = self.convert_to_complex(slice_tensor)

        # Transform to k-space
        kspace = fft2c(complex_slice)

        # Apply mask
        if self.sampling_mask == "radial":
            undersampled_kspace = self.apply_radial_mask_to_kspace(kspace)
        elif self.sampling_mask == "linear":
            undersampled_kspace = self.apply_linear_mask_to_kspace(kspace)
        else:
            raise ValueError(f"Unsupported sampling mask: {self.sampling_mask}")

        # Inverse transform
        undersampled_image = ifft2c(undersampled_kspace)
        return fastmri.complex_abs(undersampled_image)

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        row = self.metadata.row(index, named=True)

        # Load the original image using your original loading logic
        nifti_img = nib.load(str(self.data_root / row["file_path"]))
        scan = nifti_img.get_fdata()
        slice_tensor = torch.from_numpy(scan[:, :, row["slice_id"]]).float()

        if self.transform:
            slice_tensor = self.transform(slice_tensor)

        # Create undersampled version
        undersampled_tensor = self.undersample_slice(slice_tensor)

        # Prepare for CycleGAN (both need to be 3D tensors: C×H×W)
        slice_tensor = slice_tensor.unsqueeze(0)
        undersampled_tensor = undersampled_tensor.unsqueeze(0)

        # Return in CycleGAN format but using your file paths
        return {
            "A": undersampled_tensor,  # undersampled image
            "B": slice_tensor,  # fully sampled image
            "A_paths": str(row["file_path"]),
            "B_paths": str(row["file_path"]),
        }

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.metadata)

    def get_patient_data(self, patient_id):
        """Get all slices for a specific patient."""
        patient_slices_metadata = self.metadata.filter(
            pl.col("patient_id") == patient_id
        )
        patient_slices_metadata = patient_slices_metadata.sort("slice_id")

        if len(patient_slices_metadata) == 0:
            print(f"No slices found for patient_id={patient_id}")
            return []

        slices = []
        for row_idx in range(len(patient_slices_metadata)):
            row = patient_slices_metadata.row(row_idx, named=True)

            # Load and process the slice
            nifti_img = nib.load(str(self.data_root / row["file_path"]))
            scan = nifti_img.get_fdata()
            slice_tensor = torch.from_numpy(scan[:, :, row["slice_id"]]).float()

            if self.transform:
                slice_tensor = self.transform(slice_tensor)

            undersampled_tensor = self.undersample_slice(slice_tensor)

            slice_tensor = slice_tensor.unsqueeze(0)
            undersampled_tensor = undersampled_tensor.unsqueeze(0)

            slices.append((undersampled_tensor, slice_tensor))

        return slices
