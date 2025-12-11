import os
import pathlib
from typing import Optional
from skimage.transform import resize
import nibabel as nib
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from fastmri import fft2c, ifft2c
from fastmri.data.subsample import RandomMaskFunc


def min_max_slice_normalization(scan: torch.Tensor) -> torch.Tensor:
    scan_min = scan.min()
    scan_max = scan.max()
    if scan_max == scan_min:
        return scan
    normalized_scan = (scan - scan_min) / (scan_max - scan_min)
    return normalized_scan


class UcsfDataset(Dataset):
    """Dataset for MRI reconstruction."""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (dict) -- stores all the experiment flags
            train (bool) -- whether we are in training mode
        """
        self.data_root = pathlib.Path(opt["dataroot"])
        self.sampling_mask = opt.get("sampling_mask", "radial")
        self.number_of_samples = opt.get("number_of_samples", None)
        self.seed = opt.get("seed", 31415)
        self.type = opt.get("type", "FLAIR")
        self.pathology = opt.get("pathology", [])
        self.lower_slice = opt.get("lower_slice", 60)
        self.upper_slice = opt.get("upper_slice", 130)
        self.split = "test"
        self.num_rays = opt.get("num_rays", 140)
        print(f"num_rays: {self.num_rays}")

        # Load metadata
        self.metadata = self._load_metadata()

        # Set up transforms
        self.transform = transforms.Compose(
            [
                min_max_slice_normalization,
                transforms.Resize((256, 256), antialias=True),
            ]
        )

    def _load_metadata(self):
        """Load metadata for the dataset from CSV file."""
        import polars as pl

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

        # Filter by split
        df = df.filter(pl.col("split") == self.split)

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

    def create_radial_mask(self, shape):
        """Create a radial mask for undersampling k-space."""
        H, W = shape
        center = (H // 2, W // 2)
        Y, X = np.ogrid[:H, :W]
        mask = np.zeros((H, W), dtype=np.float32)
        angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)

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
        return torch.abs(undersampled_image[..., 0])

    sex_map = {"M": 0, "F": 1}
    diagnosis_map = {
        "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted": 0,
        "Astrocytoma, IDH-wildtype": 1,
        "Astrocytoma, IDH-mutant": 2,
        "Glioblastoma, IDH-wildtype": 3,
    }

    def _process_metadata(self, row):
        """Extract metadata from a row."""
        metadata = {
            "path": str(row["file_path"]) if "file_path" in row else "",
            "patient_id": str(row["patient_id"]) if "patient_id" in row else "",
            "slice_id": int(row["slice_id"]) if "slice_id" in row else 0,
            "sex": torch.tensor(self.sex_map[row["sex"]], dtype=torch.int64),
            "age": torch.tensor(row["age_at_mri"], dtype=torch.float64),
            "cns": 0 if row["who_cns_grade"] <= 3 else 1,
            "diagnosis": 0 if self.diagnosis_map[row["final_diagnosis"]] <= 2 else 1,
        }
        return metadata

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        row = self.metadata.row(index, named=True)
        metadata = self._process_metadata(row)

        # Load the original image
        nifti_img = nib.load(str(self.data_root / row["file_path"]))
        scan = nifti_img.get_fdata()
        slice_tensor = torch.from_numpy(scan[:, :, row["slice_id"]]).float()

        # Add channel dimension before applying transforms
        slice_tensor = slice_tensor.unsqueeze(0)  # Shape becomes [1, H, W]

        # Apply transforms
        if self.transform:
            slice_tensor = self.transform(slice_tensor)

        # Remove channel dimension after applying transforms
        slice_tensor = slice_tensor.squeeze(0)  # Shape becomes [H, W]

        # Create undersampled version
        undersampled_tensor = self.undersample_slice(slice_tensor)

        slice_tensor = slice_tensor.unsqueeze(0)
        undersampled_tensor = undersampled_tensor.unsqueeze(0)

        segmentation_path = row["file_path"].replace(self.type, "tumor_segmentation")
        nifti_img = nib.load(str(self.data_root / segmentation_path))
        segmentation = nifti_img.get_fdata()
        segmentation_slice = segmentation[:, :, row["slice_id"]]
        segmentation_slice[segmentation_slice == 2] = 1
        segmentation_slice[segmentation_slice == 4] = 0

        segmentation_slice = torch.from_numpy(segmentation_slice).float().unsqueeze(0)
        segmentation_slice = transforms.functional.resize(
            segmentation_slice, (256, 256)
        )

        return {
            "image_A": undersampled_tensor,  # undersampled image
            "image_B": slice_tensor,  # fully sampled image
            "segmentation": segmentation_slice,
            "image_info": {
                "patient_id": row["patient_id"],
                "slice_id": row["slice_id"],
            },
            "metadata": metadata,
        }

    def __len__(self):
        """Return the total number of images."""
        return len(self.metadata)
