import os
import pathlib
from typing import Optional
from skimage.io import imread
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd  # changed from polars to pandas
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
    """Dataset for X-ray reconstruction with CheXpert labels."""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (dict) -- stores all the experiment flags
        """
        self.data_root_A = pathlib.Path(opt["dataroot_A"])
        self.data_root_B = pathlib.Path(opt["dataroot_B"])
        self.csv_path_A = pathlib.Path(opt["csv_path_A"])
        self.csv_path_B = pathlib.Path(opt["csv_path_B"])
        self.number_of_samples = opt.get("number_of_samples", None)
        self.seed = opt.get("seed", 31415)
        self.split = opt.get("split", "test")

        # Define pathologies (labels)
        self.pathologies = [
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
            "No Finding",
        ]
        self.pathologies = sorted(self.pathologies)

        # Load metadata
        self.metadata_A, self.metadata_B = self._load_metadata()
        self.labels = self._process_labels(self.metadata_A)

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

        df_A = pd.read_csv(self.csv_path_A)
        df_B = pd.read_csv(self.csv_path_B)

        # Filter by split if present
        if "split" in df_A.columns:
            df_A = df_A[df_A["split"] == self.split]
            df_B = df_B[df_B["split"] == self.split]

        # Sample if number_of_samples is specified
        if self.number_of_samples is not None and self.number_of_samples > 0:
            df_A = df_A.sample(n=self.number_of_samples, random_state=self.seed)
            df_B = df_B.sample(n=self.number_of_samples, random_state=self.seed)

        return df_A, df_B

    def _process_labels(self, df):
        # First identify healthy cases
        healthy = df["No Finding"] == 1

        labels = []
        for pathology in self.pathologies:
            assert pathology in df.columns

            if pathology == "No Finding":
                # Handle NaN in No Finding when other pathologies exist
                for idx, row in df.iterrows():
                    if row["No Finding"] != row["No Finding"]:  # check for NaN
                        if (
                            row[self.pathologies] == 1
                        ).sum():  # if any pathology present
                            df.loc[idx, "No Finding"] = 0
            elif pathology != "Support Devices":
                # If healthy, other pathologies (except Support Devices) must be 0
                df.loc[healthy, pathology] = 0

            mask = df[pathology]
            labels.append(mask.values)

        # Convert to numpy array and transpose to get samples x labels
        labels = np.asarray(labels).T
        labels = labels.astype(np.float32)

        # Convert -1 to NaN
        labels[labels == -1] = np.nan

        return torch.from_numpy(labels)

    def _process_metadata(self, row):
        """Extract metadata from a row."""
        metadata = {
            "path": str(row["Path"]) if "Path" in row else "",
            "patient_id": str(row["PatientID"]) if "PatientID" in row else "",
            "age": float(row["Age"]) if "Age" in row else 0.0,
            "sex": str(row["Sex"]) if "Sex" in row else "",
            "view": str(row["view"]) if "view" in row else "",
            "race": str(row["Mapped_Race"]) if "Mapped_Race" in row else "",
        }
        return metadata

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        row_A = self.metadata_A.iloc[index]
        row_B = self.metadata_B.iloc[index]

        # Load images
        image_path_A = os.path.join(self.data_root_A, row_A["Path"])
        image_path_B = os.path.join(self.data_root_B, row_B["Path"])

        # Check if files exist
        if not os.path.exists(image_path_A):
            raise FileNotFoundError(f"Image A not found: {image_path_A}")
        if not os.path.exists(image_path_B):
            raise FileNotFoundError(f"Image B not found: {image_path_B}")

        image_A = imread(image_path_A, as_gray=True).astype(np.float32)
        image_B = imread(image_path_B, as_gray=True).astype(np.float32)

        # Apply transforms
        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        # Process labels and metadata
        labels = self.labels[index]
        metadata = self._process_metadata(row_A)

        return {
            "image_A": image_A,  # degraded image
            "image_B": image_B,  # original image
            "labels": labels,  # pathology labels
            "metadata": metadata,  # patient metadata
            "image_path": row_A["Path"],
        }

    def __len__(self):
        """Return the total number of images."""
        return len(self.metadata_A)
