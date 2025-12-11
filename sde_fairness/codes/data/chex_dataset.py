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
import pandas as pd


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
        self.metadata_LQ, self.metadata_GT = self._load_metadata()
        self.labels = self._process_labels(self.metadata_LQ)

    def _load_metadata(self):
        """Load metadata for the dataset from CSV file."""
        if not self.csv_path_LQ.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path_LQ}")
        if not self.csv_path_GT.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path_GT}")

        df_LQ = pd.read_csv(self.csv_path_LQ)
        df_GT = pd.read_csv(self.csv_path_GT)

        # Filter for validation split first
        if self.train:
            df_LQ = df_LQ[df_LQ["split"] == "val_recon"]
            df_GT = df_GT[df_GT["split"] == "val_recon"]
        else:
            df_LQ = df_LQ[df_LQ["split"] == "val_class"]
            df_GT = df_GT[df_GT["split"] == "val_class"]

        # Get total number of validation samples
        if self.number_of_samples is not None and self.number_of_samples > 0:
            df_LQ = df_LQ.sample(n=self.number_of_samples, random_state=self.seed)
            df_GT = df_GT.sample(n=self.number_of_samples, random_state=self.seed)

        return df_LQ, df_GT

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

    def __getitem__(self, index):
        row_LQ = self.metadata_LQ.iloc[index]
        row_GT = self.metadata_GT.iloc[index]

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

        sex = float(0 if row_LQ["Sex"] == "F" else 1)  # Assuming binary F/M encoding
        age = float(row_LQ["Age"] <= 61)  # Already boolean, convert to float

        # Map race to numeric values
        race_mapping = {
            "Other": 0,
            "White": 1,
            "Black": 2,
            "Native Hawaiian or Other Pacific Islander": 3,
            "Asian": 4,
            "American Indian or Alaska Native": 5,
        }
        race = float(
            race_mapping.get(row_LQ["Mapped_Race"], 0)
        )  # Default to 'Other' if not found

        # Add protected attributes to the tensor
        protected_attrs = torch.tensor([sex, age, race])

        labels = self.labels[index]

        return {
            "LQ": image_LQ,  # degraded image
            "GT": image_GT,  # original image
            "LQ_paths": image_path_LQ,
            "GT_paths": image_path_GT,
            "labels": labels,  # labels
            "protected_attrs": protected_attrs,
        }

    def __len__(self):
        return len(self.metadata_LQ)

    def compute_sample_weights(dataset):
        # Get the metadata DataFrame; assuming self.metadata_A is used
        df = dataset.metadata_LQ.copy()

        # Use the sensitive columns directly, e.g., 'Sex', 'Age', 'Mapped_Race'
        # Adjust the column names as necessary.
        sensitive_cols = ["Sex", "Age", "Mapped_Race"]

        # Group by sensitive attributes and count frequency
        group_counts = df.groupby(sensitive_cols).size().reset_index(name="count")

        # Merge the count back to the DataFrame on the sensitive columns
        df = df.merge(group_counts, on=sensitive_cols, how="left")

        # Create a weights array: inverse of the count
        weights = 1.0 / df["count"]

        # Convert to list (or a numpy array) so it matches the order of samples in the dataset
        return weights.tolist()
