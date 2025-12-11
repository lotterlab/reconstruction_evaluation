import argparse
import os
import time
from datetime import datetime
import logging
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
from skimage.io import imsave
from pathlib import Path
import sys
import importlib.util
import polars as pl
from tqdm import tqdm

# Add necessary paths
current_dir = Path(__file__).resolve().parent
project_root = (
    current_dir.parent.parent.parent
)  # This should get us to the root containing all repos
sys.path.append(str(project_root))
sys.path.append(str(current_dir.parent))  # For local imports like data/

from data.chex_dataset import ChexDataset
from data.ucsf_dataset import UcsfDataset


def import_module_from_path(module_name, module_path):
    """Import a module from a path dynamically."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def setup_logger(name, save_dir, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(os.path.join(save_dir, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_reconstruction_model(config, device):
    model_config = config["models"]
    if model_config["network"] == "UNet":
        # Import from recon_bias repository's src directory
        sys.path.append(str(Path(config["paths"]["recon_bias"]) / "src"))
        from model.reconstruction.reconstruction_model import ReconstructionModel
        from model.reconstruction.unet import UNet

        model = ReconstructionModel()
        network = UNet()
        model.set_network(network)
        model.load_state_dict(
            torch.load(model_config["model_path"], map_location=device)
        )
        model.network.eval()
        return model.to(device)

    elif model_config["network"] == "GAN":
        # Import from pix2pix models directory
        pix2pix_path = Path(config["paths"]["pix2pix"])
        sys.path.append(str(pix2pix_path))
        from models.networks import UnetGenerator

        model = UnetGenerator(input_nc=1, output_nc=1, num_downs=7)
        model.load_state_dict(
            torch.load(model_config["model_path"], map_location=device)
        )
        model.eval()
        return model.to(device)

    elif model_config["network"] == "Diffusion":
        # Import from image-restoration-sde repository
        config["is_train"] = False
        config["train"] = None
        try:
            sde_path = Path(config["paths"]["sde"])
            if not sde_path.exists():
                raise ValueError(f"SDE path does not exist: {sde_path}")

            # Import required modules dynamically from codes directory
            codes_path = sde_path / "codes"
            sys.path.append(str(codes_path))

            models_module = import_module_from_path(
                "models", str(codes_path / "config/deblurring/models/__init__.py")
            )

            util = import_module_from_path(
                "util", str(codes_path / "utils/__init__.py")
            )

        except Exception as e:
            raise ImportError(f"Failed to import diffusion modules. Error: {e}")

        # Create model
        model = models_module.create_model(config)

        # Setup SDE
        sde = util.IRSDE(
            max_sigma=config["sde"]["max_sigma"],
            T=config["sde"]["T"],
            schedule=config["sde"]["schedule"],
            eps=config["sde"]["eps"],
            device=device,
        )
        sde.set_model(model.model)

        return (model, sde)

    else:
        raise ValueError(f"Unknown network type: {model_config['network']}")


def process_batch(batch, model, model_type, save_dir, device, dataset_type):
    # Process batch with model
    input_imgs = batch["A"].to(device)
    gt_imgs = batch["B"].to(device)
    with torch.no_grad():
        if model_type == "Diffusion":
            model_obj, sde = model
            noisy_state = sde.noise_state(input_imgs)
            model_obj.feed_data(noisy_state, input_imgs, gt_imgs)
            model_obj.test(sde)
            outputs = model_obj.get_current_test_visuals()["Output"]
        else:  # UNet or GAN
            outputs = model(input_imgs)

    metadata = []

    # Save outputs based on dataset type
    if dataset_type == "ucsf":
        # For UCSF, use patient_id and slice_id to create filenames
        patient_ids = batch["patient_id"]
        slice_ids = batch["slice_id"]

        for patient_id, slice_id, output in zip(patient_ids, slice_ids, outputs):
            # Save as .npy to preserve raw float values
            output_filename = f"{patient_id}_{slice_id:03d}.npy"
            output_path = save_dir / output_filename

            output_np = output.cpu().numpy().squeeze()
            np.save(str(output_path), output_np)

            metadata.append(
                {
                    "patient_id": patient_id,
                    "slice_id": slice_id,
                    "path": output_filename,
                }
            )

        # Append to existing metadata if file exists, otherwise create new
        metadata_file = save_dir / "metadata.csv"
        new_df = pl.DataFrame(metadata)
        if metadata_file.exists():
            existing_df = pl.read_csv(metadata_file)
            combined_df = pl.concat([existing_df, new_df])
            combined_df.write_csv(metadata_file)
        else:
            new_df.write_csv(metadata_file)

    else:  # chex or other datasets
        paths = batch["A_paths"]
        for path, output in zip(paths, outputs):
            output_np = output.cpu().numpy().squeeze()

            # Change extension to .npy
            output_path = os.path.join(save_dir, os.path.splitext(path)[0] + ".npy")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, output_np)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt", type=str, required=True, help="Path to options YAML file."
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.opt, "r") as f:
        config = yaml.safe_load(f)

    # Add repository paths from config
    if "paths" in config:
        for repo_path in config["paths"].values():
            sys.path.append(str(Path(repo_path).resolve()))

    # Setup results directory
    results_dir = Path(config["results_path"]) / f"{config['name']}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger("base", results_dir)
    logger.info(f"Config:\n{yaml.dump(config, default_flow_style=False)}")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Find enabled dataset and model
    dataset_name = config["datasets"]["name"]
    model_name = config["models"]["network"]

    # Create dataset
    if dataset_name == "chex":
        dataset = ChexDataset(config["datasets"], train=False)
    elif dataset_name == "ucsf":
        dataset = UcsfDataset(config["datasets"], train=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # Load model
    logger.info(f"Loading {model_name} model...")
    model = load_reconstruction_model(config, device)

    # Process dataset
    total_batches = len(dataloader)
    logger.info(
        f"Starting processing of {len(dataset)} images in {total_batches} batches"
    )

    start_time = time.time()
    progress_bar = tqdm(dataloader, total=total_batches, desc="Processing images")
    for batch_idx, batch in enumerate(progress_bar):

        batch["A"] = batch["A"].unsqueeze(1)
        process_batch(batch, model, model_name, results_dir, device, dataset_name)

        # Update progress bar description with current stats
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            images_processed = (batch_idx + 1) * config["batch_size"]
            avg_time_per_image = elapsed / images_processed
            progress_bar.set_postfix(
                {"img/s": f"{1/avg_time_per_image:.2f}", "processed": images_processed}
            )

    total_time = time.time() - start_time
    logger.info(
        f"Processing complete! Total time: {total_time:.2f}s, "
        f"Average time per image: {total_time/len(dataset):.3f}s"
    )


if __name__ == "__main__":
    main()
