import argparse
import os
from datetime import datetime
import logging
import yaml
import torch
from pathlib import Path
import sys
from tqdm import tqdm
import skimage
import segmentation_models_pytorch as smp
from predict_chex import evaluate_chexpert
from predict_ucsf import evaluate_ucsf
from unet.reconstruction_model import ReconstructionModel
from unet.unet_chex import UNetChex
from unet.unet_ucsf import UNetUCSF
from gan.GAN import UnetGenerator
import numpy as np
from skimage.io import imread
from data.chex_dataset import min_max_slice_normalization
from classifier.classification_model import TTypeBCEClassifier, TGradeBCEClassifier
from classifier.resnet_classification_network import ResNetClassifierNetwork
from segmentation.segmentation_model import SegmentationModel
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


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

        model = ReconstructionModel()
        if config["datasets"]["name"] == "chex":
            print("Loading UNetChex")
            network = UNetChex()
        elif config["datasets"]["name"] == "ucsf":
            print("Loading UNetUCSF")
            network = UNetUCSF()
        model.set_network(network)
        model.load_state_dict(
            torch.load(
                model_config["model_path"], map_location=device, weights_only=False
            )
        )
        model.network.eval()
        model.to(device)
        return lambda x: model(x["image_A"].to(device))

    elif model_config["network"] == "GAN":

        model = UnetGenerator(input_nc=1, output_nc=1, num_downs=7)
        model.load_state_dict(
            torch.load(
                model_config["model_path"], map_location=device, weights_only=False
            )
        )
        model.eval()
        model.to(device)
        return lambda x: model(x["image_A"].to(device))

    elif model_config["network"] == "Diffusion":

        def get_processed_image(x):
            if config["datasets"]["name"] == "chex":
                # Initialize a list to store processed images
                processed_images = []

                # Process each image in the batch
                for image_path in x["image_path"]:
                    full_path = os.path.join(
                        config["datasets"]["processed_image_root"],
                        image_path.replace(".jpg", ".npy"),
                    )
                    # load and process image
                    image = np.load(full_path)
                    image = min_max_slice_normalization(image)
                    processed_images.append(image)

                # Stack all processed images into a batch and convert to torch tensor
                output = torch.from_numpy(np.stack(processed_images, axis=0)).float()
                output = output.unsqueeze(1)
                output = output.to(device)
                return output
            elif config["datasets"]["name"] == "ucsf":
                processed_images = []
                patient_ids, slice_ids = (
                    x["image_info"]["patient_id"],
                    x["image_info"]["slice_id"],
                )
                for patient_id, slice_id in zip(patient_ids, slice_ids):
                    slice_id = (
                        slice_id.item()
                        if isinstance(slice_id, torch.Tensor)
                        else slice_id
                    )
                    # make slice id 3 digits
                    slice_id = f"{slice_id:03d}"
                    full_path = os.path.join(
                        config["datasets"]["processed_image_root"],
                        f"{patient_id}_{slice_id}.npy",
                    )
                    # load and process image
                    image = np.load(full_path)
                    image = min_max_slice_normalization(image)
                    processed_images.append(image)
                output = torch.from_numpy(np.stack(processed_images, axis=0)).float()
                output = output.unsqueeze(1)
                output = output.to(device)
                return output
            else:
                raise ValueError(f"Dataset {config['datasets']['name']} not supported")

        return lambda x: get_processed_image(x)


def load_task_model(config, device):
    if config["name"] == "chexpert-classifier":
        model = torch.load(config["path"], map_location=device, weights_only=False)
        model.to(device)
        return model
    elif config["name"] == "ucsf":
        task_models = []
        for model in config["models"]:
            model_type = model["type"]
            model_name = model["name"]
            model_path = model["path"]

            if model_type == "classifier":
                if model_name == "TTypeBCEClassifier":
                    model = TTypeBCEClassifier()
                elif model_name == "TGradeBCEClassifier":
                    model = TGradeBCEClassifier()
                model.to(device)
                network = ResNetClassifierNetwork(num_classes=model.num_classes)
                network.to(device)
                model.set_network(network)
                model.load_state_dict(
                    torch.load(model_path, map_location=device, weights_only=False)
                )
                model.network.eval()
                task_models.append((model_type, model_name, model))
            elif model_type == "segmentation":
                model = SegmentationModel()
                model = model.to(device)

                network = smp.Unet(
                    in_channels=1,
                    classes=1,
                    encoder_name="resnet34",
                    encoder_depth=5,
                    encoder_weights=None,
                    decoder_channels=(256, 128, 64, 32, 16),
                    activation="sigmoid",
                )

                network = network.to(device)
                model.set_network(network)
                model.load_state_dict(
                    torch.load(
                        model_path, map_location=torch.device("cpu"), weights_only=False
                    )
                )
                model.network.eval()
                task_models.append((model_type, model_name, model))
        return task_models
    else:
        raise ValueError(f"Task model {config['name']} not found")


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

    # Setup results directory
    # add timestamp to results_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config["results_path"]) / f"{config['name']}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("base", results_dir)
    logger.info(f"Config:\n{yaml.dump(config, default_flow_style=False)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load reconstruction model
    logger.info(f'Loading {config["models"]["network"]} model...')
    reconstruction_model = load_reconstruction_model(config, device)
    task_model = load_task_model(
        config["task_models"], device
    )  # before recoding sys otherwise not working

    if config["datasets"]["name"] == "chex":
        evaluate_chexpert(
            config, task_model, reconstruction_model, results_dir, device, logger
        )
    elif config["datasets"]["name"] == "ucsf":
        evaluate_ucsf(
            config, task_model, reconstruction_model, results_dir, device, logger
        )


if __name__ == "__main__":
    main()
