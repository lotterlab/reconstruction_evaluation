#!/usr/bin/env python3
import os
import yaml
import subprocess
import tempfile
import argparse
from typing import List, Dict, Any


def create_temp_config(template_path: str, model_path: str, lambda_value: float) -> str:
    """
    Create a temporary config file by modifying the template with the given model path and lambda value.

    Args:
        template_path: Path to the template YAML file
        model_path: Path to the model file
        lambda_value: Lambda value to use

    Returns:
        Path to the temporary config file
    """
    # Read the template file
    with open(template_path, "r") as f:
        config = yaml.safe_load(f)

    lambda_str = str(lambda_value).replace(".", "")

    # Update the config
    config["name"] = config["name"].replace("<lambda>", str(lambda_str))
    config["models"]["model_path"] = config["models"]["model_path"].replace(
        "<path_to_model>", model_path
    )

    # Create a temporary file - replace dots with empty string in lambda value for filename
    fd, temp_path = tempfile.mkstemp(
        prefix=f"config_lambda_{lambda_str}", suffix=".yml", dir="options"
    )

    # Write the updated config to the temporary file
    with os.fdopen(fd, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return temp_path


def run_prediction(config_path: str) -> None:
    """
    Run the prediction script with the given config file.

    Args:
        config_path: Path to the config file
    """
    cmd = ["python", "prediction.py", f"--opt={config_path}"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    # Define models and their corresponding lambda values
    model_configs = [
        # (model_path, lambda_value)
        (
            "/lotterlab/users/matteo/code/recon_bias/output/unet-fairness-chex-10000-eodd-lambda-01_20250404_083458/checkpoints/unet-fairness-chex-10000-eodd-lambda-01_20250404_083458_epoch_7_best.pth",
            0.1,
        ),
        (
            "/lotterlab/users/matteo/code/recon_bias/output/unet-fairness-chex-10000-eodd-lambda-0316_20250404_160357/checkpoints/unet-fairness-chex-10000-eodd-lambda-0316_20250404_160357_epoch_11_best.pth",
            0.316,
        ),
        (
            "/lotterlab/users/matteo/code/recon_bias/output/unet-fairness-chex-10000-eodd-lambda-1_20250405_020119/checkpoints/unet-fairness-chex-10000-eodd-lambda-1_20250405_020119_epoch_7_best.pth",
            1,
        ),
        (
            "/lotterlab/users/matteo/code/recon_bias/output/unet-fairness-chex-10000-eodd-lambda-316_20250405_092854/checkpoints/unet-fairness-chex-10000-eodd-lambda-316_20250405_092854_epoch_15_best.pth",
            3.16,
        ),
        (
            "/lotterlab/users/matteo/code/recon_bias/output/unet-fairness-chex-10000-eodd-lambda-10_20250405_215923/checkpoints/unet-fairness-chex-10000-eodd-lambda-10_20250405_215923_epoch_2_best.pth",
            10,
        ),
    ]

    # Template path
    template_path = "options/chex_unet_eodd_lambda.yml"

    # Make sure options directory exists
    os.makedirs("options", exist_ok=True)

    temp_files = []

    try:
        # Process each model with its corresponding lambda
        for model_path, lambda_value in model_configs:
            # Create temporary config
            temp_path = create_temp_config(template_path, model_path, lambda_value)
            temp_files.append(temp_path)

            # Run prediction
            run_prediction(temp_path)

            print(
                f"Completed prediction for model: {model_path}, lambda: {lambda_value}"
            )
    finally:
        # Clean up temporary files
        for temp_path in temp_files:
            try:
                os.remove(temp_path)
                print(f"Removed temporary config: {temp_path}")
            except Exception as e:
                print(f"Failed to remove temporary config {temp_path}: {e}")


if __name__ == "__main__":
    main()
