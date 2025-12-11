import argparse
import datetime
import os

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

# Import your dataset, models, and trainer
from src.data.ucsf_reconstruction_dataset import UcsfDataset
from src.data.chex_reconstruction_dataset import ChexDataset
from src.model.reconstruction.reconstruction_model import ReconstructionModel
from src.model.reconstruction.reconstruction_unet import UNet
from src.trainer.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train a reconstruction model.")
    parser.add_argument(
        "--opt",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.opt, "r") as f:
        config = yaml.safe_load(f)

    # Extract parameters from the configuration
    output_dir = config["output_dir"]
    output_name = config["output_name"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    save_interval = config.get("save_interval", 1)
    early_stopping_patience = config.get("early_stopping_patience", None)

    dataset = config["dataset"]

    # Append timestamp to output_name to make it unique
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{output_name}_{timestamp}"
    output_dir = os.path.join(output_dir, output_name)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the configuration back into the output directory for tracking
    config_save_path = os.path.join(output_dir, "config.yaml")
    with open(config_save_path, "w") as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

    if dataset == "ucsf":
        # Datasets and DataLoaders
        train_dataset = UcsfDataset(
            opt=config,
            train=True,
        )
        val_dataset = UcsfDataset(
            opt=config,
            train=False,
        )
    elif dataset == "chex":
        train_dataset = ChexDataset(
            opt=config,
            train=True,
        )
        val_dataset = ChexDataset(
            opt=config,
            train=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = ReconstructionModel()
    model = model.to(device)
    network = UNet()
    network = network.to(device)
    model.set_network(network)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        log_dir=os.path.join(output_dir, "logs"),
        output_dir=output_dir,
        output_name=output_name,
        save_interval=save_interval,
        early_stopping_patience=early_stopping_patience,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
