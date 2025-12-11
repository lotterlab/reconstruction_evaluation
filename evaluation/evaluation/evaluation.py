import argparse
import yaml
from datetime import datetime
from pathlib import Path
import logging
from evaluate_ucsf import evaluate_ucsf
from evaluate_chex import evaluate_chex
from evaluate_combined import evaluate_combined
import os


def setup_logger(name, save_dir, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()  # Console handler
    f_handler = logging.FileHandler(os.path.join(save_dir, filename))

    # Create formatters and add it to handlers
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format)
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger  # Make sure to return the logger!


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt", type=str, required=True, help="Path to options YAML file."
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.opt, "r") as f:
        config = yaml.safe_load(f)

    print(config)

    # Setup results directory
    # add timestamp to results_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config["results_path"]) / f"{config['name']}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("base", results_dir)
    logger.info(f"Config:\n{yaml.dump(config, default_flow_style=False)}")

    if config["mode"] == "ucsf":
        logger.info("Evaluating UCSF dataset")
        evaluate_ucsf(config, results_dir, config["name"])
    elif config["mode"] == "chex":
        logger.info("Evaluating CHEX dataset")
        evaluate_chex(config, results_dir, config["name"])
    elif config["mode"] == "combined":
        logger.info("Evaluating combined dataset")
        evaluate_combined(config, results_dir, config["name"])
    else:
        raise ValueError(f'Dataset {config["mode"]} not supported')


if __name__ == "__main__":
    main()
