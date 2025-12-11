import argparse
import os
import time
import csv
import torch
import logging
import numpy as np
from skimage.io import imsave
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from datetime import datetime

import options as option
from models import create_model
import utils as util
from data import create_dataloader, create_test_dataset


def print_tensor_stats(tensor, name="Tensor"):
    """Print min, max, mean values of a tensor"""
    print(f"\n{name} Statistics:")
    print(f"Min value: {tensor.min().item():.3f}")
    print(f"Max value: {tensor.max().item():.3f}")
    print(f"Mean value: {tensor.mean().item():.3f}")
    print(f"Shape: {tensor.shape}\n")


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array, handling device and gradient"""
    return tensor.detach().cpu().numpy().squeeze()


def normalize_minmax(img):
    """Normalize image to [0,1] range using min-max normalization"""
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt", type=str, required=True, help="Path to options YAML file."
    )
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    # Create timestamped results directory using opt['name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./results/{opt['name']}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Setup logger
    util.setup_logger(
        "base",
        results_dir,
        f"test_{opt['name']}",
        level=logging.INFO,
        screen=True,
        tofile=True,
    )
    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))

    # Create test dataset and dataloader
    test_set = create_test_dataset(opt["datasets"])
    test_loader = create_dataloader(test_set, opt)
    logger.info(f"Number of test images: {len(test_set)}")

    # Load model
    model = create_model(opt)
    device = model.device

    # Setup SDE
    sde = util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"],
        T=opt["sde"]["T"],
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=device,
    )
    sde.set_model(model.model)

    # Setup CSV file with timestamp
    csv_path = os.path.join(results_dir, f"{opt['name']}_{timestamp}.csv")
    csv_exists = os.path.exists(csv_path)

    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)

    # Write header if file is new
    if not csv_exists:
        csv_writer.writerow(["image_name", "psnr", "ssim", "processing_time"])

    try:
        for i, test_data in enumerate(test_loader):
            # Get batch of image paths
            img_paths = (
                test_data["GT_paths"]
                if test_data.get("GT_paths")
                else test_data["LQ_paths"]
            )
            img_names = [str(path) for path in img_paths]

            # Process batch of images
            LQ, GT = test_data["LQ"], test_data["GT"]
            batch_size = LQ.size(0)  # Get current batch size
            print(f"Batch size: {batch_size}")
            print(f"LQ shape: {LQ.shape}")

            noisy_state = sde.noise_state(LQ)
            model.feed_data(noisy_state, LQ, GT)

            # Time the inference
            start_time = time.time()
            model.test(sde)
            processing_time = time.time() - start_time

            # Get output tensors (should be in range [0, 1])
            visuals = model.get_current_test_visuals()
            print(f"Visuals shape: {visuals['Output'].shape}")

            # Process each image in the batch
            for b in range(batch_size):
                sr_tensor = visuals["Output"][b]  # Get individual image from batch
                gt_tensor = visuals["GT"][b]

                # Convert tensors to numpy arrays (keeping [0,1] range)
                sr_img = tensor_to_numpy(sr_tensor)
                gt_img = tensor_to_numpy(gt_tensor)

                # Normalize reconstructed image to [0,1]
                sr_img_norm = normalize_minmax(sr_img)

                # Calculate PSNR using scikit-image (data_range=1 since images are in [0,1])
                psnr_value = psnr(gt_img, sr_img_norm, data_range=1)
                ssim_value = ssim(gt_img, sr_img_norm, data_range=1)

                # Write to CSV
                csv_writer.writerow(
                    [img_names[b], psnr_value, ssim_value, processing_time / batch_size]
                )  # Divide time by batch size for per-image time
                csv_file.flush()  # Ensure writing to disk

                # logger.info(f'Image: {img_names[b]} | PSNR: {psnr_value:.2f} dB | SSIM: {ssim_value:.2f} | Time: {processing_time/batch_size:.3f}s')

            print(f"\nProcessed batch: {i+1}/{len(test_loader)} ({batch_size} images)")

    finally:
        csv_file.close()
        logger.info(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
