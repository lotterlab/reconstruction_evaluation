import os
import yaml
import pandas as pd
from torch.utils.data import DataLoader
from data.ucsf_dataset import UcsfDataset
import time
from tqdm import tqdm
import torch
import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
    mean_squared_error,
)
import lpips
from torchvision.utils import save_image


def dice_coefficient(y_true, y_pred):
    """
    Compute the Dice coefficient between two binary images.

    Args:
        y_true: Ground truth binary image
        y_pred: Predicted binary image

    Returns:
        float: Dice coefficient
    """
    intersection = np.sum(y_true * y_pred)
    return (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred))


def predict_ucsf(
    dataloader, task_model, reconstruction_model, number_of_samples, device, logger
):
    predictions = []

    # Initialize LPIPS
    loss_fn_alex = lpips.LPIPS(net="alex").to(device)

    def preprocess_for_lpips(img):
        # Normalize to [-1, 1]
        img = (img * 2) - 1
        # Repeat grayscale channel 3 times for RGB
        img = img.repeat(1, 3, 1, 1)
        return img

    total_batches = len(dataloader)
    logger.info(f"Starting processing of {total_batches} batches")

    start_time = time.time()
    progress_bar = tqdm(dataloader, total=total_batches, desc="Processing images")
    index = 0

    for batch_idx, batch in enumerate(progress_bar):
        if number_of_samples is not None and index >= number_of_samples:
            break

        with torch.no_grad():

            batch_size = batch["image_A"].shape[0]
            index += batch_size

            batch_results = []
            metadata = batch["metadata"]
            for i in range(batch_size):
                result = {
                    "path": metadata["path"][i],
                    "patient_id": metadata["patient_id"][i],
                    "sex": metadata["sex"][i],
                    "age": metadata["age"][i],
                    "cns": metadata["cns"][i],
                    "diagnosis": metadata["diagnosis"][i],
                }
                batch_results.append(result)

            x = batch["image_A"]
            x = x.to(device)

            y_recon = batch["image_B"]
            y_recon = y_recon.to(device)

            for model_type, model_name, model in task_model:
                if model_type == "classifier":
                    pred = model(y_recon)
                    pred = pred.squeeze(0)
                    pred = torch.sigmoid(pred)

                    for i in range(batch_size):
                        batch_results[i][model_name] = float(pred[i])
                elif model_type == "segmentation":
                    pred = model(y_recon)
                    pred = pred.squeeze(1)
                    pred = (pred > 0.5).float()
                    pred = pred.detach().cpu().numpy()

                    for i in range(batch_size):
                        # Calculate metrics per image
                        dice_score = dice_coefficient(
                            pred[i], batch["segmentation"][i].cpu().numpy()
                        )
                        sum_val = np.sum(pred[i])

                        batch_results[i][f"{model_name}_dice"] = float(dice_score)
                        batch_results[i][f"{model_name}_sum"] = float(sum_val)

            recon = reconstruction_model(batch)

            for model_type, model_name, model in task_model:
                if model_type == "classifier":
                    pred = model(recon)
                    pred = pred.squeeze(0)
                    pred = torch.sigmoid(pred)

                    for i in range(batch_size):
                        batch_results[i][f"{model_name}_recon"] = float(pred[i])
                elif model_type == "segmentation":
                    pred = model(recon)
                    pred = pred.squeeze(1)
                    pred = (pred > 0.5).float()
                    pred = pred.detach().cpu().numpy()

                    for i in range(batch_size):
                        # Calculate metrics per image
                        dice_score = dice_coefficient(
                            pred[i], batch["segmentation"][i].cpu().numpy()
                        )
                        sum_val = np.sum(pred[i])

                        batch_results[i][f"{model_name}_recon_dice"] = float(dice_score)
                        batch_results[i][f"{model_name}_recon_sum"] = float(sum_val)

            for i in range(batch_size):

                batch_results[i]["psnr"] = peak_signal_noise_ratio(
                    y_recon[i].cpu().detach().numpy().squeeze(),
                    recon[i].cpu().detach().numpy().squeeze(),
                    data_range=1,
                )
                # print(batch_results[i]["psnr"])
                # print(f"min: {np.min(y_recon[i].cpu().detach().numpy().squeeze())}")
                # print(f"max: {np.max(y_recon[i].cpu().detach().numpy().squeeze())}")
                # print(f"mean: {np.mean(y_recon[i].cpu().detach().numpy().squeeze())}")
                # print(f"min: {np.min(recon[i].cpu().detach().numpy().squeeze())}")
                # print(f"max: {np.max(recon[i].cpu().detach().numpy().squeeze())}")
                # print(f"mean: {np.mean(recon[i].cpu().detach().numpy().squeeze())}")
                # exit()
                batch_results[i]["ssim"] = structural_similarity(
                    y_recon[i].cpu().detach().numpy().squeeze(),
                    recon[i].cpu().detach().numpy().squeeze(),
                    data_range=1,
                )
                batch_results[i]["nrmse"] = mean_squared_error(
                    y_recon[i].cpu().detach().numpy().squeeze(),
                    recon[i].cpu().detach().numpy().squeeze(),
                )

                # LPIPS
                y_recon_lpips = preprocess_for_lpips(y_recon[i].to(device))
                recon_lpips = preprocess_for_lpips(recon[i].to(device))
                batch_results[i]["lpips"] = loss_fn_alex(
                    y_recon_lpips, recon_lpips
                ).item()

            predictions += batch_results

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                progress_bar.set_postfix(
                    {
                        "batches/s": f"{batch_idx/elapsed:.2f}",
                        "processed": batch_idx + 1,
                    }
                )

    total_time = time.time() - start_time
    logger.info(
        f"Processing complete! Total time: {total_time:.2f}s, "
        f"Average time per batch: {total_time/len(dataloader):.3f}s"
    )
    return predictions


def evaluate_ucsf(
    config: dict, task_model, reconstruction_model, results_path: str, device, logger
):

    if "number_of_samples" in config["datasets"]:
        number_of_samples = config["datasets"]["number_of_samples"]
    else:
        number_of_samples = None

    os.makedirs(results_path, exist_ok=True)

    # Save config to output directory for reproducibility
    with open(os.path.join(results_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    dataset = UcsfDataset(config["datasets"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
    )

    results = predict_ucsf(
        dataloader=dataloader,
        task_model=task_model,
        reconstruction_model=reconstruction_model,
        number_of_samples=number_of_samples,
        device=device,
        logger=logger,
    )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        os.path.join(results_path, f"{config['name']}_results.csv"), index=False
    )
