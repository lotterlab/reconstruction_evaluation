import os
import pandas as pd
import torch
import torchvision.transforms as transforms
import yaml
import sys
from data.chex_dataset import ChexDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import torch
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
    mean_squared_error,
)
import lpips


def predict_chexpert(
    dataloader,
    pathologies,
    classifier,
    reconstruction,
    number_of_samples,
    device,
    logger,
):
    """Process all patients in the metadata file."""
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
                    "race": metadata["race"][i],
                }
                batch_results.append(result)

            x_class = batch["image_A"]
            x_class = x_class.to(device)
            y_class = batch["labels"]
            y_class = torch.tensor(y_class, dtype=torch.float32).squeeze()
            y_recon = batch["image_B"]
            y_recon = y_recon.to(device)
            # predictions on gt
            pred = classifier(y_recon)
            pred = pred.squeeze(0)
            pred = torch.sigmoid(pred)

            for i in range(batch_size):
                for j, pathology in enumerate(pathologies):
                    batch_results[i][pathology] = float(y_class[i][j])
                    batch_results[i][f"{pathology}_class"] = float(pred[i][j])

            recon = reconstruction(batch)
            pred_recon = classifier(recon)
            # print(f"min: {recon.min()}, max: {recon.max()}, mean: {recon.mean()}")
            # print(f"min: {y_recon.min()}, max: {y_recon.max()}, mean: {y_recon.mean()}")
            # normalize recon to [0, 1]
            recon = (recon - recon.min()) / (recon.max() - recon.min())
            pred_recon = pred_recon.squeeze(0)
            pred_recon = torch.sigmoid(pred_recon)

            for i in range(batch_size):
                for j, pathology in enumerate(pathologies):
                    batch_results[i][f"{pathology}_recon"] = float(pred_recon[i][j])

            for i in range(batch_size):
                # Standard metrics
                psnr = peak_signal_noise_ratio(
                    y_recon[i].detach().cpu().numpy().squeeze(),
                    recon[i].detach().cpu().numpy().squeeze(),
                    data_range=1,
                )
                # print(f"psnr: {psnr}")
                batch_results[i]["psnr"] = peak_signal_noise_ratio(
                    y_recon[i].detach().cpu().numpy().squeeze(),
                    recon[i].detach().cpu().numpy().squeeze(),
                    data_range=1,
                )
                batch_results[i]["ssim"] = structural_similarity(
                    y_recon[i].detach().cpu().numpy().squeeze(),
                    recon[i].detach().cpu().numpy().squeeze(),
                    data_range=1,
                )
                batch_results[i]["nrmse"] = mean_squared_error(
                    y_recon[i].detach().cpu().numpy().squeeze(),
                    recon[i].detach().cpu().numpy().squeeze(),
                )

                # LPIPS
                y_recon_lpips = preprocess_for_lpips(y_recon[i].to(device))
                recon_lpips = preprocess_for_lpips(recon[i].to(device))
                batch_results[i]["lpips"] = loss_fn_alex(
                    y_recon_lpips, recon_lpips
                ).item()

            predictions += batch_results

            # Update progress bar description with current stats
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


def evaluate_chexpert(
    config: dict, classifier, reconstruction_model, results_path: str, device, logger
):

    if "number_of_samples" in config["datasets"]:
        number_of_samples = config["datasets"]["number_of_samples"]
    else:
        number_of_samples = None

    os.makedirs(results_path, exist_ok=True)

    # Save config to output directory for reproducibility
    with open(os.path.join(results_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Initialize classifier

    dataset = ChexDataset(config["datasets"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
    )

    results = predict_chexpert(
        dataloader=dataloader,
        pathologies=dataset.pathologies,
        classifier=classifier,
        reconstruction=reconstruction_model,
        number_of_samples=number_of_samples,
        device=device,
        logger=logger,
    )

    # Create DataFrame for results
    results_df = pd.DataFrame(results)

    # Save results to output directory
    results_df.to_csv(
        os.path.join(results_path, f"{config['name']}_results.csv"),
        index=False,
    )
