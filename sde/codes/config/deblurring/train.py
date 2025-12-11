import argparse
import logging
import math
import os
import random
import sys
import copy

import cv2
import numpy as np
import torch
import wandb  # Add this import at the top
from tqdm import tqdm  # Add this import at the top

# from IPython import embed

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### set random seed
    seed = opt["train"]["manual_seed"]

    # Set distributed training to false
    opt["dist"] = False

    torch.backends.cudnn.benchmark = True

    # Add val_images to path options
    opt["path"]["val_images"] = os.path.join(
        opt["path"]["experiments_root"], "val_images"
    )

    util.mkdir_and_rename(
        opt["path"]["experiments_root"]
    )  # rename experiment folder if exists
    util.mkdirs(
        (
            path
            for key, path in opt["path"].items()
            if not key == "experiments_root"
            and "pretrain_model" not in key
            and "resume" not in key
        )
    )
    os.system("rm ./log")
    os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

    # config loggers. Before it, the log will not work
    util.setup_logger(
        "base",
        opt["path"]["log"],
        "train_" + opt["name"],
        level=logging.INFO,
        screen=False,
        tofile=True,
    )
    util.setup_logger(
        "val",
        opt["path"]["log"],
        "val_" + opt["name"],
        level=logging.INFO,
        screen=False,
        tofile=True,
    )
    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))
    # Replace tensorboard setup with wandb
    if opt["use_tb_logger"] and "debug" not in opt["name"]:
        wandb.init(
            project="image-restoration-sde",  # Your project name
            name=opt["name"],
            config={
                # Model parameters
                "model": opt["model"],
                "network_G": opt["network_G"]["which_model_G"],
                # SDE parameters
                "sde_max_sigma": opt["sde"]["max_sigma"],
                "sde_T": opt["sde"]["T"],
                "sde_schedule": opt["sde"]["schedule"],
                "sde_eps": opt["sde"]["eps"],
                # Dataset parameters
                "batch_size": opt["datasets"]["batch_size"],
                "dataset": opt["datasets"]["dataset"],
                # Training parameters
                "niter": opt["train"]["niter"],
                "lr_G": opt["train"]["lr_G"],
                "optimizer": opt["train"]["optimizer"],
            },
        )

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch

    train_set = create_dataset(opt["datasets"], train=True)
    train_size = int(math.ceil(len(train_set) / opt["datasets"]["batch_size"]))
    total_iters = int(opt["train"]["niter"])
    total_epochs = int(math.ceil(total_iters / train_size))
    train_loader = create_dataloader(train_set, opt)
    logger.info(
        "Number of train images: {:,d}, iters: {:,d}".format(len(train_set), train_size)
    )
    logger.info(
        "Total epochs needed: {:d} for iters {:,d}".format(total_epochs, total_iters)
    )

    val_set = create_dataset(opt["datasets"], train=False)
    val_loader = create_dataloader(val_set, opt)

    logger.info(
        "Number of val images in [{:s}]: {:d}".format(
            opt["datasets"]["dataset"], len(val_set)
        )
    )

    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt)
    device = model.device
    print(f"device: {device}", flush=True)

    current_step = 0
    start_epoch = 0

    sde = util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"],
        T=opt["sde"]["T"],
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=device,
    )
    sde.set_model(model.model)

    scale = opt["degradation"]["scale"]

    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    # Define frequencies for different operations
    train_freq = train_size / opt["logger"]["print_per_epoch"]

    best_psnr = 0.0
    best_iter = 0

    for epoch in range(start_epoch, total_epochs + 1):
        # Add progress bar for each epoch
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}")

        for train_data in train_bar:
            current_step += 1
            if current_step > total_iters:
                break

            # Training step
            LQ, GT = train_data["LQ"], train_data["GT"]
            timesteps, states = sde.generate_random_states(x0=GT, mu=LQ)
            model.feed_data(states, LQ, GT)
            model.optimize_parameters(current_step, timesteps, sde)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

            # Update progress bar with loss
            train_bar.set_postfix({"loss": model.get_current_log()["loss"]})

            # Print training progress and log training samples
            if current_step % train_freq == 0:
                # Log training metrics only
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                wandb.log({f"train/loss": logs["loss"]}, step=current_step)
                message += "{:s}: {:.4e} ".format("loss", logs["loss"])
                logger.info(message)

                val_images = []
                num_val_samples = min(2, len(val_loader))  # Show 5 validation samples

                for idx, val_data in enumerate(val_loader):
                    if idx >= num_val_samples:
                        break

                    LQ, GT = val_data["LQ"], val_data["GT"]
                    noisy_state = sde.noise_state(LQ)
                    model.feed_data(noisy_state, LQ, GT)
                    model.test(sde)
                    visuals = model.get_current_visuals()

                    # Process images for this sample
                    output = (
                        util.tensor2img(visuals["Output"].squeeze())
                        if "Output" in visuals
                        else None
                    )
                    gt_img = util.tensor2img(visuals["GT"].squeeze())
                    lq_img = util.tensor2img(visuals["Input"].squeeze())

                    if output is not None:
                        val_images.extend(
                            [
                                wandb.Image(lq_img, caption=f"Sample {idx+1} - Input"),
                                wandb.Image(output, caption=f"Sample {idx+1} - Output"),
                                wandb.Image(
                                    gt_img, caption=f"Sample {idx+1} - Ground Truth"
                                ),
                            ]
                        )

                wandb.log({"val/samples": val_images}, step=current_step)

        # Full validation at the end of each epoch
        logger.info(f"Doing full validation at epoch {epoch}...")
        avg_psnr = 0.0

        for idx, val_data in enumerate(val_loader):
            if idx >= 1:
                break
            LQ, GT = val_data["LQ"], val_data["GT"]
            noisy_state = sde.noise_state(LQ)
            model.feed_data(noisy_state, LQ, GT)
            model.test(sde)
            visuals = model.get_current_visuals()

            output = util.tensor2img(visuals["Output"].squeeze())
            gt_img = util.tensor2img(visuals["GT"].squeeze())
            avg_psnr += util.calculate_psnr(output, gt_img)

        avg_psnr = avg_psnr / len(val_loader)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_iter = current_step
        # Update best model
        """if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_iter = current_step
            logger.info("Saving best model...")
            model.save('best')"""

        logger.info(
            "# Epoch {} Validation # PSNR: {:.4f}, Best PSNR: {:.4f} @ iter {}".format(
                epoch, avg_psnr, best_psnr, best_iter
            )
        )

        wandb.log(
            {
                "val/epoch": epoch,
                "val/full_psnr": avg_psnr,
                "val/best_psnr": best_psnr,
                "val/best_iter": best_iter,
            },
            step=current_step,
        )

        # Save epoch checkpoint
        if epoch % opt["logger"]["save_checkpoint_freq"] == 0:
            logger.info("Saving checkpoint...")
            model.save(epoch)
            model.save_training_state(epoch, current_step)

    logger.info("Saving final model...")
    model.save("latest")
    logger.info("End of training.")
    if opt["use_tb_logger"] and "debug" not in opt["name"]:
        wandb.finish()


if __name__ == "__main__":
    main()
