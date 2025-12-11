# Evaluating the Impact of Medical Image Reconstruction on Downstream AI Fairness and Performance

## Code Structure

The code is organized into the following directories, each containing its own README with detailed information:

### Core Directories

- **`evaluation/`** - Analysis and plot generation for the submission
- **`reconstruction_bias/`** - Segmentation and classification for UCSF, U-Net training for CheXpert and UCSF from scratch, and U-Net mitigation methods
- **`sde/`** - SDE training from scratch for UCSF and CheXpert
- **`sde_fairness/`** - SDE fine-tuning for fairness on UCSF and CheXpert
- **`gan/`** - GAN training from scratch for UCSF and CheXpert  
- **`gan_fairness/`** - GAN fine-tuning for fairness on UCSF and CheXpert

## Implementation Notes

- U-Net implementation is our own custom implementation
- GAN and SDE implementations are forked and adapted, with separate codebases for training from scratch versus fairness mitigation
- CheXpert classification is performed using the [torchxrayvision](https://github.com/mlmed/torchxrayvision) repository
