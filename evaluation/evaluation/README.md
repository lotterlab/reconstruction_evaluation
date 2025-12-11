

# Evaluation Pipeline

The evaluation process consists of three sequential stages:

1. **Processing** - Preprocess computationally intensive model outputs (e.g., SDE inference)
2. **Prediction** - Apply downstream models to reconstructions and measure performance/fairness metrics  
3. **Evaluation** - Generate statistical analyses, significance tests, plots, and tables

## 1. Processing

Pre-process outputs from computationally intensive models (especially SDE inference) to avoid repeated computation during evaluation.

### Configuration Examples

#### CheXpert SDE Processing
```yaml
# Basic Configuration
name: chex_SDE-10000-eodd
results_path: ./processed_images/eodd/chex
batch_size: 32
num_workers: 4

# Code Paths
paths:
  recon_bias: ./code/recon_bias
  pix2pix: ./code/pix2pix
  sde: ./code/image-restoration-sde

# Dataset Configuration
datasets:
  name: chex
  csv_path_A: ./CheXpert_noise/metadata_photon_10000.csv
  csv_path_B: ./CheXpert/metadata.csv
  dataroot_A: ./CheXpert_noise
  dataroot_B: ./CheXpert
  seed: 31415
  split: test

# Model Configuration
models:
  network: Diffusion
model: denoising
distortion: deblur
gpu_ids: [0]

# SDE Parameters
sde:
  max_sigma: 10
  T: 100
  schedule: cosine  # linear, cosine
  eps: 0.005

# Degradation Settings
degradation:
  sigma: 25
  noise_type: G

# Network Architecture
network_G:
  which_model_G: ConditionalUNet
  setting:
    in_nc: 1
    out_nc: 1
    nf: 64
    depth: 4

# Model Path
path:
  pretrain_model_G: ./code/image-restoration-sde/experiments/deblurring/chex10000-sde-fairness-eodd/models/40_G.pth
  strict_load: True
```

#### UCSF-PDGM SDE Processing
```yaml
# Basic Configuration
name: ucsf_SDE-8-eodd
results_path: ./processed_images/eodd/ucsf
batch_size: 16
num_workers: 4

# Code Paths
paths:
  recon_bias: ./code/recon_bias
  pix2pix: ./code/pix2pix
  sde: ./code/image-restoration-sde

# Dataset Configuration
datasets:
  name: ucsf
  dataroot: ./data/UCSF-PDGM
  sampling_mask: radial
  seed: 31415
  type: FLAIR
  pathology: []
  lower_slice: 60
  upper_slice: 130
  split: test
  num_rays: 60

# Model Configuration
models:
  network: Diffusion
model: denoising
distortion: deblur
gpu_ids: [0]

# SDE Parameters
sde:
  max_sigma: 10
  T: 100
  schedule: cosine
  eps: 0.005

# Degradation Settings
degradation:
  sigma: 25
  noise_type: G

# Network Architecture
network_G:
  which_model_G: ConditionalUNet
  setting:
    in_nc: 1
    out_nc: 1
    nf: 64
    depth: 4

# Model Path
path:
  pretrain_model_G: ./code/image-restoration-sde/experiments/deblurring/ucsf-sde-8-fairness-eodd/models/160_G.pth
  strict_load: True
```

**Usage:** See specific documentation for processing script invocation.


## 2. Prediction

Apply downstream task models to reconstructed images and measure performance and fairness metrics.

### Configuration Examples

#### CheXpert Prediction
```yaml
# Basic Configuration
name: chex_unet_10000
results_path: ./predictions_adv
batch_size: 8
num_workers: 2

# Dataset Configuration
datasets:
  name: chex
  csv_path_A: ./CheXpert_noise/metadata_photon_10000.csv
  csv_path_B: ./CheXpert/metadata.csv
  dataroot_A: ./CheXpert_noise
  dataroot_B: ./CheXpert
  seed: 31415
  split: test

# Reconstruction Model
models:
  network: UNet
  model_path: ./models/unet-chex.pth

# Downstream Task Models
task_models: 
  name: chexpert-classifier
  path: ./code/torchxrayvision/output/class_norm01/chex-densenet-class_norm01-best.pt
```

#### UCSF-PDGM Prediction
```yaml
# Basic Configuration
name: ucsf_unet_08
results_path: ./predictions_adv
batch_size: 64
num_workers: 8

# Dataset Configuration
datasets:
  name: ucsf
  dataroot: ./data/UCSF-PDGM
  sampling_mask: radial
  seed: 31415
  type: FLAIR
  pathology: []
  lower_slice: 60
  upper_slice: 130
  split: test
  num_rays: 60

# Reconstruction Model
models:
  network: UNet
  model_path: ./models/unet-ucsf.pth

# Downstream Task Models
task_models: 
  name: ucsf
  models: 
    - type: classifier
      name: TGradeBCEClassifier
      path: ./models/tgrade/checkpoints/tgrade.pth
    - type: classifier
      name: TTypeBCEClassifier
      path: ./models/ttype/checkpoints/ttype.pth
    - type: segmentation
      name: UNet
      path: ./models/segmentation_skews/segmentation/segmentation.pth
```

**Usage:** See specific documentation for prediction script invocation.

## 3. Evaluation

Generate comprehensive statistical analyses, significance tests, plots, and tables from prediction results.

### Process Overview

1. **Individual Evaluations**: Run evaluation for each dataset (CheXpert, UCSF-PDGM) and each mitigation technique separately
2. **Combined Analysis**: Aggregate results across experiments to create comprehensive fairness and performance comparisons  
3. **Output Generation**: Create statistical significance tests, visualization plots, and summary tables

### Usage

**Command:**
```bash
python evaluation.py --opt <path_to_config.yaml>
```

**Workflow:**
1. Run `evaluation.py` with CheXpert-specific configuration
2. Run `evaluation.py` with UCSF-PDGM-specific configuration  
3. Specify fairness and performance CSV files in combined configuration yaml file for cross-experiment analysis

**Outputs:**
- Statistical significance tests comparing mitigation techniques
- Performance metrics across different reconstruction methods
- Fairness metrics demonstrating bias reduction
- Publication-ready plots and tables
