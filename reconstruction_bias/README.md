# Reconstruction Bias

This repository contains code to train classification, segmentation, and reconstruction from scratch, as well as fine-tuning for bias mitigation. 

## Repository Structure

The repository is organized into two main components:

### 1. Training (`src/`)
- `data/` - Data loading and preprocessing utilities
- `model/` - Model architectures and definitions  
- `trainer/` - Training loops and optimization
- `fairness/` - Fairness metrics and mitigation techniques
- `preprocessing/` - Data preprocessing pipelines

**Capabilities:**
- Train U-Net reconstruction models for UCSF-PDGM and CheXpert datasets from scratch
- Train segmentation and classification models for UCSF
- Fine-tune U-Net models with fairness-aware techniques

## Training from Scratch

### Dependencies 
Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

**Additional Requirements:**
- Download required datasets (TODO: add download instructions)

### Preprocessing

#### CheXpert Low-Dose Generation
Use `src/preprocessing/chex_low_dose.py` to generate low-dose samples from the CheXpert dataset. This preprocessing step is essential to reduce training time and data in this format is required for training.

**Usage:**
```bash
python src/preprocessing/chex_low_dose.py \
    --source_dir <path_to_original_chexpert> \
    --target_dir <path_to_output_low_dose> \
    --metadata <path_to_metadata.csv> \
    --photon_counts "1e4,1e5,1e6" \
    --num_files <optional_file_limit> \
    --num_workers <optional_worker_count>
```

**Arguments:**
- `--source_dir`: Path to original CheXpert dataset
- `--target_dir`: Output directory for low-dose samples  
- `--metadata`: Path to metadata CSV file
- `--photon_counts`: Comma-separated photon count levels (e.g., "1e4,1e5,1e6")
- `--num_files`: Limit number of files to process (optional)
- `--num_workers`: Number of parallel workers (optional, defaults to CPU count)

### Model Training Overview

This repository supports training multiple model types:
- **Classifiers** - For tumor classification tasks
- **Segmentation** - For medical image segmentation  
- **Reconstruction** - For image reconstruction from corrupted inputs

Each training run creates an output folder containing model checkpoints and training metrics.

### Configuration

Model parameters are configured through YAML files located in the `configuration/` folder. 

#### General Configuration Parameters

**Required Parameters:**
- `output_dir`: Directory where all outputs will be saved
- `output_name`: Name of the output model and logs
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for the optimizer
- `batch_size`: Batch size for the DataLoader
- `data_root`: Root directory containing the dataset

**Optional Parameters:**
- `number_of_samples`: Limit number of training samples
- `seed`: Random seed for reproducibility (default: 31415)
- `save_interval`: Model checkpoint save interval in epochs (default: 1)
- `early_stopping_patience`: Early stopping patience in epochs
- `type`: Data type to use (default: "T2")
- `pathology`: List of pathologies to consider (default: ["edema", "non_enhancing", "enhancing"])
- `lower_slice`: Lower slice index for dataset
- `upper_slice`: Upper slice index for dataset

#### Dataset-Specific Configuration

**UCSF-PDGM Dataset:**
```yaml
dataroot: './data/UCSF-PDGM'
type: 'T2'
lower_slice: 60
upper_slice: 130
age_bins: [0, 58, 100]
```

**CheXpert Dataset:**
```yaml
csv_path_A: ../CheXpert_noise/metadata_photon_10000.csv
csv_path_B: ../CheXpert/metadata.csv
dataroot_A: ../CheXpert_noise/
dataroot_B: ../datasets/
```
> **Note:** Path A requires preprocessing to create noisy samples (see Preprocessing section above)

### Training Scripts

#### 1. Classifier Training (UCSF-PDGM)

**Command:**
```bash
python train_classifier.py --opt <path_to_config.yaml>
```

**Classifier-Specific Parameters:**
- `classifier_type`: Classifier type
  - `TTypeBCEClassifier` - Tumor type classification
  - `TGradeBCEClassifier` - Tumor grade classification  
- `network_type`: Neural network architecture (optional)
  - `ResNet18` (default)
  - `ResNet50`
- `age_bins`: Age classification bins (optional, default: [0, 3, 18, 42, 67, 96])
- `eps`: Numerical stability constant for survival classification (optional, default: 1e-8)
- `balancing`: Enable dataset rebalancing during training (optional)

#### 2. Reconstruction Training (UCSF-PDGM)

**Command:**
```bash
python train_reconstruction.py --opt <path_to_config.yaml>
```

**Reconstruction-Specific Parameters:**
- `network_type`: Network architecture (optional)
  - `UNet` (default)
- `network_path`: Path to pre-trained model (optional)
- `sampling_mask`: Undersampling pattern (optional, default: "radial")

#### 3. Segmentation Training (UCSF-PDGM)

**Command:**
```bash
python train_segmentation.py --opt <path_to_config.yaml>
```

**Note:** Uses general configuration parameters only.

#### 4. CheXpert Classification

For CheXpert classification, we use pre-trained models from `torchxrayvision` rather than training from scratch.

## Fine-Tuning for Bias Mitigation

Fine-tune pre-trained reconstruction models with fairness-aware techniques to reduce bias in downstream tasks.

**Command:**
```bash
python train_mitigation.py --opt <path_to_config.yaml>
```

### Mitigation Techniques
- `reweighting` - Reweight training samples based on demographic attributes
- `adversarial` - Adversarial training to reduce demographic prediction  
- `eodd` - Equalized Odds post-processing

### Configuration Examples

#### CheXpert Mitigation
```yaml
# Basic Configuration
output_dir: './output'
output_name: 'unet-fairness-chex'
dataset: 'chex'
mitigation: 'reweighting'  # or 'adversarial', 'eodd'

# Training Parameters
num_epochs: 20
learning_rate: 0.0001
batch_size: 64
early_stopping_patience: 5
save_interval: 5
fairness_lambda: 0.005

# Data Paths
csv_path_A: ../CheXpert_noise/metadata_photon_10000.csv
csv_path_B: ../CheXpert/metadata.csv
dataroot_A: ../CheXpert_noise
dataroot_B: ../CheXpert

# Model Paths
network_type: 'UNet'
model_path: ../models/unet-chex.pth
classifier_path: ../code/torchxrayvision/output/class_norm01/chex-densenet-class_norm01-best.pt
seed: 31415
```

#### UCSF-PDGM Mitigation
```yaml
# Basic Configuration
output_dir: './output'
output_name: 'unet-fairness-ucsf'
dataset: 'ucsf'
mitigation: 'reweighting'  # or 'adversarial', 'eodd'

# Training Parameters
num_epochs: 20
learning_rate: 0.0001
batch_size: 32
early_stopping_patience: 5
save_interval: 5
fairness_lambda: 1

# Data Configuration
dataroot: ../data/UCSF-PDGM
sampling_mask: radial
type: FLAIR
pathology: []
lower_slice: 60
upper_slice: 130
num_rays: 60

# Model Configuration
network_type: 'UNet'
model_path: ../models/unet-ucsf.pth
seed: 31415

# Downstream Classifiers
classifiers: 
  - name: TGradeBCEClassifier
    path: ../models/tgrade/checkpoints/tgrade.pth
  - name: TTypeBCEClassifier
    path: ../models/ttype/checkpoints/ttype.pth
```
