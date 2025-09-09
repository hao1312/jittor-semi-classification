# Dual-Model Semi-Supervised Medical Image Classification

## Project Overview

This project implements a dual-model semi-supervised learning framework for medical image classification using Jittor deep learning framework. The system combines Swin Transformer and Vision Transformer (ViT) architectures with advanced semi-supervised techniques to address class imbalance and limited labeled data challenges in medical imaging.

## Key Features

- **Dual-Model Architecture**: Simultaneous training of Swin Transformer and Vision Transformer models
- **Semi-Supervised Learning**: Leverages unlabeled data to enhance model performance
- **Adaptive Weight Fusion**: Uses SMC (Self-Mutual Confidence) integration for dynamic model prediction fusion
- **Class Imbalance Handling**: Implements RW-LDAM-DRW loss function for addressing class imbalance
- **EMA Teacher Models**: Utilizes Exponential Moving Average (EMA) for training stability
- **K-Fold Cross Validation**: Supports cross-validation training strategy

## Methodology

### 1. Co-Training Mechanism
Two architecturally different models (Swin and ViT) learn from each other through CDCR (Consensus-Divergence Collaborative Regulation) loss, promoting model collaboration.

### 2. Semi-Supervised Learning
EMA teacher models generate pseudo-labels for unlabeled data, which student models then learn from.

### 3. Dynamic Confidence Weighting
Adaptive weighting of pseudo-labels based on model prediction consistency and confidence.

### 4. Class-Balanced Sampling
Oversampling strategy based on class distribution to improve learning for minority classes.

## Usage

### Training the Model

```bash
python train.py --transform_cfg /path/to/transform.yml --root_path /path/to/dataset --res_path /path/to/results --exp experiment_name --fold 0 --total_folds 4

## Dependencies

- **Python 3.7+**
- **Jittor** deep learning framework
- Standard scientific computing libraries (`numpy`, `yaml`, etc.)

## Dataset Preparation

1. Organize labeled data in the structure: `root_path/images/train/`
2. Place unlabeled data in the directory specified by `--unlabeled_dir`
3. Create a label file at `root_path/labels/trainval.txt` with format: `image_path label`

## Outputs

Training generates:
- Best models for both Swin and ViT (standard and EMA versions)
- Training logs
- Validation results and confusion matrices
- Periodic checkpoint models

## Notes

1. Prepare both labeled and unlabeled medical image datasets
2. Adjust paths in configuration files according to your environment
3. The default implementation is for 6-class classification; adjust for different tasks
4. Models are saved at the end of training and for best performance checkpoints

This project is particularly suited for medical image analysis scenarios where labeled data is limited but unlabeled data is relatively abundant. The semi-supervised learning approach with dual-model co-training effectively enhances model performance in such settings.
