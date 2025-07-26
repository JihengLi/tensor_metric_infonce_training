# Tensor Metric InfoNCE Training

**Author:** Jiheng Li  
**Copyright:** This project and all its code, documentation, and related content are owned by Jiheng Li. Unauthorized reproduction or commercial use is prohibited.

A PyTorch-based pipeline for training metric learning models using contrastive learning frame. This repository provides flexible configuration, data handling, model architectures, loss implementations, and utilities to streamline contrastive learning experiments.

## Introduction

This project was developed specifically for the Beyond FA competition, which aims to discover diffusion tensor metrics that outperform fractional anisotropy (FA). It implements an end-to-end model that accepts a diffusion tensor image as input and generates a 128-dimensional embedding suited for downstream analysis.

Key highlights:

- **End-to-End Metric Learning:** Maps raw diffusion tensor images to 128-D vectors using a contrastive, unsupervised InfoNCE framework.
- **Data Augmentation:** Applies extensive augmentations to enhance model robustness.
- **High KNN Performance:** Achieved a Top-1 hit rate of **99.7%** on a 2,000-image hold-out evaluation set using a simple K-Nearest Neighbors test.
- **Competition Alzheimer Test Set:** The competition organizers evaluated embeddings by feeding them into their own projection head on the official Alzheimer’s dataset, yielding performance far above random baseline.
- **Future Extensions:** ConvNeXt and EfficientNet backbones have been integrated and will be benchmarked next.

This introduction focuses solely on the project’s design and empirical achievements, without detailing training or execution steps.

## Repository Structure

```
├── configs/ # Experiment configuration files
├── datasets/ # Dataset definitions and loaders
├── losses/ # Loss functions (e.g., InfoNCE, triplet)
├── models/ # Model architectures and backbones
├── scripts/ # Utility scripts for data preparation and evaluation
├── utils/ # Helper functions and common utilities
├── train.py # Main training script
├── inference.py # Script for feature extraction and inference
├── data_extract.ipynb # Notebook for data extraction and visualization
├── test_knn.ipynb # Notebook for KNN evaluation on learned embeddings
├── requirements.txt # Python dependencies
├── tensor_paths.txt # Paths to dataset tensors or precomputed features
└── README.md # Project overview and instructions
```

## Model Architecture

The model consists of a configurable backbone encoder paired with a lightweight projection head to produce 128-D embeddings:

- **Backbone Encoders** (chosen via `configs/*.yaml`):

  - **SE-ResNet3D-18**: ResNet-18 enhanced with Squeeze-and-Excitation (SE) modules [Hu et al., 2018]. The SE blocks recalibrate channel-wise feature responses, and we remove the final classification layer, applying global average pooling to obtain a 256-D feature vector.
  - **ConvNeXt3D-Pico**: A modern convolutional design inspired by Transformer architectures [Liu et al., 2022]. We pool features from the last stage into a 512-D vector.
  - **EfficientNet3D-B3**: Compound-scaled CNN architecture [Tan & Le, 2019], producing a 1024-D representation after global pooling.

- **Projection Head**: A two-layer MLP that maps backbone features to the 128-D embedding space:
  ```yaml
  head:
    - Linear(in_features=<backbone_output_dim>, out_features=512, bias=True)
    - ReLU()
    - Linear(in_features=512, out_features=128, bias=True)
  ```
  The resulting vectors are L2-normalized before computing the InfoNCE loss.

Backbone selection and specific hyperparameters (e.g., pretrained weights, dropout rates) are defined in the corresponding YAML files under `configs/`.
