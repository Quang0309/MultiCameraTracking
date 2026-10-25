# Multi-Camera Person Re-Identification with Cross-Outfit Evaluation

Query-based person ReID system using the MEVID dataset, comparing ResNet-50, OSNet, and TransReID architectures.

## Architecture Overview
Refer to the system architecture diagram for an overview of the pipeline components.

## Features
- **3 ReID architectures**: ResNet-50-IBN, OSNet-x1.0, TransReID ViT-Base
- **3 evaluation protocols**: General, Same-Outfit, Cross-Outfit

## Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# Install training dependencies
pip install -r requirements-train.txt

# Note: fast-reid is installed from source separately
```

## Project Structure

```
MultiCameraTracking/
├── data/           # Dataset files
├── weights/        # Model checkpoints
├── results/        # Experiment outputs
└── fast-reid/      # Cloned fast-reid repository
```
