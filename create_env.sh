#!/bin/bash

# Load CUDA modules
module load cuda/11.7
module load cudnn/8.5.0.96-11.7-cuda

# Create conda environment
conda env create -f environment.yml

# Activate
conda activate japanal

# Install PyTorch GPU-enabled (CPU wheel but uses system CUDA)
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install torchvision --index-url https://download.pytorch.org/whl/cpu
