# Vision Transformer (ViT) From Scratch

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)

Implementation of Vision Transformer (ViT) architecture from scratch following the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929). This project includes a complete implementation of the ViT model with training on a 10% subset of ImageNet-1K.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Visualization](#visualization)
- [Usage](#usage)
- [Results](#results)
- [References](#references)
- [License](#license)

## Features

- Complete implementation of Vision Transformer from scratch
- Custom patch embedding layer
- Multi-head self-attention blocks
- Transformer encoder implementation
- Training on ImageNet-1K subset
- Comprehensive visualizations of image patching and attention
- Learning rate scheduling with warmup

## Installation

First, clone this repository:

```bash
git clone https://github.com/404-ammar-not-found/ViT-In-Pytorch.git
cd ViT-In-Pytorch
```

Create a virtual environment and install dependencies:

```bash
conda create -n vit python=3.9 -y
conda activate vit
pip install -r requirements.txt
```

Alternatively using pip:

```bash
python -m venv vit_env
source vit_env/bin/activate  # Linux/MacOS
# vit_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Requirements

- PyTorch 2.8.0+
- torchvision 0.24.0+
- torchinfo 1.8.0+
- matplotlib 3.10.8+
- tqdm 4.67.1+

## Dataset

This implementation uses a 10% subset of the ImageNet-1K dataset for training and evaluation. To use the dataset:

1. Download the ImageNet dataset (or the subset) from [Kaggle](https://www.kaggle.com/datasets/ambityga/imagenet100)
2. Extract the archive to the `archive/` directory with the following structure:
   ```
   archive/
   ├── train/
   │   ├── class1/
   │   ├── class2/
   │   └── ...
   └── test/
       ├── class1/
       ├── class2/
       └── ...
   ```

## Model Architecture

The Vision Transformer model implemented in this notebook follows the architecture described in the original ViT paper:

```
Input Image → Patch Embedding → Positional Encoding → Transformer Encoder (x12) → MLP Head → Classification
```

Key components:
- **Patch Embedding**: Splits the image into 16×16 patches
- **Positional Encoding**: Adds positional information to patches
- **Transformer Encoder**: 12 layers with 6 attention heads
- **MLP Head**: Final classification layer


## Training

The model is trained with the following hyperparameters:
- Batch size: 512
- Learning rate: 3e-3 with linear warmup
- Optimizer: AdamW with weight decay 0.3
- Loss function: Cross-Entropy
- Training epochs: 2 (example in notebook)

## References

1. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
2. [ImageNet Dataset](http://www.image-net.org/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Thanks to the original authors of the Vision Transformer paper
- PyTorch team for the excellent deep learning framework
- [torchinfo](https://github.com/TylerYep/torchinfo) for model visualization tools

---

**Note**: This is a research implementation and may require modifications for production use. Training full ViT models requires significant computational resources. The notebook demonstrates the concepts with a smaller subset of ImageNet for accessibility on consumer hardware.