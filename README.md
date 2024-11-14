# Image Translation: PSMA-PET/CT to Bone Scan via GANs

## Overview
This repository provides code for training a deep learning model for translating whole-body PSMA-PET/CT images to bone scans using a Pix2Pix network. The model leverages a revised U-Net architecture for the generator, with adjustments to both the architecture and the loss function to improve compatability with 3D-input, 2D-output data and translation quality. **Note:** This project is in its initial phase and is based on private UCSF data.

## Model Details
- **Framework**: TensorFlow
- **GAN Architecture**: [Pix2Pix](https://arxiv.org/abs/1611.07004) with max pooling in the coronal axis for concatenation in the Generator. Loss function modifications inspired by [MedGAN](https://arxiv.org/abs/1806.06397) and [Rajagopal et al](https://arxiv.org/abs/2206.05618).
- **Metrics**: PSNR, SSIM, intensity-histogram loss

## Dataset
The model was trained on a private dataset from UCSF. Weights will be released after approval. If you'd like to train the model on your own data, please ensure it aligns with the preprocessing requirements outlined below.
- PET/CT scans should be resampled to (2, 2, 4) voxel spacing.
- Files must be organized in the following directory structure:
```bash
train/
├── patient_001/
│   ├── CT.npy
│   ├── PT_SUV.npy
│   ├── frontNM.npy
│   └── backNM.npy
├── patient_002/

val/
├── patient_100/
│   ├── CT.npy
│   ├── PT_SUV.npy
│   ├── frontNM.npy
│   └── backNM.npy
├── patient_200/
```
## Project Structure
models.py: Contains the Pix2Pix generator and discriminator models.
data_loader.py: Manages loading and preprocessing of PET/CT and bone scan data.
train.py: Runs the training loop, validation loop, outputs metrics, and saves results.
utils.py: Utility functions for data handling and metric calculations.

## Setup Instructions
This project requires Python 3.8. To set up the environment and dependencies, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Hong-Laboratory/PSMA2BoneScan.git
   cd PSMA
   ```
2. **Install dependencies**:
   ```bash
   pip install requirements.txt
   ```
3. Organize your dataset following the required structure above.

## Usage
Run the training with:
```bash
python train.py --data_dir ./train --val_dir ./val --sample_image_dir ./train/sample_patient --output_path ./results --epochs 100 --save_every 20
```
For additional options, use:
```bash
python train.py --help
```

## Results
After every `--save_every` epoch, metrics and samples are saved to the specified output directory

- Losses & Metrics: Generator loss, intensity histogram loss, style loss, validation losses, PSNR, SSIM as lineplots in PNG format
- Output Samples: Images saved in PNG format for both training and validation samples
- Checkpoint: Model weights are saved (max keep = 5)


## License
```bash
# Copyright (C) UCSF/Inkyu/Julian 2024
# GNU General Public License v2.0
# Please see LICENSE and README.md
```

## Acknowledgments
Code is inspired by [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix?tab=readme-ov-file)
