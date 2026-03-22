# 🧠 MS Lesion Segmentation using UNet

## Overview
This project implements a UNet-based deep learning pipeline for **Multiple Sclerosis (MS) lesion segmentation** using **multi-modal MRI** from the **ISBI dataset (Kaggle version)**.

The model performs binary lesion segmentation from 4 MRI modalities:
- FLAIR
- T2
- PD
- MPRAGE

---

## Model
- **Architecture:** UNet
- **Input channels:** 4
- **Output channels:** 1
- **Task:** Binary segmentation
- **Loss:** BCE + Dice Loss
- **Optimizer:** Adam

---

## Dataset
This project uses the **ISBI MS lesion segmentation dataset** downloaded from **Kaggle**.

**Note:** The dataset is **not included** in this repository.

---

## Training Setup
- **Epochs:** 30
- **Batch size:** 4
- **Learning rate:** 1e-4
- **Validation split:** 15%
- **Test split:** 15%
- **Empty slice ratio:** 0.25

The training pipeline converts 3D MRI volumes into 2D slices and keeps a controlled number of empty slices to reduce imbalance.

---

## Results

### Validation Results
- **Best Validation Dice:** 0.8691
- **Best Validation Precision:** 0.9090
- **Best Validation Score:** 0.8891

### Test Results
- **Test Dice:** 0.7242
- **Test Precision:** 0.9014
- **Test Score:** 0.8128

---

## Project Structure

```text
train.py      # Training pipeline
test.py       # Evaluation on test set
visual.py     # Visualization of predictions
model.py      # UNet model architecture
losses.py     # Loss functions
metrics.py    # Evaluation metrics
MS-img/       # Result images
