# 🧠 MS Lesion Segmentation using 2D UNet

## 📌 Overview
This project presents a deep learning pipeline for **Multiple Sclerosis (MS) lesion segmentation** using a **2D UNet architecture** trained on multi-modal MRI scans from the **ISBI dataset (Kaggle version)**.

The objective is to automatically detect and segment MS lesions from brain MRI images by leveraging complementary information from multiple imaging modalities.

---

## 🧬 Input Data & Modalities
Each sample consists of four MRI modalities:

- FLAIR (Fluid Attenuated Inversion Recovery)
- T2-weighted
- PD (Proton Density)
- MPRAGE (T1-weighted anatomical scan)

These modalities are stacked into a **4-channel input tensor**:

Input shape: (4, H, W)

---

## 🧠 Methodology

### 🔹 2D Slice-Based Training
- MRI volumes are converted from 3D into 2D slices
- Each slice is treated as an independent training sample
- Reduces computational cost and memory usage
- Increases effective dataset size

---

### 🔹 Handling Class Imbalance
MS lesion segmentation is highly imbalanced because many slices contain no lesions.

To address this:
- All lesion-containing slices are kept
- Only a fraction of empty slices is used
- Controlled by:
  EMPTY_SLICE_RATIO = 0.25

This prevents the model from learning trivial background predictions.

---

### 🔹 Multi-Modal Fusion
- All four modalities are combined as input channels
- Allows the network to learn complementary features
- Improves lesion detection accuracy

---

## ⚙️ Model Architecture

The model is a **2D UNet**, a fully convolutional encoder-decoder network with skip connections.

### Encoder (Downsampling)
- Two convolution layers per block
- Batch Normalization + ReLU
- MaxPooling for downsampling

### Bottleneck
- Deep feature representation
- Captures global context

### Decoder (Upsampling)
- Transposed convolutions for upsampling
- Skip connections from encoder
- Feature concatenation for detail recovery

### Output Layer
- 1×1 convolution
- Produces binary segmentation mask

---

## 📉 Loss Function

A hybrid loss is used:

- Binary Cross Entropy (BCE)
- Dice Loss

Final Loss:
Final Loss = 0.5 * BCE + 0.5 * Dice

This combination improves performance on small lesion regions.

---

## 📊 Evaluation Metrics

- Dice Score (overlap between prediction and ground truth)
- Precision (reduces false positives)

Combined Score:
Score = (Dice + Precision) / 2

---

## 🧪 Training Configuration

- Epochs: 30
- Batch Size: 4
- Learning Rate: 1e-4
- Optimizer: Adam
- Validation Split: 15%
- Test Split: 15%
- Empty Slice Ratio: 0.25
- Device: CPU

---

## 📊 Results

### Validation Results
- Dice Score: 0.8691
- Precision: 0.9090
- Combined Score: 0.8891

### Test Results
- Dice Score: 0.7242
- Precision: 0.9014
- Final Score: 0.8128

---

## 🖼️ Sample Results

### Training (Last Epoch)
![Training](MS-img/last%20epoch.PNG)

### Final Test Results
![Test](MS-img/final%20test.PNG)

---

## 📁 Project Structure

train.py      # Training pipeline  
test.py       # Evaluation script  
visual.py     # Visualization of predictions  
model.py      # UNet architecture  
losses.py     # Loss functions  
metrics.py    # Evaluation metrics  
MS-img/       # Sample output images  

---

## ▶️ How to Run

### 1. Install dependencies
pip install torch torchvision numpy nibabel matplotlib tqdm

---

### 2. Set dataset path
Update DATA_DIR in scripts:

DATA_DIR = r"E:\training"

---

### 3. Train
python train.py

---

### 4. Test
python test.py

---

### 5. Visualize predictions
python visual.py

---

## 💾 Model Outputs
During training:
- Best model checkpoint is saved
- Checkpoints saved every epoch
- Final trained model saved

Note: `.pth` files are not included in this repository.

---

## ⚠️ Notes
- Dataset is not included (ISBI dataset from Kaggle)
- Model weights are not included
- Training was performed on CPU

---

## 🚀 Future Improvements
- Extend to 3D UNet for volumetric learning
- Apply data augmentation
- Improve generalization on unseen data
- Use attention mechanisms
- Hyperparameter tuning

---

## 👤 Author
Maryam Pirzadeh

---

## ⭐ Acknowledgment
Dataset: ISBI MS Lesion Segmentation Challenge (Kaggle version)
