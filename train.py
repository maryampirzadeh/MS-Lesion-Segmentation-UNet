import os
import re
import glob #to find files
import random
import numpy as np
import nibabel as nib
#data formats like NIfTI (like .nii, .nii.gz). 
#This is crucial because medical images are often stored in this format.
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import UNet


# =========================
# CONFIG(reading data)
# =========================
DATA_DIR = r"E:\training"   # Change if needed
#r  prevents backslashes from being interpreted as escape sequences.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#cpu was used in this project

BATCH_SIZE = 4
NUM_EPOCHS = 30
LR = 1e-4 #used 1e-8 for last 10 epochs
SEED = 42

VAL_RATIO = 0.15
TEST_RATIO = 0.15
EMPTY_SLICE_RATIO = 0.25
# handling slices with missing or empty data, which can occur in medical images

SAVE_DIR = "weights"
#creating folder for weight
BEST_MODEL_NAME = "best_unet_dice_precision.pth"
#considering both dice and precision
BEST_DICE = "dices.pth"
CHECKPOINT_NAME = "checkpoint_unet.pth"
#to save state of the model
FINAL_MODEL_NAME = "final_unet_model.pth"
#final trained model’s weights

os.makedirs(SAVE_DIR, exist_ok=True)
#Creates the weights directory if it doesn't already exist. 
#The exist_ok=True argument prevents an error if the directory already exists.

BEST_MODEL_PATH = os.path.join(SAVE_DIR, BEST_MODEL_NAME)
CHECKPOINT_PATH = os.path.join(SAVE_DIR, CHECKPOINT_NAME)
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, FINAL_MODEL_NAME)
DICE_PATH = os.path.join(SAVE_DIR, BEST_DICE)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
#Sets the random seeds for all three major Python libraries (random, NumPy, and PyTorch).


# =========================
# DATA HELPERS
# =========================
def load_nifti(path):
    return nib.load(path).get_fdata().astype(np.float32)
#converting images to NumPy arrays


def normalize(img):
    return (img - np.mean(img)) / (np.std(img) + 1e-8)
#to scale features to a similar range


def merge_masks(mask_paths):
    masks = [load_nifti(p) for p in sorted(mask_paths)]
    merged = np.zeros_like(masks[0], dtype=bool)
    # Creates a NumPy array filled with zeros
    for m in masks:
        merged |= (m > 0)
        # OR opt
        # It sets the corresponding pixel in the merged array to True if the value of the current mask (m) at that pixel is greater than zero
        # (meaning it's a segmentation).
    return merged.astype(np.float32)


def get_patient_scans(patient_dir, image_folder="preprocessed"):
    img_dir = os.path.join(patient_dir, image_folder)
    mask_dir = os.path.join(patient_dir, "masks")

    img_files = glob.glob(os.path.join(img_dir, "*.nii"))
    mask_files = glob.glob(os.path.join(mask_dir, "*.nii"))

    scans = {}
    '''
    Assuming a patient named "training123_456" has scans for FLAIR and MPRAGE
    scans = {
    "training123_456": {
        "flair": "/path/to/patient_dir/preprocessed/training123_456_flair.nii",
        "mprage": "/path/to/patient_dir/preprocessed/training123_456_mprage.nii",
        }
    }
    '''

    for f in img_files:
        name = os.path.basename(f).replace(".nii", "")
        m = re.match(r"(training\d+_\d+)_(flair|mprage|pd|t2).*", name)
        if m:
            scan_id, modality = m.groups()
            #scan_id: A unique identifier for each scan (e.g., "training123_456").
            #modality: The type of image data (e.g., "flair", "mprage").
            scans.setdefault(scan_id, {})
            scans[scan_id][modality] = f
            #It builds a dictionary (scans) where keys are scan_ids and values are dictionaries 
            #containing the paths to the images for that scan.

    for f in mask_files:
        name = os.path.basename(f)
        m = re.match(r"(training\d+_\d+)_mask\d+\.nii", name)
        if m:
            scan_id = m.group(1)
            scans.setdefault(scan_id, {})
            scans[scan_id].setdefault("masks", [])
            scans[scan_id]["masks"].append(f)

    return scans
    #This dictionary contains all the image data organized by patient ID.

def build_sample_index(data_dir):
    patient_dirs = sorted(glob.glob(os.path.join(data_dir, "training*")))
    #sorts them alphabetically
    positive_samples = []
    empty_samples = []
    #Initializes two lists to store positive (non-zero mask) and empty (zero mask) samples

    for patient_dir in patient_dirs:
        scans = get_patient_scans(patient_dir, image_folder="preprocessed")
        for scan_id, info in sorted(scans.items()):
            needed = ["flair", "t2", "pd", "mprage", "masks"]
            # required image files for a scan
            if not all(k in info for k in needed):
                #Checks if all the necessary files are present in the info dictionary
                print(f"Skipping {scan_id}: missing files")
                continue

            flair = normalize(load_nifti(info["flair"]))
            t2 = normalize(load_nifti(info["t2"]))
            pd = normalize(load_nifti(info["pd"]))
            mprage = normalize(load_nifti(info["mprage"]))
            mask = merge_masks(info["masks"])
            #into a single mask array

            if not (flair.shape == t2.shape == pd.shape == mprage.shape == mask.shape):
                print(f"Skipping {scan_id}: shape mismatch")
                # Checks if all loaded image arrays have the same shape. 
                continue

            for z in range(mask.shape[2]):
                #Iterates through each slice (z-axis) of the 3D mask
                m = mask[:, :, z]

                sample = {
                    "scan_id": scan_id,
                    "z": z,
                    # slice index (z)
                    "flair": info["flair"],
                    "t2": info["t2"],
                    "pd": info["pd"],
                    "mprage": info["mprage"],
                    "mask_paths": sorted(info["masks"]),
                }

                if np.sum(m) > 0:
                    #Checks if any non-zero pixels are present in the mask for the current slice
                    positive_samples.append(sample)
                else:
                    empty_samples.append(sample)

    n_empty_keep = int(len(positive_samples) * EMPTY_SLICE_RATIO)
    #Calculates the number of empty samples
    if len(empty_samples) > 0:
        #The code then randomly selects (n_empty_keep) empty samples from the list. 
        empty_samples = random.sample(empty_samples, min(n_empty_keep, len(empty_samples)))
    else:
        #If there are no empty samples, it creates an empty list for empty_samples.
        empty_samples = []

    all_samples = positive_samples + empty_samples
    random.shuffle(all_samples)
    return all_samples
    #Returns the shuffled list of samples.


def split_by_scan(samples, val_ratio=0.15, test_ratio=0.15):
    scan_ids = sorted(list(set(s["scan_id"] for s in samples)))
    #This line extracts all unique scan_id values from the input samples.
    random.shuffle(scan_ids)

    n_total = len(scan_ids)
    n_test = max(1, int(n_total * test_ratio))
    #ensures that at least one scan is assigned to the test set, even if the ratio is very small.
    n_val = max(1, int(n_total * val_ratio))

    test_scans = set(scan_ids[:n_test])
    val_scans = set(scan_ids[n_test:n_test + n_val])
    train_scans = set(scan_ids[n_test + n_val:])

    train_samples = [s for s in samples if s["scan_id"] in train_scans]
    #Filters the original samples list to include only those samples whose scan_id is present in the train_scans set.
    val_samples = [s for s in samples if s["scan_id"] in val_scans]
    test_samples = [s for s in samples if s["scan_id"] in test_scans]

    return train_samples, val_samples, test_samples


# =========================
# DATASET
# =========================
class MSDataset2D(Dataset):
#It loads and preprocesses multiple modalities (flair, T2, PD, MPRAge) and creates a combined image and mask tensor for each sample.
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        z = s["z"]

        flair = normalize(load_nifti(s["flair"]))[:, :, z]
        t2 = normalize(load_nifti(s["t2"]))[:, :, z]
        pd = normalize(load_nifti(s["pd"]))[:, :, z]
        mprage = normalize(load_nifti(s["mprage"]))[:, :, z]
        mask = merge_masks(s["mask_paths"])[:, :, z]

        image = np.stack([flair, t2, pd, mprage], axis=0).astype(np.float32)
        # It loads and preprocesses multiple modalities (flair, T2, PD, MPRAge) and creates a combined image and mask tensor for each sample.
        mask = np.expand_dims(mask.astype(np.float32), axis=0)
        #Converts the mask (which is a 3D NumPy array) into a 4D array with shape (1, z, z, z) by adding a batch dimension (axis=0).
        #This is necessary because PyTorch expects data to be in batches.

        return torch.tensor(image), torch.tensor(mask)
        # Converts the NumPy arrays representing the image and mask into PyTorch tensors and returns them as a tuple.


# =========================
# LOSS + METRICS
# =========================
def dice_loss_from_logits(logits, targets, smooth=1e-6):
    #especially when dealing with sigmoid activation functions.
    #logits: A PyTorch tensor containing the raw output of a convolutional layer (before applying sigmoid). These are the unnormalized predictions.
    #targets: A PyTorch tensor containing the ground truth segmentation mask (typically 0 or 1, representing background and foreground respectively)
    probs = torch.sigmoid(logits)
    probs = probs.reshape(-1)
    #Reshapes the probs tensor into a 1D vector. 
    targets = targets.reshape(-1)

    intersection = (probs * targets).sum()
    #Calculates the intersection between the predicted probabilities and the ground truth masks.
    #The result represents the overlap between the predicted and true regions.
    dice = (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
    #(2 * |Prediction ∩ Ground Truth|)/(|Prediction| + |Ground Truth|)
    return 1.0 - dice


def dice_score_from_logits(logits, targets, threshold=0.5, smooth=1e-6):
    #A threshold value (defaulting to 0.5) used to convert probabilities into binary predictions.
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    #It converts the predicted probabilities (probs) into binary predictions (preds).

    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()
    #.item() extracts the numerical value from the PyTorch tensor


def precision_score_from_logits(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()

    precision = (tp + smooth) / (tp + fp + smooth)
    # Calculates the Precision score using the standard formula
    return precision.item()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        #Binary Cross-Entropy (BCE) and Dice Loss for segmentation tasks

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        dice = dice_loss_from_logits(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice
        #Combines the BCE loss and the Dice loss using their respective weights, returning the final combined loss value.


# =========================
# TRAIN / EVAL
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, num_epochs):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_precision = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=True)
    # Creates a tqdm progress bar to display the training progress.

    for images, masks in pbar:
        #Iterates through each batch of data provided by the loader
        images = images.to(device)
        masks = masks.to(device)
        #Moves the input images and masks to the specified device (CPU or GPU).

        optimizer.zero_grad()
        #Resets the gradients of all parameters in the model before calculating new gradients. 
        logits = model(images)
        #Performs a forward pass through the model with the input images
        loss = criterion(logits, masks)
        # Calculates the combined loss using the criterion
        loss.backward()
        # Performs backpropagation to compute the gradients of the loss
        optimizer.step()
        # Updates the model's parameters 

        batch_dice = dice_score_from_logits(logits.detach(), masks)
        #Calculates the Dice score for the current batch
        batch_precision = precision_score_from_logits(logits.detach(), masks)
        #Calculates the precision score for the current batch 

        total_loss += loss.item()
        total_dice += batch_dice
        total_precision += batch_precision

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{batch_dice:.4f}",
            "prec": f"{batch_precision:.4f}"
        })

    n = len(loader)
    #Gets the total number of batches in the loader.
    return total_loss / n, total_dice / n, total_precision / n


@torch.no_grad()
#it disables gradient tracking and memory allocation for the forward pass calculations.
def evaluate(model, loader, criterion, device, desc="Validation"):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_precision = 0.0

    pbar = tqdm(loader, desc=desc, leave=True)

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        batch_dice = dice_score_from_logits(logits, masks)
        batch_precision = precision_score_from_logits(logits, masks)

        total_loss += loss.item()
        total_dice += batch_dice
        total_precision += batch_precision

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{batch_dice:.4f}",
            "prec": f"{batch_precision:.4f}"
        })

    n = len(loader)
    return total_loss / n, total_dice / n, total_precision / n


# =========================
# MAIN
# =========================
def main():
    print(f"Device: {DEVICE}")
    print("Building dataset...")

    samples = build_sample_index(DATA_DIR)
    train_samples, val_samples, test_samples = split_by_scan(
        samples,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )

    print(f"Total samples: {len(samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    train_dataset = MSDataset2D(train_samples)
    val_dataset = MSDataset2D(val_samples)
    test_dataset = MSDataset2D(test_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda")
        #It tells PyTorch to copy the data from GPU memory to CPU memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda")
    )

    model = UNet(in_channels=4, out_channels=1).to(DEVICE)
    #Creates an instance of UNet model
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #Creates an Adam optimizer 

    start_epoch = 1
    best_score = -1.0
    #To ensure that any valid validation loss (which will always be greater than or equal to 0) will immediately replace this initial value. 
    best_val_dice = 0.0
    best_val_precision = 0.0

    # =========================
    # RESUME FROM CHECKPOINT
    # =========================
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])
        #Loads the model's weights from the checkpoint into your model object.
        #The keys in the dictionary ("model_state_dict") are used to identify the correct parameters.
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        #Loads the optimizer’s state (learning rate, momentum, etc.) from the checkpoint.

        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint.get("best_val_score", -1.0)
        #Retrieves the best validation score saved in the checkpoint.
        #-1 is to still have a sensible initial value for comparison
        best_val_dice = checkpoint.get("best_val_dice", 0.0)
        best_val_precision = checkpoint.get("best_val_precision", 0.0)

        print(f"Resumed from checkpoint: {CHECKPOINT_PATH}")
        print(f"Starting from epoch {start_epoch}")
        print(f"Best saved score so far: {best_score:.4f}")

    elif os.path.exists(BEST_MODEL_PATH):
        #used to store the model with the highest validation score during training
        best_checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        best_score = best_checkpoint.get("best_val_score", -1.0)
        best_val_dice = best_checkpoint.get("best_val_dice", 0.0)
        best_val_precision = best_checkpoint.get("best_val_precision", 0.0)

        print(f"Loaded best model weights from: {BEST_MODEL_PATH}")
        print("Training will start from epoch 1 because optimizer state was not restored from checkpoint.")

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_loss, train_dice, train_precision = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, epoch, NUM_EPOCHS
        )

        val_loss, val_dice, val_precision = evaluate(
            model, val_loader, criterion, DEVICE, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]"
        )

        val_score = (val_dice + val_precision) / 2.0

        print(
            f"\nEpoch [{epoch}/{NUM_EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Dice: {train_dice:.4f} "
            f"| Train Precision: {train_precision:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Dice: {val_dice:.4f} "
            f"| Val Precision: {val_precision:.4f} "
            f"| Val Score: {val_score:.4f}"
        )

        # save best model
        if val_score > best_score:
            best_score = val_score
            best_val_dice = val_dice
            best_val_precision = val_precision

            torch.save(
                #Saves the model's state (model weights, best Dice, best Precision, and overall score) to the BEST_MODEL_PATH
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_dice": best_val_dice,
                    "best_val_precision": best_val_precision,
                    "best_val_score": best_score,
                    "config": {
                        "batch_size": BATCH_SIZE,
                        "num_epochs": NUM_EPOCHS,
                        "lr": LR,
                        "val_ratio": VAL_RATIO,
                        "test_ratio": TEST_RATIO,
                        "empty_slice_ratio": EMPTY_SLICE_RATIO,
                        "seed": SEED,
                    },
                },
                BEST_MODEL_PATH
            )
            print(f"Saved best model to: {BEST_MODEL_PATH}")

        # save checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "best_val_precision": best_val_precision,
                "best_val_score": best_score,
                "config": {
                    "batch_size": BATCH_SIZE,
                    "num_epochs": NUM_EPOCHS,
                    "lr": LR,
                    "val_ratio": VAL_RATIO,
                    "test_ratio": TEST_RATIO,
                    "empty_slice_ratio": EMPTY_SLICE_RATIO,
                    "seed": SEED,
                },
            },
            CHECKPOINT_PATH
        )
        print(f"Checkpoint saved to: {CHECKPOINT_PATH}")

    print("\nTraining finished.")
    print(f"Best validation score: {best_score:.4f}")
    print(f"Best validation dice: {best_val_dice:.4f}")
    print(f"Best validation precision: {best_val_precision:.4f}")

    print("\nEvaluating final model on test set...")
    test_loss, test_dice, test_precision = evaluate(
        model, test_loader, criterion, DEVICE, desc="Testing"
    )
    test_score = (test_dice + test_precision) / 2.0

    print("\nFinal test results with last epoch model:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Dice: {test_dice:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Score: {test_score:.4f}")
    # It handles the final evaluation on the test set.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "test_dice": test_dice,
            "test_precision": test_precision,
            "test_score": test_score,
        },
        FINAL_MODEL_PATH
    )
    print(f"Final model saved to: {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
    
#lr
#best_unet