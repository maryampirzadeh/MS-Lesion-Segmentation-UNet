import os
import re
import glob
import random
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

from model import UNet


# =========================
# CONFIG
# =========================
DATA_DIR = r"E:\training"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15
EMPTY_SLICE_RATIO = 0.25

WEIGHT_PATH = r"weights\best_unet_dice_precision.pth"
SAVE_DIR = "visualizations"
NUM_SAMPLES_TO_SAVE = 20
THRESHOLD = 0.5

os.makedirs(SAVE_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================
# DATA HELPERS
# =========================
def load_nifti(path):
    return nib.load(path).get_fdata().astype(np.float32)


def normalize(img):
    return (img - np.mean(img)) / (np.std(img) + 1e-8)


def merge_masks(mask_paths):
    masks = [load_nifti(p) for p in sorted(mask_paths)]
    merged = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        merged |= (m > 0)
    return merged.astype(np.float32)


def get_patient_scans(patient_dir, image_folder="preprocessed"):
    img_dir = os.path.join(patient_dir, image_folder)
    mask_dir = os.path.join(patient_dir, "masks")

    img_files = glob.glob(os.path.join(img_dir, "*.nii"))
    mask_files = glob.glob(os.path.join(mask_dir, "*.nii"))

    scans = {}

    for f in img_files:
        name = os.path.basename(f).replace(".nii", "")
        m = re.match(r"(training\d+_\d+)_(flair|mprage|pd|t2).*", name)
        if m:
            scan_id, modality = m.groups()
            scans.setdefault(scan_id, {})
            scans[scan_id][modality] = f

    for f in mask_files:
        name = os.path.basename(f)
        m = re.match(r"(training\d+_\d+)_mask\d+\.nii", name)
        if m:
            scan_id = m.group(1)
            scans.setdefault(scan_id, {})
            scans[scan_id].setdefault("masks", [])
            scans[scan_id]["masks"].append(f)

    return scans


def build_sample_index(data_dir):
    patient_dirs = sorted(glob.glob(os.path.join(data_dir, "training*")))
    positive_samples = []
    empty_samples = []

    for patient_dir in patient_dirs:
        scans = get_patient_scans(patient_dir, image_folder="preprocessed")

        for scan_id, info in sorted(scans.items()):
            needed = ["flair", "t2", "pd", "mprage", "masks"]
            if not all(k in info for k in needed):
                continue

            flair = normalize(load_nifti(info["flair"]))
            t2 = normalize(load_nifti(info["t2"]))
            pd = normalize(load_nifti(info["pd"]))
            mprage = normalize(load_nifti(info["mprage"]))
            mask = merge_masks(info["masks"])

            if not (flair.shape == t2.shape == pd.shape == mprage.shape == mask.shape):
                continue

            for z in range(mask.shape[2]):
                m = mask[:, :, z]

                sample = {
                    "scan_id": scan_id,
                    "z": z,
                    "flair": info["flair"],
                    "t2": info["t2"],
                    "pd": info["pd"],
                    "mprage": info["mprage"],
                    "mask_paths": sorted(info["masks"]),
                }

                if np.sum(m) > 0:
                    positive_samples.append(sample)
                else:
                    empty_samples.append(sample)

    n_empty_keep = int(len(positive_samples) * EMPTY_SLICE_RATIO)
    if len(empty_samples) > 0:
        empty_samples = random.sample(empty_samples, min(n_empty_keep, len(empty_samples)))
    else:
        empty_samples = []

    all_samples = positive_samples + empty_samples
    random.shuffle(all_samples)
    return all_samples


def split_by_scan(samples, val_ratio=0.15, test_ratio=0.15):
    scan_ids = sorted(list(set(s["scan_id"] for s in samples)))
    random.shuffle(scan_ids)

    n_total = len(scan_ids)
    n_test = max(1, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))

    test_scans = set(scan_ids[:n_test])
    val_scans = set(scan_ids[n_test:n_test + n_val])
    train_scans = set(scan_ids[n_test + n_val:])

    train_samples = [s for s in samples if s["scan_id"] in train_scans]
    val_samples = [s for s in samples if s["scan_id"] in val_scans]
    test_samples = [s for s in samples if s["scan_id"] in test_scans]

    return train_samples, val_samples, test_samples


# =========================
# DATASET
# =========================
class MSDataset2D(Dataset):
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
        mask = np.expand_dims(mask.astype(np.float32), axis=0)

        return torch.tensor(image), torch.tensor(mask), s["scan_id"], z


# =========================
# METRICS
# =========================
def dice_score_binary(pred, target, smooth=1e-6):
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return float(dice)


def precision_score_binary(pred, target, smooth=1e-6):
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()

    precision = (tp + smooth) / (tp + fp + smooth)
    return float(precision)


# =========================
# SAVE FIGURE
# =========================
def save_prediction_figure(flair, gt, pred, scan_id, z, save_path, dice_value, precision_value):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(flair, cmap="gray")
    plt.title("FLAIR")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(flair, cmap="gray")
    plt.imshow(pred, alpha=0.4, cmap="Reds")
    plt.title(f"Overlay\nDice={dice_value:.4f} | Prec={precision_value:.4f}")
    plt.axis("off")

    plt.suptitle(f"{scan_id} | slice {z}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# MAIN
# =========================
def main():
    print("Device:", DEVICE)

    samples = build_sample_index(DATA_DIR)
    _, _, test_samples = split_by_scan(samples, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO)

    print("Total test samples:", len(test_samples))

    test_dataset = MSDataset2D(test_samples)

    model = UNet(in_channels=4, out_channels=1).to(DEVICE)

    checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded weights from:", WEIGHT_PATH)

    n_to_save = min(NUM_SAMPLES_TO_SAVE, len(test_dataset))
    selected_indices = random.sample(range(len(test_dataset)), n_to_save)

    all_dices = []
    all_precisions = []

    for i, idx in enumerate(selected_indices, start=1):
        image, mask, scan_id, z = test_dataset[idx]

        image_in = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(image_in)
            probs = torch.sigmoid(logits)
            pred = (probs > THRESHOLD).float().cpu().squeeze().numpy()

        flair = image[0].numpy()
        gt = mask.squeeze().numpy()

        dice_value = dice_score_binary(pred, gt)
        precision_value = precision_score_binary(pred, gt)

        all_dices.append(dice_value)
        all_precisions.append(precision_value)

        filename = f"{i:02d}_{scan_id}_slice_{z}_dice_{dice_value:.4f}_prec_{precision_value:.4f}.png"
        save_path = os.path.join(SAVE_DIR, filename)

        save_prediction_figure(
            flair=flair,
            gt=gt,
            pred=pred,
            scan_id=scan_id,
            z=z,
            save_path=save_path,
            dice_value=dice_value,
            precision_value=precision_value
        )

        print(f"Saved: {save_path}")

    print("\nSaved visualization images to:", SAVE_DIR)
    print("Average Dice of saved samples:", round(float(np.mean(all_dices)), 4))
    print("Average Precision of saved samples:", round(float(np.mean(all_precisions)), 4))


if __name__ == "__main__":
    main()
    
