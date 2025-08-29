import logging
from pathlib import Path
from typing import List, Union, Tuple

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

# Local imports
from models.convnext import ConvNextV2Classifier
from utils.augmentations import ElasticTransformHistopath


# -----------------------------
# Dataset
# -----------------------------
class ClassificationDataset(Dataset):
    def __init__(self, images: List[Union[str, Path]], labels: List[int], transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.images[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label


# -----------------------------
# Trainer
# -----------------------------
class MitosisTrainer:
    """Trainer for mitotic figure classification with ConvNeXt"""

    def __init__(
        self,
        model_name: str,
        weights: Union[str, None],
        experiment_dir: str,
        num_epochs: int = 20,
        batch_size: int = 128,
        lr: float = 1e-4,
        num_folds: int = 5,
    ):
        self.model_name = model_name
        self.weights = weights
        self.experiment_dir = experiment_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_folds = num_folds
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCEWithLogitsLoss()
        self.classes = ["Atypical", "Normal"]

    # -----------------------------
    # Augmentations
    # -----------------------------
    @property
    def train_transform(self):
        return A.Compose([
            A.CenterCrop(60, 60),

            A.OneOf([
                A.D4(p=1.0),
                A.Rotate(limit=180, p=1.0, border_mode=cv2.BORDER_REFLECT_101),
                A.RandomRotate90(p=1.0),
            ], p=0.9),

            A.OneOf([
                A.ShiftScaleRotate(
                    shift_limit=0.08,
                    scale_limit=0.15,
                    rotate_limit=30,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.8
                ),
                ElasticTransformHistopath(alpha=40, sigma=4, alpha_affine=8, p=0.7),
                A.GridDistortion(num_steps=5, distort_limit=0.2,
                                 border_mode=cv2.BORDER_REFLECT_101, p=0.6),
                A.OpticalDistortion(distort_limit=0.15,
                                    border_mode=cv2.BORDER_REFLECT_101, p=0.5)
            ], p=0.6),

            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.08, p=0.8),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.8),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.4)
            ], p=0.8),

            A.OneOf([
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.6),
                A.ChannelShuffle(p=0.3),
                A.ToGray(p=0.1),
            ], p=0.4),

            A.OneOf([
                A.GaussianBlur(blur_limit=(1, 5), sigma_limit=0, p=0.5),
                A.Defocus(radius=(1, 4), alias_blur=(0.1, 0.3), p=0.4),
                A.MotionBlur(blur_limit=5, allow_shifted=True, p=0.3),
            ], p=0.5),

            A.OneOf([
                A.GaussNoise(mean=0, per_channel=True, p=0.4),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.4), p=0.3),
                A.MultiplicativeNoise(multiplier=(0.95, 1.05), per_channel=True, p=0.2),
            ], p=0.4),

            A.Resize(224, 224, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    @property
    def val_transform(self):
        return A.Compose([
            A.CenterCrop(60, 60),
            A.Resize(224, 224, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    # -----------------------------
    # Logging & Experiment setup
    # -----------------------------
    def setup_experiment(self):
        self.exp_dir = Path(self.experiment_dir)
        self.exp_dir.mkdir(exist_ok=True, parents=True)
        log_file = self.exp_dir / 'training.log'

        self.logger = logging.getLogger("MitosisLogger")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file)
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

        self.logger.info(f"Created experiment directory: {str(self.exp_dir)}")

    # -----------------------------
    # DataLoader preparation
    # -----------------------------
    def prepare_data_loaders(self, images, labels, transform, is_training=True):
        dataset = ClassificationDataset(images, labels, transform=transform)

        if is_training:
            class_counts = [labels.count(0), labels.count(1)]
            class_weights = [1.0 / max(c, 1) for c in class_counts]
            sample_weights = [class_weights[lbl] for lbl in labels]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

            loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler,
                                num_workers=8, pin_memory=True)
        else:
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=8, pin_memory=True)
        return loader

    # -----------------------------
    # Training loop
    # -----------------------------
    def train_epoch(self, model, train_loader, optimizer):
        model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for images_batch, labels_batch in tqdm(train_loader, desc="Training"):
            images_batch = images_batch.to(self.device)
            labels_batch = labels_batch.to(self.device)

            optimizer.zero_grad()
            logits, Y_prob, Y_hat = model(images_batch)
            loss = self.criterion(logits, labels_batch.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(Y_hat.cpu().numpy())
            all_targets.extend(labels_batch.cpu().numpy())

        return total_loss / len(train_loader), all_preds, all_targets

    def evaluate(self, model, data_loader, phase="Validation"):
        model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for images_batch, labels_batch in tqdm(data_loader, desc=phase):
                images_batch = images_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)

                logits, Y_prob, Y_hat = model(images_batch)
                loss = self.criterion(logits, labels_batch.float())

                total_loss += loss.item()
                all_preds.extend(Y_hat.cpu().numpy())
                all_targets.extend(labels_batch.cpu().numpy())

        return total_loss / len(data_loader), all_preds, all_targets

    # -----------------------------
    # Training one fold
    # -----------------------------
    def train_fold(self, fold: int, train_images, train_labels, val_images, val_labels):
        train_loader = self.prepare_data_loaders(train_images, train_labels, self.train_transform, is_training=True)
        val_loader = self.prepare_data_loaders(val_images, val_labels, self.val_transform, is_training=False)

        model = ConvNextV2Classifier(self.model_name, self.weights).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-7)

        best_val_acc = 0.0
        best_model_path = self.exp_dir / f"ConvnextV2_fold{fold + 1}_best.pth"

        for epoch in range(self.num_epochs):
            train_loss, train_preds, train_targets = self.train_epoch(model, train_loader, optimizer)
            train_acc = balanced_accuracy_score(train_targets, train_preds)

            val_loss, val_preds, val_targets = self.evaluate(model, val_loader)
            val_acc = balanced_accuracy_score(val_targets, val_preds)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)

            scheduler.step()

            self.logger.info(
                f"Fold {fold + 1}, Epoch {epoch + 1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Balanced Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Balanced Acc: {val_acc:.4f}"
            )

        return best_val_acc, best_model_path

    # -----------------------------
    # Cross-validation training
    # -----------------------------
    def train(self, train_images, train_labels):
        self.setup_experiment()

        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold_accuracies, best_model_paths = [], []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_images, train_labels)):
            self.logger.info(f"--- Starting Fold {fold + 1} ---")

            fold_train_images = [train_images[i] for i in train_idx]
            fold_train_labels = [train_labels[i] for i in train_idx]
            fold_val_images = [train_images[i] for i in val_idx]
            fold_val_labels = [train_labels[i] for i in val_idx]

            best_acc, best_path = self.train_fold(fold, fold_train_images, fold_train_labels, fold_val_images, fold_val_labels)

            fold_accuracies.append(best_acc)
            best_model_paths.append(best_path)

            self.logger.info(f"Fold {fold + 1} - Best Validation Balanced Accuracy: {best_acc:.4f}")

        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        self.logger.info(f"Average Balanced Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

        return fold_accuracies
