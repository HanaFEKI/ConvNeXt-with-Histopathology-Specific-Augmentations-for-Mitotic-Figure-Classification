import numpy as np
from pathlib import Path
from typing import List, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset

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
