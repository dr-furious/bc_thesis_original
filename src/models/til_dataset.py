import os
from typing import List
import cv2
import torch
from torch.utils.data import Dataset


class TILDataset(Dataset):
    images_dir: str
    masks_dir: str
    image_ids: List[str]

    def __init__(self, images_dir: str, masks_dir: str):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_ids = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, image_id)
        mask_path = os.path.join(self.masks_dir, image_id)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask
