import os
from typing import Tuple, List
import cv2
import numpy as np


def load_images(img_dir: str, supported_formats: Tuple[str] = (".jpg", ".jpeg", ".png")) \
        -> List[Tuple[np.ndarray, str]]:
    images_files = [f for f in os.listdir(img_dir) if f.endswith(supported_formats)]
    images = []

    for image_file in images_files:
        img_path = os.path.join(img_dir, image_file)
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append((img, image_file))
        except FileNotFoundError and FileExistsError as e:
            print(f"Error opening image: {img_path}\nError: {e}")
            continue
    return images
