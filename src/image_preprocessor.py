import os
import random
from typing import List, Tuple, Dict, Union

import cv2
import numpy as np
from torchvision import transforms
import torchstain

from src.utils import load_images


class ImageProcessor:
    images_dir: str | None
    images: List[Tuple[np.ndarray, str]]

    def __init__(self, images_dir, supported_formats: Tuple[str] = (".jpg", ".jpeg", ".png"),
                 images: List[np.ndarray] = None) -> None:
        if images is not None:
            self.images_dir = None
            self.images = [(img, "") for img in images]
            return
        self.images_dir = images_dir
        self.images = load_images(images_dir, supported_formats)

    def select_random_images(self, num: int = 10) -> List[Tuple[np.ndarray, str]]:
        if num > len(self.images):
            print("The number of images to select is greater than the total number of images.")
        return random.sample(self.images, num)

    def normalize(self, target_images: List[Tuple[np.ndarray, str]], inplace: bool = False, out_dir: str | None = None)\
            -> List[Dict[str, Union[str, np.ndarray]]]:

        normalized_dir = None
        hematoxylin_dir = None
        eosin_dir = None
        if out_dir is not None:
            # Make dirs for norm, hematoxylin and eosin
            normalized_dir = os.path.join(out_dir, "normalized")
            hematoxylin_dir = os.path.join(out_dir, "hematoxylin")
            eosin_dir = os.path.join(out_dir, "eosin")

            os.makedirs(normalized_dir, exist_ok=True)
            os.makedirs(hematoxylin_dir, exist_ok=True)
            os.makedirs(eosin_dir, exist_ok=True)

        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)
        ])

        # Images to tensors
        target_images = [T(img[0]) for img in target_images]

        # Prepare normalizer
        normalizer = torchstain.normalizers.MultiMacenkoNormalizer(backend="torch")
        normalizer.fit(target_images)

        # Apply normalization
        result = []
        norm_images = []
        for image in self.images:
            img, name = image
            img = T(img)
            norm_img_tensor, hematoxylin, eosin = normalizer.normalize(I=img, stains=True)

            norm_image = norm_img_tensor.cpu().numpy()
            hematoxylin_img = hematoxylin.cpu().numpy()
            eosin_img = eosin.cpu().numpy()

            if out_dir is not None:
                cv2.imwrite(os.path.join(normalized_dir, name), norm_image)
                cv2.imwrite(os.path.join(hematoxylin_dir, name), hematoxylin_img)
                cv2.imwrite(os.path.join(eosin_dir, name), eosin_img)

            if inplace is True:
                norm_images.append((norm_image, name))

            result.append({
                "img_name": name,
                "norm_image": norm_image,
                "hematoxylin_img": hematoxylin_img,
                "eosin_img": eosin_img
            })

        if inplace is True:
            self.images = norm_images
        return result





