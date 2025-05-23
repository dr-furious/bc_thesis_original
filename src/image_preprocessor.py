import math
import os
import random
import re
import shutil
from enum import Enum
from typing import List, Tuple, Dict, Union
import cv2
import numpy as np
from sklearn.model_selection import GroupKFold
from torchvision import transforms
import torchstain

from src.utils import load_images


# Compute stride so the patches do not contain any black pixels that are outside the image
def _compute_stride(dimension: int, patch_size: int) -> int:
    if dimension <= patch_size:
        return patch_size
    num_patches = math.ceil((dimension - patch_size) / patch_size) + 1
    stride = (dimension - patch_size) / (num_patches - 1)
    return int(stride)


# Splits the images and masks into k-folds, where the pattern is used to identify groups from image name
def k_split(source_dir: str, out_dir: str, k: int = 3,
            images_dir: str = "images", masks_dir: str = "masks", pattern=r"(\d+)_") \
        -> None:
    image_dir = os.path.join(source_dir, images_dir)
    image_paths = sorted(
        [os.path.join(image_dir, img_name) for img_name in os.listdir(image_dir) if img_name.endswith('.png')])

    case_ids = [re.match(pattern, os.path.basename(path)).group(1) for path in image_paths]
    groups = [int(pid) for pid in case_ids]

    gkf = GroupKFold(n_splits=k)
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(image_paths, groups=groups)):
        # Create directories for this fold
        fold_dir = os.path.join(out_dir, f"fold_{fold_idx}")
        os.makedirs(os.path.join(fold_dir, images_dir), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, masks_dir), exist_ok=True)

        # Copy validation images and corresponding masks to the fold directory
        for idx in val_idx:
            image_path = image_paths[idx]
            mask_path = image_path.replace(images_dir, masks_dir)
            shutil.copy(image_path, os.path.join(fold_dir, images_dir, os.path.basename(image_path)))
            shutil.copy(mask_path, os.path.join(fold_dir, masks_dir, os.path.basename(mask_path)))


class InplaceOption(Enum):
    NORM = "norm"
    HEMATOXYLIN = "hematoxylin"
    EOSIN = "eosin"


class ImageProcessor:
    images_dir: str | None
    images: List[Tuple[np.ndarray, str]]

    def __init__(self, images_dir: str, supported_formats: Tuple[str] = (".jpg", ".jpeg", ".png"),
                 images: List[Tuple[np.ndarray, str]] = None) -> None:
        if images is not None:
            self.images_dir = None
            self.images = images
            return
        self.images_dir = images_dir
        self.images = load_images(images_dir, supported_formats)

    # Sets the images from the provided images_dir
    def set_images_from_dir(self, images_dir: str, supported_formats: Tuple[str] = (".jpg", ".jpeg", ".png")) -> None:
        self.images_dir = images_dir
        self.images = load_images(images_dir, supported_formats)

    # Selects random images of size num
    def select_random_images(self, num: int = 10) -> List[Tuple[np.ndarray, str]]:
        if num > len(self.images):
            print("The number of images to select is greater than the total number of images.")
        return random.sample(self.images, num)

    # Performs histogram equalization on the images
    def equalize_hist(self, inplace: bool = False,
                      out_dir: str | None = None) \
            -> List[Dict[str, Union[str, np.ndarray]]]:

        result = []
        eq_images = []
        for image in self.images:
            img, name = image
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
            equalized_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

            if out_dir is not None:
                cv2.imwrite(os.path.join(out_dir, name), equalized_img)

            if inplace is True:
                eq_images.append((equalized_img, name))

            result.append({
                "img_name": name,
                "equalized_img": equalized_img
            })
        if inplace is True:
            self.images = eq_images
        return result

    # Normalizes the images and separates hematoxylin and eosin images from the original image
    def normalize(self, target_images: List[Tuple[np.ndarray, str]],
                  inplace: bool = False,
                  inplace_option: InplaceOption = InplaceOption.NORM,
                  out_dir: str | None = None,
                  norm_dir_name: str = "normalized",
                  hematoxylin_dir_name: str = "hematoxylin",
                  eosin_dir_name: str = "eosin") -> List[Dict[str, Union[str, np.ndarray]]]:

        normalized_dir = None
        hematoxylin_dir = None
        eosin_dir = None
        if out_dir is not None:
            # Make dirs for norm, hematoxylin and eosin
            normalized_dir = os.path.join(out_dir, norm_dir_name)
            hematoxylin_dir = os.path.join(out_dir, hematoxylin_dir_name)
            eosin_dir = os.path.join(out_dir, eosin_dir_name)

            os.makedirs(normalized_dir, exist_ok=True)
            os.makedirs(hematoxylin_dir, exist_ok=True)
            os.makedirs(eosin_dir, exist_ok=True)

        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)
        ])

        # Images to tensors
        target_images = [T(cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)) for img in target_images]

        # Prepare normalizer
        normalizer = torchstain.normalizers.MultiMacenkoNormalizer(backend="torch")
        normalizer.fit(target_images)

        # Apply normalization
        result = []
        norm_images = []
        for image in self.images:
            img, name = image
            img = T(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            norm_img_tensor, hematoxylin, eosin = normalizer.normalize(I=img, stains=True)

            # Pull the images into cpu and convert them to numpy array as values between 0 and 255 and as np.uint8 type
            norm_image = np.clip(norm_img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
            hematoxylin_img = np.clip(hematoxylin.cpu().numpy(), 0, 255).astype(np.uint8)
            eosin_img = np.clip(eosin.cpu().numpy(), 0, 255).astype(np.uint8)

            norm_image = cv2.cvtColor(norm_image, cv2.COLOR_RGB2BGR)
            hematoxylin_img = cv2.cvtColor(hematoxylin_img, cv2.COLOR_RGB2BGR)
            eosin_img = cv2.cvtColor(eosin_img, cv2.COLOR_RGB2BGR)

            if out_dir is not None:
                cv2.imwrite(os.path.join(normalized_dir, name), norm_image)
                cv2.imwrite(os.path.join(hematoxylin_dir, name), hematoxylin_img)
                cv2.imwrite(os.path.join(eosin_dir, name), eosin_img)

            if inplace is True:
                if inplace_option == InplaceOption.NORM:
                    norm_images.append((norm_image, name))
                elif inplace_option == InplaceOption.HEMATOXYLIN:
                    norm_images.append((hematoxylin_img, name))
                elif inplace_option == InplaceOption.EOSIN:
                    norm_images.append((eosin_img, name))

            result.append({
                "img_name": name,
                "norm_image": norm_image,
                "hematoxylin_img": hematoxylin_img,
                "eosin_img": eosin_img
            })

        if inplace is True:
            self.images = norm_images
        return result

    # Extracts the patches from the image and all it's provided masks
    def extract_patches(self, masks_dirs: List[str], out_dir: str, patch_size=128, pad: bool = True,
                        images_dir: str = "images", masks_dir_name: str = "masks") -> None:
        for image in self.images:
            img, name = image
            name = os.path.splitext(os.path.basename(name))[0]
            h, w = img.shape[:2]

            # Compute dynamic stride
            stride_h = _compute_stride(h, patch_size)
            stride_w = _compute_stride(w, patch_size)

            # Compute padding
            pad_h = (patch_size - h % stride_h) % stride_h
            pad_w = (patch_size - w % stride_w) % stride_w

            if pad:
                img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            padded_h, padded_w = img.shape[:2]
            patch_coords = []
            for y in range(0, padded_h - patch_size + 1, stride_h):
                for x in range(0, padded_w - patch_size + 1, stride_w):
                    if y + patch_size > h or x + patch_size > w:
                        continue  # Skip patch that crosses original image border
                    patch_coords.append((y, x))

            # Save image patches
            for y, x in patch_coords:
                img_patch = img[y:y + patch_size, x:x + patch_size]
                patch_name = f"{name}_({y}_{x}).png"
                img_out_path = os.path.join(out_dir, images_dir, patch_name)
                os.makedirs(os.path.dirname(img_out_path), exist_ok=True)
                cv2.imwrite(img_out_path, img_patch)

            # Save corresponding mask patches for each mask dir
            for masks_dir in masks_dirs:
                mask_path = os.path.join(masks_dir, f"{name}.png")
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if pad:
                    mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,))

                mask_subdir = os.path.basename(masks_dir)

                for y, x in patch_coords:
                    mask_patch = mask[y:y + patch_size, x:x + patch_size]
                    patch_name = f"{name}_({y}_{x}).png"
                    mask_out_path = os.path.join(out_dir, masks_dir_name, mask_subdir, patch_name)
                    if len(masks_dirs) == 1:
                        mask_out_path = os.path.join(out_dir, masks_dir_name, patch_name)
                    os.makedirs(os.path.dirname(mask_out_path), exist_ok=True)
                    cv2.imwrite(mask_out_path, mask_patch)

    # Relabels a multi-label mask to binary mask.
    # The target class is kept as the foreground class, rest are set to background
    def relabel_binary(self, masks_dir: str, out_dir: str, target_class: int) -> None:
        for image in self.images:
            img, name = image
            mask_path = os.path.join(masks_dir, name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            mask[mask != target_class] = 0
            mask[mask == target_class] = 1

            cv2.imwrite(os.path.join(out_dir, name), mask)

    # Scales the image and all it's masks by a factor
    def scale(self, masks_dirs: List[str], out_dir: str, factor: float = 0.5, inplace: bool = False,
              images_dir: str = "images", masks_dir_name: str = "masks") -> None:
        for i, image in enumerate(self.images):
            img, name = image
            img_scaled = cv2.resize(img, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
            img_out_path = os.path.join(out_dir, images_dir, name)
            os.makedirs(os.path.dirname(img_out_path), exist_ok=True)
            cv2.imwrite(img_out_path, img_scaled)
            if inplace:
                self.images[i] = (img_scaled, name)

            for masks_dir in masks_dirs:
                mask_path = os.path.join(masks_dir, name)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask_scaled = cv2.resize(mask, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
                mask_out_path = os.path.join(out_dir, masks_dir_name, os.path.basename(masks_dir), name)
                if len(masks_dirs) == 1:
                    mask_out_path = os.path.join(out_dir, masks_dir_name, name)
                os.makedirs(os.path.dirname(mask_out_path), exist_ok=True)
                cv2.imwrite(mask_out_path, mask_scaled)
