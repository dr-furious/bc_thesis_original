import math
import os
import random
from enum import Enum
from typing import List, Tuple, Dict, Union
import cv2
import numpy as np
from torchvision import transforms
import torchstain

from src.utils import load_images


def _compute_stride(dimension, patch_size):
    if dimension <= patch_size:
        return patch_size
    num_patches = math.ceil((dimension - patch_size) / patch_size) + 1
    stride = (dimension - patch_size) / (num_patches - 1)
    return int(stride)


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

    def set_images_from_dir(self,  images_dir: str, supported_formats: Tuple[str] = (".jpg", ".jpeg", ".png")) -> None:
        self.images_dir = images_dir
        self.images = load_images(images_dir, supported_formats)

    def select_random_images(self, num: int = 10) -> List[Tuple[np.ndarray, str]]:
        if num > len(self.images):
            print("The number of images to select is greater than the total number of images.")
        return random.sample(self.images, num)

    def equalize_hist(self, inplace: bool = False, out_dir: str | None = None) \
            -> List[Dict[str, Union[str, np.ndarray]]]:

        result = []
        eq_images = []
        for image in self.images:
            img, name = image
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
            equalized_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

            if out_dir is not None:
                cv2.imwrite(os.path.join(out_dir, name), equalized_img)

            if inplace is True:
                eq_images.append((cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB), name))

            result.append({
                "img_name": name,
                "equalized_img": equalized_img
            })
        if inplace is True:
            self.images = eq_images
        return result

    def normalize(self, target_images: List[Tuple[np.ndarray, str]],
                  inplace: bool = False,
                  inplace_option: InplaceOption = InplaceOption.NORM,
                  out_dir: str | None = None) -> List[Dict[str, Union[str, np.ndarray]]]:

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
                    norm_images.append((cv2.cvtColor(norm_image, cv2.COLOR_BGR2RGB), name))
                elif inplace_option == InplaceOption.HEMATOXYLIN:
                    norm_images.append((cv2.cvtColor(hematoxylin_img, cv2.COLOR_BGR2RGB), name))
                elif inplace_option == InplaceOption.EOSIN:
                    norm_images.append((cv2.cvtColor(eosin_img, cv2.COLOR_BGR2RGB), name))

            result.append({
                "img_name": name,
                "norm_image": norm_image,
                "hematoxylin_img": hematoxylin_img,
                "eosin_img": eosin_img
            })

        if inplace is True:
            self.images = norm_images
        return result

    def extract_patches(self, masks_dirs: List[str], out_dir: str, patch_size=128, pad=True) -> None:
        for image in self.images:
            img, name = image
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
                img_out_path = os.path.join(out_dir, "images", patch_name)
                os.makedirs(os.path.dirname(img_out_path), exist_ok=True)
                cv2.imwrite(img_out_path, img_patch)

            # Save corresponding mask patches for each mask dir
            for masks_dir in masks_dirs:
                mask_path = os.path.join(masks_dir, name)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if pad:
                    mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,))

                mask_subdir = os.path.basename(masks_dir)

                for y, x in patch_coords:
                    mask_patch = mask[y:y + patch_size, x:x + patch_size]
                    patch_name = f"{name}_({y}_{x}).png"
                    mask_out_path = os.path.join(out_dir, "masks", mask_subdir, patch_name)
                    if len(masks_dirs) == 1:
                        mask_out_path = os.path.join(out_dir, "masks", patch_name)
                    os.makedirs(os.path.dirname(mask_out_path), exist_ok=True)
                    cv2.imwrite(mask_out_path, mask_patch)

    def relabel_binary(self, masks_dir: str, out_dir: str, target_class: int) -> None:
        for image in self.images:
            img, name = image
            mask_path = os.path.join(masks_dir, name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            mask[mask != target_class] = 0
            mask[mask == target_class] = 1

            cv2.imwrite(os.path.join(out_dir, name), mask)

    def scale(self, masks_dirs: List[str], out_dir: str, factor: float = 0.5, inplace: bool = False) -> None:
        for i, image in enumerate(self.images):
            img, name = image
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_scaled = cv2.resize(img, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
            img_out_path = os.path.join(out_dir, "images", name)
            os.makedirs(os.path.dirname(img_out_path), exist_ok=True)
            cv2.imwrite(img_out_path, img_scaled)
            if inplace:
                self.images[i] = (cv2.cvtColor(img_scaled, cv2.COLOR_BGR2RGB), name)

            for masks_dir in masks_dirs:
                mask_path = os.path.join(masks_dir, name)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask_scaled = cv2.resize(mask, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
                mask_out_path = os.path.join(out_dir, "masks", os.path.basename(masks_dir), name)
                if len(masks_dirs) == 1:
                    mask_out_path = os.path.join(out_dir, "masks", name)
                os.makedirs(os.path.dirname(mask_out_path), exist_ok=True)
                cv2.imwrite(mask_out_path, mask_scaled)





