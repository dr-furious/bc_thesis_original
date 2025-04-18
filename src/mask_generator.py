import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Union, Callable
from PIL import Image
from matplotlib import pyplot as plt

from src.utils import load_images, load_json


# Mask Generating Workflow:
# Large PNGs --> Cut cells --> Generate cell masks --> Combine into large masks


def _cut(image_path: str, regions: List[List[int]], out_dir: str | None = None) \
        -> List[Dict[str, Union[np.ndarray, int]]]:
    image = Image.open(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    cut_roi_images = []

    for (x, y, width, height) in regions:
        # Crop each ROI from image
        roi = image.crop((x, y, x + width, y+height))
        cut_roi_images.append({
            "image": np.array(roi),
            "x": x,
            "y": y,
            "width": width,
            "height": height
        })

        # Store each ROI into desired directory
        if out_dir is not None:
            roi.save(os.path.join(out_dir, f"{image_name}_-_{x}_{y}_{width}_{height}.png"))

    return cut_roi_images


def _check_bbox(roi: np.ndarray, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]) \
        -> Tuple[int, int, int, int, np.ndarray]:
    x, y, w, h = bbox
    print(f"x, y, w, h: {x, y, w, h}")
    if x < 0:
        roi = roi[:, abs(x):]
        w = w + x
        x = 0
    if y < 0:
        roi = roi[abs(y):, :]
        h = h + y
        y = 0

    if x + w > image_shape[1]:
        roi = roi[:, :-(x + w - image_shape[1])]
        w = image_shape[1] - x
    if y + h > image_shape[0]:
        roi = roi[:-(y + h - image_shape[0]), :]
        h = image_shape[0] - y

    return x, y, w, h, roi


class MaskGenerator:
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

    def cut_cells(self, json_path: str, out_dir: str | None = None) \
            -> List[Dict[str, Union[str, List[dict]]]]:
        data = load_json(json_path)
        images = data['images']
        cut_results = []
        for image in images:
            image_id = image['id']
            image_name = image['file_name']
            image_path = os.path.join(self.images_dir, image_name)
            rois = [ann['bbox'] for ann in data['annotations'] if ann['image_id'] == image_id]
            roi_images = _cut(image_path=image_path, regions=rois, out_dir=out_dir)

            # Save all cut rois per single image
            cut_results.append({
                "image_name": image_name,
                "roi_cuts": roi_images
            })
        return cut_results

    def generate_masks(self, image_data: List[Dict[str, Union[str, List[dict]]]], operations: List[Callable]) \
            -> List[Dict[str, Union[str, List[dict]]]]:
        if operations is None:
            return image_data
        for operation in operations:
            image_data = operation(image_data)
        return image_data

    def otsu_thresholding(self, image_data: List[Dict[str, Union[str, List[dict]]]]) \
            -> List[Dict[str, Union[str, List[dict]]]]:
        # your code here
        return image_data

    def open_close(self, image_data: List[Dict[str, Union[str, List[dict]]]]) \
            -> List[Dict[str, Union[str, List[dict]]]]:
        # your code here
        return image_data

    def hist_equalize(self, image_data: List[Dict[str, Union[str, List[dict]]]]) \
            -> List[Dict[str, Union[str, List[dict]]]]:
        # your code here
        return image_data

    def combine_masks(self, image_data: List[Dict[str, Union[str, List[dict]]]], out_dir: str) \
            -> List[Tuple[np.ndarray, str]]:
        i = 20
        for record in image_data:
            image_name, roi_cuts = record['image_name'], record['roi_cuts']
            image_path = os.path.join(self.images_dir, image_name)

            print(f"ROIs: {len(roi_cuts)}")
            print(f"Image name: {image_name}")

            whole_image = np.zeros(cv2.imread(image_path).shape[:2])

            print(f"Whole image shape: {whole_image.shape}")

            for cut in roi_cuts:
                roi = np.array(cut["image"], dtype=np.uint8)
                roi = np.ones(roi.shape[:2])
                print(f"ROI shape: {roi.shape}")
                x, y, w, h, roi = _check_bbox(roi=roi,
                                              bbox=(cut["x"], cut["y"], cut["width"], cut["height"]),
                                              image_shape=whole_image.shape)
                print(f"ROI shape after: {roi.shape}")
                buffer_image = np.zeros_like(whole_image)
                print(f"Buffer image shape: {buffer_image.shape}")
                print(f"x, y, w, h: {x, y, w, h}")
                buffer_image[y:y+h, x:x+w] = roi
                whole_image = cv2.bitwise_or(whole_image, buffer_image)

            # For the purpose of visualization
            whole_image[whole_image == 1] = 255
            plt.title(image_name)
            plt.imshow(whole_image, cmap='gray')
            plt.show()
            i = i-1
            if i < 1:
                return

    def run(self, json_path: str, out_dir: str = None, operations: List[Callable] = None, save_steps: bool = False) -> None:
        # Cutting
        cut_results = self.cut_cells(json_path)

        # Applying CV operations
        cv_results = self.generate_masks(cut_results, operations)

        # Combining masks
        self.combine_masks(cv_results, out_dir=out_dir)
