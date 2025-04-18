import json
import os
import numpy as np
from typing import List, Tuple, Dict, Union
from PIL import Image

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
            -> List[Dict[str, Union[str, dict]]]:
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

    def generate_masks(self) -> None:
        pass

    def combine_masks(self) -> None:
        pass

    def generate(self) -> None:
        pass
