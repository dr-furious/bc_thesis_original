import os

import cv2
import numpy as np
from typing import List, Tuple, Dict, Union, Callable
from PIL import Image
from matplotlib import pyplot as plt

from src.utils import load_json


# Mask Generating Workflow:
# Large PNGs --> Cut cells --> Generate cell masks --> Combine into large masks


def _cut(image_path: str, regions: List[List[int]], out_dir: str | None = None) \
        -> List[Dict[str, Union[np.ndarray, int]]]:
    image = Image.open(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    cut_roi_images = []

    for (x, y, width, height) in regions:
        # Crop each ROI from image
        roi_img = image.crop((x, y, x + width, y + height))
        roi = np.array(roi_img)
        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        cut_roi_images.append({
            "image": roi,
            "image_copy": np.array(roi),
            "x": x,
            "y": y,
            "width": width,
            "height": height
        })

        # Store each ROI into desired directory
        if out_dir is not None:
            roi_img.save(os.path.join(out_dir, f"{image_name}_-_{x}_{y}_{width}_{height}.png"))

    return cut_roi_images


def _check_bbox(roi: np.ndarray, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]) \
        -> Tuple[int, int, int, int, np.ndarray]:
    x, y, w, h = bbox
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
    image_data: List[Dict[str, Union[str, List[dict]]]]

    def __init__(self, images_dir) -> None:
        self.images_dir = images_dir

    def set_dir(self, images_dir: str) -> None:
        self.images_dir = images_dir
        self.image_data = []

    def _cut_cells(self, json_path: str, out_dir: str | None = None) \
            -> List[Dict[str, Union[str, List[dict]]]]:
        data = load_json(json_path)
        images = data["images"]
        cut_results = []
        for image in images:
            image_id = image["id"]
            image_name = image["file_name"]
            image_path = os.path.join(self.images_dir, image_name)
            rois = [ann["bbox"] for ann in data["annotations"] if ann["image_id"] == image_id]
            roi_images = _cut(image_path=image_path, regions=rois, out_dir=out_dir)

            # Save all cut rois per single image
            cut_results.append({
                "image_name": image_name,
                "roi_cuts": roi_images
            })
        self.image_data = cut_results
        return self.image_data

    def _generate_masks(self, operations: List[Callable], save: bool) \
            -> List[Dict[str, Union[str, List[dict]]]]:
        if operations is None:
            return self.image_data
        for operation in operations:
            operation(save)

        return self.image_data

    def _apply_operation_to_roi(self, operation: Callable, out_dir: str | None = None) -> None:
        for record in self.image_data:
            for cut in record["roi_cuts"]:
                roi = cut["image"]
                roi_copy = cut["image_copy"]
                roi = operation(roi, roi_copy)
                cut["image"] = roi

    def _gaussian_blur(self, save: bool) -> None:
        def do(roi: np.ndarray, roi_copy: np.ndarray) -> np.ndarray:
            return cv2.GaussianBlur(roi, (3, 3), 0)

        self._apply_operation_to_roi(do, "gaussian_blur" if save else None)

    def _otsu_thresholding(self, save: bool) -> None:
        def do(roi: np.ndarray, roi_copy: np.ndarray) -> np.ndarray:
            img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Use Otsu's thresholding
            _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return thresh

        self._apply_operation_to_roi(do, "otsu_threshold" if save else None)

    def _adaptive_thresholding(self, save: bool) -> None:
        def do(roi: np.ndarray, roi_copy: np.ndarray) -> np.ndarray:
            img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            return cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        self._apply_operation_to_roi(do, "adaptive_threshold" if save else None)

    def _morph_opening(self, save: bool) -> None:
        def do(roi: np.ndarray, roi_copy: np.ndarray) -> np.ndarray:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            return cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)

        self._apply_operation_to_roi(do, "morph_opening" if save else None)

    def _morph_closing(self, save: bool) -> None:
        def do(roi: np.ndarray, roi_copy: np.ndarray) -> np.ndarray:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            return cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

        self._apply_operation_to_roi(do, "morph_closing" if save else None)

    def _marked_watershed(self, save: bool) -> None:
        def do(roi: np.ndarray, roi_copy: np.ndarray) -> np.ndarray:
            print(f"ROI shape: {roi.shape}")
            print(f"ROI max: {roi.max()}")
            dist_transform = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            sure_bg = cv2.dilate(sure_fg, kernel, iterations=3)
            unknown = cv2.subtract(sure_bg, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(roi_copy, markers)
            result_roi = np.zeros(roi.shape, dtype=np.uint8)
            result_roi[markers > 1] = 255
            return result_roi

        self._apply_operation_to_roi(do, "marked_watershed" if save else None)

    def _combine_masks(self, save: bool) -> None:
        for record in self.image_data:
            image_name, roi_cuts = record["image_name"], record["roi_cuts"]
            image_path = os.path.join(self.images_dir, image_name)
            original_image = cv2.imread(image_path)

            print(f"ROIs: {len(roi_cuts)}")
            print(f"Image name: {image_name}")

            whole_image = np.zeros(original_image.shape[:2], dtype=np.uint8)

            print(f"Whole image shape: {whole_image.shape}")

            for cut in roi_cuts:
                roi = np.array(cut["image"], dtype=np.uint8)
                # roi = np.ones(roi.shape[:2])
                x, y, w, h, roi = _check_bbox(roi=roi,
                                              bbox=(cut["x"], cut["y"], cut["width"], cut["height"]),
                                              image_shape=whole_image.shape)
                buffer_image = np.zeros_like(whole_image, dtype=np.uint8)
                buffer_image[y:y + h, x:x + w] = roi
                whole_image = cv2.bitwise_or(whole_image, buffer_image)

            # For the purpose of visualization
            record["roi_cuts"] = [{"image": whole_image, "image_copy": original_image}]
            return

    def run(self, json_path: str, out_dir: str = None, operations: List[Callable] = None, save_steps: bool = False) \
            -> None:
        # Cutting
        self._cut_cells(json_path)

        # Applying CV operations
        operations_pipe = [self._gaussian_blur,
                           self._adaptive_thresholding,
                           self._combine_masks,
                           self._morph_opening,
                           self._morph_closing,
                           self._marked_watershed]

        self._generate_masks(operations=operations_pipe, save=save_steps)

        # Visualize results, both original image, mask, and overlay
        i = 50
        for image in self.image_data:
            plt.figure(figsize=(10, 10))

            plt.subplot(131)
            plt.imshow(cv2.cvtColor(image["roi_cuts"][0]["image_copy"], cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(132)
            plt.imshow(image["roi_cuts"][0]["image"], cmap='gray')
            plt.title("Mask")
            plt.axis('off')

            color_mask = cv2.cvtColor(image["roi_cuts"][0]["image"], cv2.COLOR_GRAY2BGR)
            # Get green color for mask
            color_mask[np.all(color_mask == [255, 255, 255], axis=-1)] = [15, 255, 0]
            overlay = cv2.addWeighted(image["roi_cuts"][0]["image_copy"], 0.7,
                                      color_mask, 0.3, 0)
            plt.subplot(133)
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title("Overlay")
            plt.axis('off')

            plt.show()

            i = i-1
            if i < 1:
                return

