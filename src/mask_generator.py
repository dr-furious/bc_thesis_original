import os
from enum import Enum
import cv2
import numpy as np
from typing import List, Tuple, Dict, Union, Callable
from PIL import Image
from matplotlib import pyplot as plt

from src.utils import load_json


# The enum of available cv operations
class MaskOperations(Enum):
    BLUR = "blur"
    ADAPTIVE_THRESHOLD = "adaptive_thresholding"
    OTSU_THRESHOLD = "otsu_thresholding"
    MORPH_OPENING = "morph_opening"
    MORPH_CLOSING = "morph_closing"
    MARKED_WATERSHED = "marked_watershed"
    COMBINE_MASKS = "combine_masks"


# Cuts a single cell roi from the image
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


# Checks if a bounding box lies partially out of the image.
# If yes it modifies it so that it could be used without error
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
    json_file_path: str
    image_data: List[Dict[str, Union[str, List[dict]]]]

    def __init__(self, images_dir: str = "", json_file_path: str = "") -> None:
        self.images_dir = images_dir
        self.json_file_path = json_file_path

    # Sets the images from a provided images_dir directory
    def set_dir(self, images_dir: str) -> None:
        self.images_dir = images_dir
        self.image_data = []

    # Cuts out single-cell rois according to the bounding box annotations
    def _cut_cells(self, json_path: str, out_dir: str | None = None) \
            -> List[Dict[str, Union[str, List[dict]]]]:
        # Load the bounding box annotations
        data = load_json(json_path)
        images = data["images"]
        cut_results = []
        for image in images:
            image_id = image["id"]
            image_name = image["file_name"]
            image_path = os.path.join(self.images_dir, image_name)
            rois = [ann["bbox"] for ann in data["annotations"] if ann["image_id"] == image_id]
            if not os.path.exists(image_path):
                continue
            # Cut a single cell roi
            roi_images = _cut(image_path=image_path, regions=rois, out_dir=out_dir)

            # Save all cut rois per single image
            cut_results.append({
                "image_name": image_name,
                "roi_cuts": roi_images
            })
        self.image_data = cut_results
        return self.image_data

    # This function maps the Enum values from MaskOperations onto the actual functions that perform them
    def _get_operation_pipeline(self, operations: List[MaskOperations]) -> List[Callable[[bool], None]]:
        op_map = {
            MaskOperations.BLUR: self._blur,
            MaskOperations.ADAPTIVE_THRESHOLD: self._adaptive_thresholding,
            MaskOperations.OTSU_THRESHOLD: self._otsu_thresholding,
            MaskOperations.MORPH_OPENING: self._morph_opening,
            MaskOperations.MORPH_CLOSING: self._morph_closing,
            MaskOperations.MARKED_WATERSHED: self._marked_watershed,
            MaskOperations.COMBINE_MASKS: self._combine_masks
        }
        return [op_map[op] for op in operations if op in op_map]

    # Sequentially applies all operations from the provided list of operations in order to create pseudo-masks
    def _generate_masks(self, operations: List[Callable]) \
            -> List[Dict[str, Union[str, List[dict]]]]:
        if operations is None:
            return self.image_data
        for operation in operations:
            operation()

        return self.image_data

    # Applies the operation to a roi
    # Since all operations operate in the same loop, to avoid redundancy we use this helper function
    def _apply_operation_to_roi(self, operation: Callable) -> None:
        for record in self.image_data:
            for cut in record["roi_cuts"]:
                roi = cut["image"]
                roi_copy = cut["image_copy"]
                roi = operation(roi, roi_copy)
                cut["image"] = roi

    def _blur(self) -> None:
        def do(roi: np.ndarray, roi_copy: np.ndarray) -> np.ndarray:
            return cv2.medianBlur(roi, 3)

        self._apply_operation_to_roi(do)

    def _otsu_thresholding(self) -> None:
        def do(roi: np.ndarray, roi_copy: np.ndarray) -> np.ndarray:
            img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Use Otsu's thresholding
            _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return thresh

        self._apply_operation_to_roi(do)

    def _adaptive_thresholding(self) -> None:
        def do(roi: np.ndarray, roi_copy: np.ndarray) -> np.ndarray:
            img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            return cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        self._apply_operation_to_roi(do)

    def _morph_opening(self) -> None:
        def do(roi: np.ndarray, roi_copy: np.ndarray) -> np.ndarray:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            return cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)

        self._apply_operation_to_roi(do)

    def _morph_closing(self) -> None:
        def do(roi: np.ndarray, roi_copy: np.ndarray) -> np.ndarray:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            return cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

        self._apply_operation_to_roi(do)

    def _marked_watershed(self) -> None:
        def do(roi: np.ndarray, roi_copy: np.ndarray) -> np.ndarray:
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

        self._apply_operation_to_roi(do)

    def _combine_masks(self) -> None:
        for record in self.image_data:
            image_name, roi_cuts = record["image_name"], record["roi_cuts"]
            image_path = os.path.join(self.images_dir, image_name)
            original_image = cv2.imread(image_path)

            # print(f"ROIs: {len(roi_cuts)}")
            # print(f"Image name: {image_name}")

            whole_image = np.zeros(original_image.shape[:2], dtype=np.uint8)
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

    def fuse_masks(self, mask_dirs: List[str], out_dir: str, colorized_dir: str = None, vote_pixel: float = 0.5) -> None:
        image_names = os.listdir(self.images_dir)

        for name in image_names:
            fused_mask = np.zeros_like(cv2.imread(os.path.join(self.images_dir, name), cv2.IMREAD_GRAYSCALE))
            for mask_dir in mask_dirs:
                mask_path = os.path.join(mask_dir, name)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                fused_mask = cv2.add(fused_mask, mask)
            mask_out_path = os.path.join(out_dir, name)
            out_mask = np.zeros_like(fused_mask)
            out_mask[fused_mask >= len(mask_dirs)*vote_pixel] = 1
            cv2.imwrite(mask_out_path, out_mask)
            if colorized_dir is not None:
                mask_path = os.path.join(colorized_dir, name)
                cv2.imwrite(mask_path, fused_mask*10)

    def run(self, out_dir: str, operations: List[MaskOperations],
            visualize_range: Tuple[int, int] = (0, 50)) -> None:
        # Cutting
        self._cut_cells(self.json_file_path)

        # Applying CV operations
        operations_pipe = self._get_operation_pipeline(operations=operations)

        self._generate_masks(operations=operations_pipe)

        for record in self.image_data:
            image_name, mask = record["image_name"], record["roi_cuts"][0]["image"]
            mask_path = os.path.join(out_dir, image_name)
            binary_mask = np.array(mask)
            binary_mask[binary_mask == 255] = 1
            cv2.imwrite(mask_path, binary_mask)

        # Visualize results, both original image, mask, and overlay
        i = 0
        for record in self.image_data:
            image, mask = record["roi_cuts"][0]["image_copy"], record["roi_cuts"][0][
                "image"]
            if visualize_range[0] > i or i >= visualize_range[1]:
                i = i+1
                continue
            i = i + 1

            plt.figure(figsize=(10, 10))

            plt.subplot(131)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(132)
            plt.imshow(mask, cmap='gray')
            plt.title("Mask")
            plt.axis('off')

            color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # Get green color for mask
            color_mask[np.all(color_mask == [255, 255, 255], axis=-1)] = [15, 255, 0]
            overlay = cv2.addWeighted(image, 0.7,
                                      color_mask, 0.3, 0)
            plt.subplot(133)
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title("Overlay")
            plt.axis('off')

            plt.show()

