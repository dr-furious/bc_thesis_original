import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils import load_json


class Sample:
    img_path: str | None = None
    mask_path: str | None = None
    img: np.ndarray | None = None
    mask: np.ndarray | None = None

    def __init__(self, img_path: str | None = None, mask_path: str | None = None) -> None:
        self.img_path = img_path
        self.mask_path = mask_path
        # Try loading the image
        try:
            self.img = cv2.imread(self.img_path)
        except Exception as e:
            print(f"Error occurred while opening the image: {e}")

        try:
            self.mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
            self.mask[self.mask == 1] = 255
        except Exception as e:
            print(f"Error occurred while opening the mask: {e}")

    def info(self):
        print(f"Image path: {self.img_path}")
        print(f"mask path: {self.mask_path}")
        print(f"Image loaded: {self.img is not None}")
        print(f"mask loaded: {self.mask is not None}")
        if self.img is not None:
            print(f"Image shape: {self.img.shape}")
        if self.mask is not None:
            print(f"mask shape: {self.mask.shape}")

    def show(self, with_mask: bool = False) -> None:
        if self.img is None:
            print("Image is None")
            return
        if with_mask is True and self.mask is None:
            print("mask is None")
            return

        display_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        if with_mask:
            self.mask[self.mask == 1] = 255
            self.mask = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2RGB)
            self.mask[np.all(self.mask == [255, 255, 255], axis=-1)] = [15, 255, 0]
            plt.figure(figsize=(16, 8), dpi=80, facecolor="w", edgecolor="k")

            # Overlay both the images for visualization
            overlay = cv2.addWeighted(display_img, 0.8, self.mask, 0.2, 0)

            # Display original image
            plt.subplot(1, 3, 1)
            plt.imshow(display_img)
            plt.title("Original Image")

            # Display self.mask
            plt.subplot(1, 3, 2)
            plt.imshow(self.mask)
            plt.title("Mask")

            # Display overlay
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title("Overlay")

            plt.show()
        else:
            plt.figure(figsize=(4, 4), dpi=80, facecolor="w", edgecolor="k")
            plt.imshow(display_img)
            plt.show()

    def save_overlay(self, out_dir: str) -> None:
        name = os.path.basename(self.img_path)

        # Save the original image
        cv2.imwrite(os.path.join(out_dir, f"image_{name}"), self.img)

        self.mask[self.mask == 1] = 255
        # Save the mask
        cv2.imwrite(os.path.join(out_dir, f"mask_{name}"), self.mask)

        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2RGB)
        self.mask[np.all(self.mask == [255, 255, 255], axis=-1)] = [15, 255, 0]
        overlay = cv2.addWeighted(self.img, 0.8, self.mask, 0.2, 0)
        # Save overlaid image and mask
        cv2.imwrite(os.path.join(out_dir, f"overlay_{name}"), overlay)

    def save_with_bbox(self, out_dir: str, json_path: str) -> None:
        json_data = load_json(json_path)
        images = json_data["images"]
        name = os.path.basename(self.img_path)
        for image in images:
            image_name = image["file_name"]
            image_id = image["id"]
            if image_name[2:] != name:
                continue
            out_path = os.path.join(out_dir, f"bbox_{name}")
            rois = [ann["bbox"] for ann in json_data["annotations"] if ann["image_id"] == image_id]
            # Draw each bounding box
            for (x, y, width, height) in rois:
                top_left = (int(x), int(y))
                bottom_right = (int(x + width), int(y + height))
                color = (0, 255, 0)  # BGR → green
                thickness = 1  # thin line
                self.img = cv2.rectangle(self.img, top_left, bottom_right, color, thickness)
            cv2.imwrite(out_path, self.img)

