from typing import List, Tuple
import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.utils import load_images


class ImageStats:
    images_dir: str | None
    images: List[Tuple[np.ndarray, str]]

    def __init__(self, images_dir: str, supported_formats: Tuple[str] = (".jpg", ".jpeg", ".png"),
                 images: List[np.ndarray] = None) -> None:
        if images is not None:
            self.images_dir = None
            self.images = [(img, "") for img in images]
            return
        self.images_dir = images_dir
        self.images = load_images(images_dir, supported_formats)

    # Display stats about images from directory
    def get_image_stats(self, visual: bool = False) -> dict:
        sizes = []
        for image in self.images:
            sizes.append(image[0].shape[:2])

        image_sizes_np = np.array(sizes)
        count = len(image_sizes_np)
        average_width = np.mean(image_sizes_np[:, 0])
        average_height = np.mean(image_sizes_np[:, 1])
        average_area = np.mean(image_sizes_np[:, 0] * image_sizes_np[:, 1])

        med_width = np.median(image_sizes_np[:, 0])
        med_height = np.median(image_sizes_np[:, 1])
        med_area = np.median(image_sizes_np[:, 0] * image_sizes_np[:, 1])

        min_width = np.min(image_sizes_np[:, 0])
        min_height = np.min(image_sizes_np[:, 1])
        min_area = np.min(image_sizes_np[:, 0] * image_sizes_np[:, 1])

        max_width = np.max(image_sizes_np[:, 0])
        max_height = np.max(image_sizes_np[:, 1])
        max_area = np.max(image_sizes_np[:, 0] * image_sizes_np[:, 1])

        print(f"Count: {count}\n")
        print(f"Average Width: {average_width}")
        print(f"Average Height: {average_height}")
        print(f"Average image area: {average_area}")
        print(f"Median Width: {med_width}")
        print(f"Median Height: {med_height}")
        print(f"Median Area: {med_area}")
        print(f"Min Width: {min_width}")
        print(f"Min Height: {min_height}")
        print(f"Min Area: {min_area}")
        print(f"Max Width: {max_width}")
        print(f"Max Height: {max_height}")
        print(f"Max Area: {max_area}")

        stats = {
            "count": count,
            "avg_w": average_width,
            "avg_h": average_height,
            "avg_area": average_area,
            "med_w": med_width,
            "med_h": med_height,
            "med_area": med_area,
            "min_w": min_width,
            "min_h": min_height,
            "min_area": min_area,
            "max_w": max_width,
            "max_h": max_height,
            "max_area": max_area
        }

        if visual is False:
            return stats

        # Create a figure and a 1x3 subplot scheme
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot distribution of widths
        axs[0].hist(image_sizes_np[:, 0], bins=20, alpha=0.6, color="cyan")
        axs[0].set_title("Widths distribution")

        # Plot distribution of heights
        axs[1].hist(image_sizes_np[:, 1], bins=20, alpha=0.6, color="yellow")
        axs[1].set_title("Heights distribution")

        # Plot distribution of image area
        image_area = np.array([w * h for w, h in sizes])
        axs[2].hist(image_area, bins=20, alpha=0.6, color="magenta")
        axs[2].set_title("Image sizes distribution")

        return stats

    # Compute contrast std
    def contrast_std(self) -> np.floating:
        std_devs = [np.std(cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)) for img in self.images]
        mean_dev = np.mean(std_devs)
        print(f"Contrast standard deviations: {mean_dev}")
        return mean_dev

    # Compute the michelson contrast
    def michelson_contrast(self) -> np.floating:
        michelson_contrasts = []
        for img in self.images:
            img_gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY).astype(np.int16)
            img_max = np.max(img_gray)
            img_min = np.min(img_gray)
            if img_max + img_min != 0:
                contrast = np.ptp(img_gray) / (img_max + img_min)
                michelson_contrasts.append(contrast)

        mean_contrast = np.mean(michelson_contrasts) if michelson_contrasts else 0
        print(f"Michelson contrast: {mean_contrast}")
        return mean_contrast

