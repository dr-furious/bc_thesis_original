import cv2
import matplotlib.pyplot as plt
import numpy as np


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
            self.mask = cv2.imread(self.mask_path)
        except Exception as e:
            print(f"Error occurred while opening the mask: {e}")

    def equalize_hist(self, inplace: bool = False) -> np.ndarray | None:
        if self.img is None:
            print("Image is None")
            return
        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
        equalized_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        if inplace is True:
            self.img = equalized_img
        return equalized_img

    def otsu_thresholding(self, inplace: bool = False) -> np.ndarray | None:
        if self.img is None:
            print("Image is None")
            return
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Use Otsu's thresholding
        _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert binary mask to 3-channel (so it matches the image dimensions)
        mask_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        # Choose mask color (e.g., red [255, 0, 0]), White pixels → Red
        mask_colored[np.where((mask_colored == [255, 255, 255]).all(axis=2))] = [255, 0, 0]

        if inplace is True:
            self.mask = mask_colored
        return mask_colored

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
            plt.figure(figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')

            # Overlay both the images for visualization
            overlay = cv2.addWeighted(display_img, 0.8, self.mask, 0.2, 0)

            # Display original image
            plt.subplot(1, 3, 1)
            plt.imshow(display_img)
            plt.title("Original Image")

            # Display mask
            plt.subplot(1, 3, 2)
            plt.imshow(self.mask)
            plt.title("Mask")

            # Display overlay
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title("Overlay")

            plt.show()
        else:
            plt.figure(figsize=(4, 4), dpi=80, facecolor='w', edgecolor='k')
            plt.imshow(display_img)
            plt.show()
