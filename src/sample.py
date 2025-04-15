import cv2
import matplotlib.pyplot as plt
import numpy as np


class Sample:
    img_path: str | None = None
    label_path: str | None = None
    img: np.ndarray | None = None
    label: np.ndarray | None = None

    def __init__(self, img_path: str | None = None, label_path: str | None = None) -> None:
        self.img_path = img_path
        self.label_path = label_path
        # Try loading the image
        try:
            self.img = cv2.imread(self.img_path)
        except Exception as e:
            print(f"Error occurred while opening the image: {e}")

        try:
            self.label = cv2.imread(self.label_path)
        except Exception as e:
            print(f"Error occurred while opening the label: {e}")

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

    def marked_watershed(self, inplace: bool = False) -> np.ndarray | None:
        if self.img is None:
            print("Image is None")
            return
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Use Otsu's thresholding
        _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure bg area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Sure fg area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labeling
        markers = cv2.connectedComponentsWithStats(sure_fg)[1]

        # Add one to all labels so that sure bg is 1 not 0
        markers = markers + 1

        # Mark the region of unknown with 0
        markers[unknown == 255] = 0

        # Perform watershed
        markers = cv2.watershed(self.img, markers)

        # Get binary mask
        binary_mask = (markers > 1).astype("uint8")
        binary_mask = (binary_mask * 255).astype("uint8")

        if inplace is True:
            self.label = binary_mask
        return binary_mask

    def info(self):
        print(f"Image path: {self.img_path}")
        print(f"Label path: {self.label_path}")
        print(f"Image loaded: {self.img is not None}")
        print(f"Label loaded: {self.label is not None}")
        if self.img is not None:
            print(f"Image shape: {self.img.shape}")
        if self.label is not None:
            print(f"Label shape: {self.label.shape}")

    def show(self, with_label: bool = False) -> None:
        if self.img is None:
            print("Image is None")
            return
        if with_label is True and self.label is None:
            print("Label is None")
            return

        plt.figure(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

        display_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        if with_label:
            display_label = cv2.cvtColor(self.label, cv2.COLOR_BGR2RGB)
            plt.subplot(1, 2, 1)
            plt.imshow(display_img)
            plt.subplot(1, 2, 2)
            plt.imshow(display_label)
            plt.show()
        else:
            plt.imshow(display_img)
            plt.show()
