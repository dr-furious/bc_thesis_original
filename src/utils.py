import os
import json
from typing import Tuple, List
import cv2
import numpy as np
import yaml


def load_images(img_dir: str, supported_formats: Tuple[str] = (".jpg", ".jpeg", ".png")) \
        -> List[Tuple[np.ndarray, str]]:
    images_files = [f for f in os.listdir(img_dir) if f.endswith(supported_formats)]
    images = []

    for image_file in images_files:
        img_path = os.path.join(img_dir, image_file)
        try:
            img = cv2.imread(img_path)
            images.append((img, image_file))
        except FileNotFoundError and FileExistsError as e:
            print(f"Error opening image: {img_path}\nError: {e}")
            continue
    return images


def load_json(json_path: str, mode="r") -> dict:
    with open(json_path, mode) as json_file:
        data = json.load(json_file)
    return data


def load_yaml(yaml_path: str, mode="r") -> dict:
    with open(yaml_path, mode) as yaml_file:
        data = yaml.safe_load(yaml_file)
    return data


def build_paths(base_dir: str, subdirs: List[str]) -> List[str]:
    return [os.path.join(base_dir, name) for name in subdirs]
