import json
import os
from PIL import Image


def load_json(json_path: str, mode='r') -> dict:
    with open(json_path, mode) as json_file:
        data = json.load(json_file)
    return data


def image_rois_from_coco_json(json_path: str, image_dir: str, output_dir: str, limit: int = 100) -> None:
    """
    """
    data = load_json(json_path)

    images = data['images']
    for image in images:
        if limit % 10 != 0:
            limit -= 1
            continue
        image_id = image['id']
        image_name = image['file_name']
        image_path = os.path.join(image_dir, image_name)
        rois = [ann['bbox'] for ann in data['annotations'] if ann['image_id'] == image_id]
        crop_image_rois(image_path=image_path, regions=rois, output_dir=output_dir)
        print(f"Processed image {image_name}")
        limit -= 1
        if limit <= 0:
            return


def crop_image_rois(image_path: str, output_dir: str, regions: list[list[int, int, int, int]]) -> None:
    # Open the image
    image = Image.open(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    for (x, y, width, height) in regions:
        # Crop each ROI from image
        roi = image.crop((x, y, x + width, y+height))

        # Store each ROI into desired directory
        roi.save(os.path.join(output_dir, f"{image_name}_-_{x}_{y}_{width}_{height}.png"))

