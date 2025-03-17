import os
import yaml
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def create_dataset_structure():
    """Create dataset directory structure for YOLOv8"""
    dirs = ['dataset/images/train', 'dataset/images/val',
            'dataset/labels/train', 'dataset/labels/val']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def prepare_data(source_dir, train_ratio=0.8):
    """
    Prepare dataset from raw images.
    Args:
        source_dir: Directory containing date fruit images.
        train_ratio: Ratio of training to validation split.
    """
    create_dataset_structure()

    # Get all image files
    image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)

    # Split into train and validation
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # Process images
    for img_file in train_files:
        process_image(os.path.join(source_dir, img_file), 'train')
    
    for img_file in val_files:
        process_image(os.path.join(source_dir, img_file), 'val')

def process_image(img_path, subset):
    """
    Detects object in image, creates bounding box, and saves YOLO annotation.
    Args:
        img_path: Path to image file.
        subset: 'train' or 'val'.
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)  # Thresholding

    # Find contours (object boundaries)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get largest contour (assuming it's the date fruit)
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)  # Bounding box

        # Normalize values for YOLO format
        img_h, img_w = img.shape[:2]
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w = w / img_w
        h = h / img_h

        # Save annotation in YOLO format
        label_filename = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        label_path = f'dataset/labels/{subset}/{label_filename}'
        with open(label_path, 'w') as f:
            f.write(f'0 {x_center} {y_center} {w} {h}\n')

    # Copy image to dataset
    new_img_path = f'dataset/images/{subset}/{os.path.basename(img_path)}'
    shutil.copy(img_path, new_img_path)

def create_data_yaml():
    """Create data.yaml configuration file for YOLOv8"""
    data_yaml = {
        'train': './dataset/images/train',
        'val': './dataset/images/val',
        'nc': 1,  # Number of classes
        'names': ['date']  # Class names
    }

    with open('dataset/data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

def train_yolo():
    """Returns YOLOv8 training command"""
    return """
    # Install Ultralytics package
    pip install ultralytics

    # Train YOLOv8 model
    yolo task=detect mode=train model=yolov8s.pt data=dataset/data.yaml epochs=100 imgsz=640 batch=16
    """

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Path to dataset (update this!)
    source_directory = r"C:\Users\anujn\OneDrive\Documents\Capstone\dates\Dates 1"

    # Prepare dataset and annotations
    prepare_data(source_directory)

    # Create YAML configuration file
    create_data_yaml()

    # Print training command
    print("To train YOLOv8, run the following command:")
    print(train_yolo())
