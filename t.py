import io
import os
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics import YOLOWorld
from rembg import remove
from PIL import Image

def Remove_BG(input_path):
    with open(input_path, 'rb') as img_file:
        img = img_file.read()
    output = remove(img)
    img = Image.open(io.BytesIO(output)).convert("RGBA")
    img_rgb = img.convert("RGB")
    return img_rgb



def Augment_Image(image_path):
    image = cv2.imread(image_path)
    augmented_images = [image.copy()]  # Start with the original image
    # Define the rotation angles, scaling factors, and translation distances
    angles = [90, 180, 270]
    scales = [0.5, 1.5, 2.0]
    translations = [(50, 50), (-50, 50), (50, -50), (-50, -50)]
    (h, w) = image.shape[:2]  # Get image height and width
    # Apply rotation transformations
    for angle in angles:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented_images.append(rotated)
    # Apply scaling transformations
    for scale in scales:
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        augmented_images.append(resized)
    # Apply translation transformations
    for translation in translations:
        M = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
        translated = cv2.warpAffine(image, M, (w, h))
        augmented_images.append(translated)
    # Apply brightness and contrast alterations
    brightness_altered = cv2.convertScaleAbs(image, alpha=1, beta=50)  # increase brightness
    contrast_altered = cv2.convertScaleAbs(image, alpha=1.5, beta=0)  # increase contrast
    augmented_images.extend([brightness_altered, contrast_altered])
    # Apply horizontal and vertical flipping
    flipped_horizontally = cv2.flip(image, 1)
    flipped_vertically = cv2.flip(image, 0)
    flipped_both = cv2.flip(image, -1)
    augmented_images.extend([flipped_horizontally, flipped_vertically, flipped_both])
    return augmented_images


user_input_img = 'search_img/s1.jpg'

background_removed_image = Remove_BG(user_input_img)

background_removed_image.show()

background_removed_image.save('no_bg_img/image.jpg')

no_bg_image_path = 'no_bg_img/image.jpg'

# Augment the single image to create a dataset
dataset_images = Augment_Image(no_bg_image_path)

# Save augmented images to a directory
os.makedirs('dataset', exist_ok=True)
for i, img in enumerate(dataset_images):
    if isinstance(img, np.ndarray):  # Check if img is a NumPy array
        cv2.imwrite(f'dataset/image_{i}.jpg', img)
    else:
        print(f"Skipping image {i} as it's not a valid NumPy array.")
