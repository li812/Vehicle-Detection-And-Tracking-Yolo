import io
import os
import shutil
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics import YOLOWorld
from rembg import remove
from PIL import Image,ImageDraw

def Remove_BG(input_path):
    with open(input_path, 'rb') as img_file:
        img = img_file.read()
    output = remove(img)
    img = Image.open(io.BytesIO(output)).convert("RGBA")
    img_rgb = img.convert("RGB")
    os.makedirs('no_bg_img', exist_ok=True)
    print("Finished removing image background!")
    img_rgb.save('no_bg_img/image.jpg')


def Augment_Image():
    image_path = 'no_bg_img/image.jpg'
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
    print("Finished generating augmented images!")
    dataset_images = augmented_images
    os.makedirs('dataset', exist_ok=True)
    for i, img_tuple in enumerate(dataset_images):
        # Check if the image is a tuple and if it contains additional information
        if isinstance(img_tuple, tuple) and len(img_tuple) == 2:
            img = img_tuple[0]  # Extract the image array from the tuple
        else:
            img = img_tuple  # Use the image directly if it's not a tuple

        #print(f"Image {i}: Shape={img.shape}, Dtype={img.dtype}")
        img_np = np.array(img)
        cv2.imwrite(f'dataset/image_{i}.jpg', img_np)
    

def Draw_Boundary():
    path = "dataset"
    # Ensure the path ends with a trailing slash
    if not path.endswith('/'):
        path += '/'

    # Get a list of all files in the folder
    files = os.listdir(path)

    # Iterate through each file
    for file_name in files:
        if file_name.endswith('.jpg'):
            # Open the image
            image_path = os.path.join(path, file_name)
            with Image.open(image_path) as img:
                # Create a drawing object
                draw = ImageDraw.Draw(img)
                
                # Draw a green rectangle around the bounding box
                draw.rectangle([(0, 0), img.size], outline="green", width=10)
                
                # Save the modified image with bounding box
                img.save(image_path)

    print("Finished drawing Boundary Box!")



def Generate_Label():
    dataset_path = "dataset"
    # Define the color range of the green box
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([86, 255,255])

    for image_name in os.listdir(dataset_path):
        image = cv2.imread(os.path.join(dataset_path, image_name))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_center = (x + w / 2) / image.shape[1]
            y_center = (y + h / 2) / image.shape[0]
            width = w / image.shape[1]
            height = h / image.shape[0]
            with open(os.path.join(dataset_path, f"{os.path.splitext(image_name)[0]}.txt"), "w") as file:
                file.write(f"0 {x_center} {y_center} {width} {height}\n")

    print("Finished labeling all images!")


def Organize_Files():
    directory = "dataset"

    images_dir = os.path.join(directory, 'images')
    labels_dir = os.path.join(directory, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for filename in os.listdir(directory):

        filepath = os.path.join(directory, filename)

        if filename.endswith(('.png', '.jpg', '.jpeg')):
            shutil.move(filepath, images_dir)

        elif filename.endswith('.txt'):
            shutil.move(filepath, labels_dir)


def Reset():
    # Define the paths of the folders to be deleted
    dataset_path = "dataset"
    no_bg_img_path = "no_bg_img"
    a = 0
    try:
        # Delete the folders and their contents
        shutil.rmtree(dataset_path)
        print("dataset folder deleted")
        a = a + 1

    except OSError as e:
            print(f"Error: {e.strerror}")
            print("dataset folder doesn't exists")

    try:
        # Delete the folders and their contents
        shutil.rmtree(no_bg_img_path)
        print("no_bg_img folder deleted")
        a = a + 1
    except OSError as e:
            print(f"Error: {e.strerror}")  
            print("no_bg_img folder doesn't exists")

    if a != 0:
        print("Resetting Successful")

    