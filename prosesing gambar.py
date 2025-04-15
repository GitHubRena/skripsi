import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dataset Class
class FundusDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, debug=False, max_debug_images=5):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.debug = debug
        self.max_debug_images = max_debug_images

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = str(self.labels_df.iloc[idx, 0])
        img_path = os.path.join(self.img_dir, img_id)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f'Image not found: {img_path}')

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f'Error reading image: {img_path}')
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract green channel
        green_channel = image_rgb[:, :, 1]
        green_channel_image = np.stack((green_channel, green_channel, green_channel), axis=-1)
        
        # Apply CLAHE
        image_clahe = self._apply_clahe(green_channel_image)
        
        # Apply sharpening
        image_sharpened = self._apply_sharpening(image_clahe)
        
        # Apply super-resolution
        image_super_res = self._apply_super_resolution(image_sharpened)
        
        # Convert to PIL image for transforms
        image_pil = Image.fromarray(image_super_res)

        # Apply transformations
        if self.transform:
            image_pil = self.transform(image_pil)
        
        label = self.labels_df.iloc[idx, 1]
        filename = self.labels_df.iloc[idx, 0]

        if self.debug and idx < self.max_debug_images:
            self.show_image_processing_steps(
                image_rgb,
                image_pil.permute(1, 2, 0).numpy(),
                green_channel_image,
                filename,
                label
            )

        return image_pil, label, filename

    def _apply_clahe(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    def _apply_sharpening(self, img):
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        return cv2.filter2D(img, -1, kernel)

    def _apply_super_resolution(self, img):
        return cv2.resize(img, (299, 299), interpolation=cv2.INTER_CUBIC)
    
    def show_image_processing_steps(self, original, transformed, green_channel_image, filename, label):
        plt.figure(figsize=(12, 6))
        
        # Tampilkan gambar asli
        plt.subplot(1, 3, 1)
        plt.imshow(original)
        plt.title(f'Original Image\nFilename: {filename}\nLabel: {label}')
        
        # Tampilkan gambar channel hijau
        plt.subplot(1, 3, 2)
        plt.imshow(green_channel_image)
        plt.title('Green Channel Image')
        
        # Tampilkan gambar yang sudah diproses
        plt.subplot(1, 3, 3)
        plt.imshow(transformed)
        plt.title('Transformed Image')
        
        plt.tight_layout()
        plt.show()

# Function to visualize one image per class
def visualize_one_image_per_class(dataset, num_classes=5):
    class_images = {}
    
    for idx in range(len(dataset)):
        image, label, filename = dataset[idx]
        
        if label not in class_images:
            class_images[label] = (image, filename)
        
        if len(class_images) >= num_classes:
            break

    fig, axes = plt.subplots(1, num_classes, figsize=(15, 5), squeeze=False)
    
    for i, (label, (image, filename)) in enumerate(class_images.items()):
        image = image.permute(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
        image = (image - image.min()) / (image.max() - image.min())
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'Class {label}\nFilename: {filename}')
        axes[0, i].axis('off')

    plt.tight_layout()
    plt.show()

# Define your transforms
def calculate_mean_std(dataset):
    mean = 0.0
    std = 0.0
    num_samples = 0
    
    for img, _, _ in dataset:
        img = np.array(img)  # Convert PIL image to numpy array
        img = img / 255.0    # Normalize to [0, 1] before calculating mean and std
        num_samples += 1
        mean += img.mean(axis=(0, 1))
        std += img.std(axis=(0, 1))

    mean /= num_samples
    std /= num_samples
    return mean, std

# Initialize dataset with a placeholder transform
temp_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])

# Initialize dataset with placeholder transform
dataset = FundusDataset(
    csv_file = r'C:\Users\renat\OneDrive\Documents\skripsi\Code\images_id_kelas.csv',  # Adjust with your CSV file path
    img_dir = r'C:\Users\renat\OneDrive\Documents\skripsi\Code\HasilCrop299',
    transform=temp_transform,
    debug=True
)

# Calculate mean and std
mean, std = calculate_mean_std(dataset)

# Define the actual transform with calculated mean and std
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])

# Initialize dataset again with updated transform
dataset = FundusDataset(
    csv_file=r'C:\Users\renat\OneDrive\Documents\skripsi\Code\data gambar dan kelas.csv',
    img_dir=r'C:\Users\renat\OneDrive\Documents\skripsi\Code\HasilCrop - Copy',
    transform=transform,
    debug=True
)

# Visualize one image per class
visualize_one_image_per_class(dataset)

# DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
