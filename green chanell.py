import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import EarlyStopping from a separate file
from early_stopping import EarlyStopping

# Define the basic convolutional layer
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

# Inception module from InceptionResNetV2
class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=5, padding=2)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1)
        )

    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        outputs = [branch0, branch1, branch2, branch3]
        return torch.cat(outputs, 1)

# Residual block for InceptionResNetV2
class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1),
            BasicConv2d(32, 48, kernel_size=3, padding=1),
            BasicConv2d(48, 64, kernel_size=3, padding=1)
        )
        self.conv2d = nn.Conv2d(32 + 32 + 64, 320, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branches = [branch0, branch1, branch2]
        mixed = torch.cat(branches, 1)
        mixed = self.conv2d(mixed)
        mixed = mixed * self.scale + x
        return self.relu(mixed)

# InceptionResNetV2 model
class InceptionResNetV2(nn.Module):
    def __init__(self, num_classes=5):
        super(InceptionResNetV2, self).__init__()
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2),
            BasicConv2d(64, 80, kernel_size=1),
            BasicConv2d(80, 192, kernel_size=3),
            nn.MaxPool2d(3, stride=2),
            Mixed_5b(),
            nn.Sequential(
                Block35(scale=0.17),
                Block35(scale=0.17),
                Block35(scale=0.17),
                Block35(scale=0.17),
                Block35(scale=0.17)
            )
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(320, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        x = self.softmax(x)
        return x

# Dataset class for fundus images
class FundusDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, debug=False):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.debug = debug

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = self.labels_df.iloc[idx, 0]
        img_id = str(img_id)  # Ensure img_id is a string
        img_path = os.path.join(self.img_dir, img_id)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f'Image not found: {img_path}')

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f'Error reading image: {img_path}')
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract the green channel first
        green_channel = image_rgb[:, :, 1]
        green_channel_image = np.stack((green_channel, green_channel, green_channel), axis=-1)
        
        # Apply CLAHE on the green channel image
        image_clahe = self._apply_clahe(green_channel_image)
        
        if self.transform:
            image_clahe = self.transform(image_clahe)
        
        label = self.labels_df.iloc[idx, 1]
        filename = self.labels_df.iloc[idx, 0]  # Get the filename for visualization

        # Convert back to PIL image for visualization
        image_clahe_pil = transforms.ToPILImage()(image_clahe)
        
        # Show images if debug mode is enabled
        if self.debug and idx == 0:  # Display only the first image for debugging
            self.show_image_processing_steps(image_rgb, image_clahe_pil, green_channel_image)
        
        return image_clahe, label, filename  # Return filename along with image and label

    def _apply_clahe(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        lab_planes = list(cv2.split(lab))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

    def show_image_processing_steps(self, image, clahe_image, green_channel_image):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(green_channel_image)
        plt.title('Green Channel')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(clahe_image)
        plt.title('CLAHE Applied')
        plt.axis('off')
        
        plt.show()

# Function to create train and test data loaders
def get_dataloaders(csv_file, img_dir, batch_size=32, test_size=0.2, random_state=42, debug=False):
    dataset = FundusDataset(csv_file=csv_file, img_dir=img_dir, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Updated normalization
    ]), debug=debug)
    
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

# Function to train the model
def train_model(model, device, train_loader, test_loader, criterion, optimizer, num_epochs=50, patience=7):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    model.to(device)
    train_losses = []
    val_losses = []
    auc_scores = []
    f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels, _ in train_loader:  # Include filename in inputs
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels, _ in test_loader:  # Include filename in inputs
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        epoch_val_loss = running_loss / len(test_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Calculate ROC AUC score and F1 score
        try:
            auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')
        except ValueError:
            auc = float('nan')
        auc_scores.append(auc)
        
        f1 = f1_score(all_labels, all_preds, average='weighted')
        f1_scores.append(f1)
        
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f}')
        print(f'Validation Loss: {epoch_val_loss:.4f}')
        print(f'ROC AUC Score: {auc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    return train_losses, val_losses, auc_scores, f1_scores

# Function to check data loader
def check_data_loader(data_loader):
    for images, labels, filenames in data_loader:  # Include filenames
        print(f'Images batch shape: {images.size()}')
        print(f'Labels batch shape: {labels.size()}')
        print(f'Filenames batch: {filenames}')
        break

# Function to visualize data and labels
def visualize_data_labels(dataset, indices):
    for idx in indices:
        image, label, filename = dataset[idx]
        plt.figure(figsize=(6, 6))
        plt.imshow(transforms.ToPILImage()(image))
        plt.title(f'Filename: {filename}, Label: {label}')
        plt.axis('off')
        plt.show()

# Function to check model predictions
def check_model_predictions(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for inputs, labels, filenames in data_loader:  # Include filenames
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            print(f'Predictions: {preds.cpu().numpy()}')
            print(f'Labels: {labels.cpu().numpy()}')
            print(f'Filenames: {filenames}')
            break

# Main function
def main():
    csv_file = r'C:\Users\renat\OneDrive\Documents\skripsi\Code\data gambar dan kelas.csv'  # Adjust with your CSV file path
    img_dir = r'C:\Users\renat\OneDrive\Documents\skripsi\Code\HasilCrop - Copy'  # Adjust with your image directory

    batch_size = 16
    num_epochs = 50
    patience = 7

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InceptionResNetV2(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader, test_loader = get_dataloaders(csv_file, img_dir, batch_size=batch_size, debug=True)
    
    # Debugging: Check data loader
    check_data_loader(train_loader)
    
    # Debugging: Visualize some images and labels from the dataset
    visualize_data_labels(train_loader.dataset, [0, 10, 20])
    
    # Debugging: Check predictions from the model
    check_model_predictions(model, test_loader, device)

    train_losses, val_losses, auc_scores, f1_scores = train_model(model, device, train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs, patience=patience)

    # Plotting training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
