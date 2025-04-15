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
        # Get the output of the first convolutional layer
        x1 = self.features[0](x)
        
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        x = self.softmax(x)
        
        return x1, x

# Function to visualize feature maps
def visualize_feature_maps(feature_maps):
    # Select the first image's feature maps for visualization
    feature_maps = feature_maps[0]
    
    # Number of feature maps
    num_feature_maps = feature_maps.shape[0]
    
    # Plot the feature maps
    plt.figure(figsize=(20, 20))
    for i in range(num_feature_maps):
        plt.subplot(int(np.sqrt(num_feature_maps)), int(np.sqrt(num_feature_maps)), i + 1)
        plt.imshow(feature_maps[i].cpu().detach().numpy(), cmap='gray')
        plt.axis('off')
    
    plt.show()

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

        # Convert back to PIL image for visualization
        image_clahe_pil = transforms.ToPILImage()(image_clahe)
        
        # Show images if debug mode is enabled
        if self.debug and idx == 0:  # Display only the first image for debugging
            self.show_image_processing_steps(image_rgb, image_clahe_pil, green_channel_image, img_id, label)
        
        return image_clahe, label

    def _apply_clahe(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        lab_planes = list(cv2.split(lab))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

    def show_image_processing_steps(self, image, clahe_image, green_channel_image, img_id, label):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title(f'Original Image\nLabel: {label}\nFilename: {img_id}')
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
def get_dataloaders(csv_file, img_dir, batch_size=32, test_size=0.2):
    full_dataset = FundusDataset(csv_file=csv_file, img_dir=img_dir)
    train_indices, test_indices = train_test_split(range(len(full_dataset)), test_size=test_size, stratify=full_dataset.labels_df['label'])
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Function to evaluate the model
def evaluate_model(model, test_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    # Calculate ROC AUC Score
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    print(f'ROC AUC Score: {roc_auc:.4f}')
    
    # Calculate F1 Score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'F1 Score: {f1:.4f}')
    
    # Print Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(cm)

# Define the transformation for images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Set up the dataset and dataloaders
csv_file = r'C:\Users\renat\OneDrive\Documents\skripsi\Code\data gambar dan kelas.csv'  # Adjust with your CSV file path
img_dir = r'C:\Users\renat\OneDrive\Documents\skripsi\Code\HasilCrop - Copy'  # Adjust with your image directory

train_loader, test_loader = get_dataloaders(csv_file, img_dir, batch_size=32)

# Initialize model, criterion, optimizer
model = InceptionResNetV2(num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Set up early stopping
early_stopping = EarlyStopping(patience=3, verbose=True)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Evaluate the model and visualize feature maps
model.eval()
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(torch.device)
        feature_maps, _ = model(inputs)
        visualize_feature_maps(feature_maps)
        break  # Only visualize for the first batch
