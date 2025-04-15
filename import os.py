import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
import cv2
import numpy as np
import time

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
        self.branch0 = BasicConv2d(320, 96, kernel_size=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(320, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=5, padding=2)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(320, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(320, 64, kernel_size=1)
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
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1)
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
    def __init__(self, num_classes=5):  # Adjusted to 5 classes
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
        self.dropout = nn.Dropout(p=0.5)  # Changed dropout rate to 0.5
        self.fc1 = nn.Linear(320, 512)    # Added additional dense layers
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)  # Adjusted output layer
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for output

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.bn1(self.fc1(x)))  # Added BatchNorm and ReLU after each dense layer
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        x = self.softmax(x)
        return x

# Dataset class for fundus images
class FundusDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = self.labels_df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f'{img_id}.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, f'{img_id}.png')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self._apply_clahe(image)
        if self.transform:
            image = self.transform(image)
        label = self.labels_df.iloc[idx, 1]
        return image, label

    def _apply_clahe(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

# Function to create train and test data loaders
def get_dataloaders(csv_file, img_dir, batch_size=32, test_size=0.2, random_state=42):
    dataset = FundusDataset(csv_file, img_dir, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((75, 75)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    train_idx, test_idx = train_test_split(
        list(range(len(dataset))), test_size=test_size, random_state=random_state, stratify=dataset.labels_df['adjudicated_dr_grade']
    )

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4)

    return train_loader, test_loader

# Function to train the model
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        elapsed_time = time.time() - start_time

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {elapsed_time:.2f} seconds')

        # Evaluate on test set
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        sensitivity, specificity, accuracy, auc, f1 = evaluate_performance(all_labels, all_preds)
        print(f'Evaluation Metrics: Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1 Score: {f1:.4f}')

# Function to evaluate model performance
def evaluate_performance(labels, preds):
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds, multi_class='ovo')
    f1 = f1_score(labels, preds, average='weighted')
    return sensitivity, specificity, accuracy, auc, f1

# Main function to run the training process
def main():
    csv_file = '/content/drive/MyDrive/Projek ML Diabetes/messidor_data_new3.csv'
    img_dir = ''
    num_classes = 5  # Adjust number of classes as needed
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.0001

    train_loader, test_loader = get_dataloaders(csv_file, img_dir, batch_size)
    model = InceptionResNetV2(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs)

if __name__ == '__main__':
    main()
