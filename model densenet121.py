import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from torch.optim.lr_scheduler import StepLR
from collections import Counter

from early_stopping import EarlyStopping  # Import early stopping class

# GPU Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------
# Custom Dataset Class
# -------------------------------------------------
class FundusDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, for_model='inception'):
        """
        Args:
            csv_file (str): Path to the CSV file containing image metadata.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
            for_model (str): Target model ('inception' or 'densenet121').
        """
        self.label_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.for_model = for_model.lower()  # Ensure consistent lowercase
        self.label_severity = self.label_df['adjudicated_dr_grade'].values  # Adjust this line if necessary
    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        # Fetch image ID and label
        img_id = str(self.label_df.iloc[idx, 0])
        img_path = os.path.join(self.img_dir, img_id)
        original_label = self.label_df.iloc[idx, 1]
        # Load image
        if not os.path.exists(img_path):
            raise FileNotFoundError(f'Image not found: {img_path}')
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f'Error reading image: {img_path}')
        if image is None:
            return None, None, None 
        
        # Preprocess image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Extract the green channel first ("terlihat hijau" (tetapi sebenarnya seluruh citra akan terlihat abu-abu karena semua saluran memiliki nilai yang sama))
        green_channel = image_rgb[:, :, 1]
        green_channel_image = np.stack((green_channel, green_channel, green_channel), axis=-1)
        image_clahe = self._apply_clahe(green_channel_image)
        image_sharpened = self._apply_sharpening(image_clahe)
        image_super_res = self._apply_super_resolution(image_sharpened)
        
        if self.transform:
            image_super_res = self.transform(image_super_res)

        if self.for_model == 'inception':
            label_dr = 0 if original_label == 0 else 1
            label_severity = None  # Placeholder
        elif self.for_model == 'densenet121':
            if original_label == 0:
                return None, None, None
            label_dr = None  # Not required for severity model
            label_severity = original_label
        else:
            raise ValueError(f"Unknown model type: {self.for_model}")

        # label_severity = torch.tensor(self.label_severity[idx]).long() if self.label_severity[idx] is not None else torch.tensor(0).long()

        return image_super_res,label_dr, label_severity

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

# -------------------------------------------------
# Utility Functions
# -------------------------------------------------
def calculate_mean_std(dataset):
    """
    Calculate mean and standard deviation for a dataset.
    """
    mean, std = 0.0, 0.0
    num_samples = len(dataset)
    

    for img, _, _ in dataset:
        img = img.numpy()
        mean += img.mean(axis=(1, 2))
        std += img.std(axis=(1, 2))
    
    mean /= num_samples
    std /= num_samples
    return mean, std

class ResampledDataset(Dataset):
    def __init__(self, features, label_dr=None, label_severity=None,is_severity_task=False):
        """
        Dataset wrapper for resampled features and labels.

        Args:
            features (list or ndarray): Pre-processed features, typically flattened images.
            label_dr (list or ndarray, optional): Labels for binary classification (DR/Non-DR).
            label_severity (list or ndarray, optional): Labels for severity classification.
            is_severity_task (bool): If True, the dataset is used for severity training.
        """
        # assert len(features) > 0, "Features must not be empty."
        # if label_dr is not None:
        #     assert len(features) == len(label_dr), "Features and label_dr must have the same length."
        # # if label_severity is not None:
        # #     assert len(features) == len(label_severity), "Features and label_severity must have the same length."

        self.features = features
        self.label_dr = label_dr
        self.label_severity = label_severity
        self.is_severity_task = is_severity_task

        self.dummy_severity_label = [0] * len(features) if not is_severity_task else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Load the image
        image = torch.tensor(self.features[idx]).reshape(3, 299, 299).float()

        # Load labels if they exist
        label_dr = torch.tensor(self.label_dr[idx]).long() if self.label_dr is not None else None
        label_severity = (
            torch.tensor(self.label_severity[idx]).long()
            if self.is_severity_task and self.label_severity is not None
            else torch.tensor(0).long()
        )
        return image, label_dr, label_severity

# -------------------------------------------------
# Dataset and DataLoader Setup
# -------------------------------------------------
# File paths
csv_file = r'C:\Users\renat\OneDrive\Documents\skripsi\Code\images_id_kelas.csv'
img_dir = r'C:\Users\renat\OneDrive\Documents\skripsi\Code\HasilCrop299'

# Placeholder transform for mean and std calculation
temp_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])

# Initialize dataset
dataset = FundusDataset(csv_file=csv_file, img_dir=img_dir, transform=temp_transform)
mean, std = calculate_mean_std(dataset)

# Define actual transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Dataset for InceptionResNetV2 (binary classification)
dataset_inception = FundusDataset(csv_file=csv_file, img_dir=img_dir, transform=transform, for_model='inception')

# Extract features and label for binary classification
features_inception, label_inception = [], []
for img, label_dr, _ in dataset_inception:
    features_inception.append(img.numpy().flatten())
    label_inception.append(label_dr)

# Oversample binary dataset
ros = RandomOverSampler(random_state=42)
X_resampled_binary, y_resampled_binary = ros.fit_resample(np.array(features_inception), np.array(label_inception))

# Prepare binary dataset

resampled_dataset_inception = ResampledDataset(X_resampled_binary, y_resampled_binary, is_severity_task=False)
train_size = int(0.8 * len(resampled_dataset_inception))
test_size = len(resampled_dataset_inception) - train_size
train_dataset_inception, test_dataset_inception = random_split(resampled_dataset_inception, [train_size, test_size])
train_loader_inception = DataLoader(train_dataset_inception, batch_size=8, shuffle=True, pin_memory=True)
test_loader_inception = DataLoader(test_dataset_inception, batch_size=8, shuffle=False, pin_memory=True)

# Dataset untuk DenseNet121
dataset_densenet = FundusDataset(
    csv_file=csv_file,
    img_dir=img_dir,
    transform=transform,
    for_model='densenet121'
)

# Filter and oversample severity dataset
features_densenet, label_densenet = [], []
for img, _, label_severity in dataset_inception:
    if label_severity is not None:
        features_densenet.append(img.numpy().flatten())
        label_densenet.append(label_severity)

print("Before resampling:", Counter(label_densenet))

_ , y_densenet_resampled = ros.fit_resample(np.array(features_densenet), np.array(label_densenet))
resampled_dataset_densenet = ResampledDataset(features_densenet, y_densenet_resampled,is_severity_task=True)
print("After resampling:", Counter(y_densenet_resampled))

train_size_densenet = int(0.8 * len(resampled_dataset_densenet))
test_size_densenet = len(resampled_dataset_densenet) - train_size_densenet
train_dataset_densenet, test_dataset_densenet = random_split(resampled_dataset_densenet, [train_size_densenet, test_size_densenet])
train_loader_densenet = DataLoader(train_dataset_densenet, batch_size=8, shuffle=True, pin_memory=True)
test_loader_densenet = DataLoader(test_dataset_densenet, batch_size=8, shuffle=False, pin_memory=True)
for batch in train_loader_densenet:
    img,_, label_severity = batch
    print("Batch severity label:", label_severity)

# -------------------------------------------------
# Dataset and DataLoader Setup for Combined Model
# -------------------------------------------------

# Dataset with the appropriate transform
dataset_combined = FundusDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
# Prepare binary and severity dataset for combined model
features_combined, label_dr_combined, label_severity_combined = [], [], []
for img, label_dr, label_severity in dataset_combined:
    features_combined.append(img.numpy().flatten())
    label_dr_combined.append(label_dr)
    label_severity_combined.append(label_severity)

# Oversample both binary and severity label
ros = RandomOverSampler(random_state=42)
X_resampled_combined, y_resampled_dr_combined = ros.fit_resample(features_combined, label_dr_combined)
_, y_resampled_severity_combined = ros.fit_resample(features_combined, label_severity_combined)
# Verify lengths
assert len(X_resampled_combined) == len(y_resampled_dr_combined) == len(y_resampled_severity_combined)
# Create the resampled dataset for binary and severity combined
resampled_dataset_combined = ResampledDataset(
    features=X_resampled_binary,
    label_dr=y_resampled_binary,
    label_severity=y_resampled_severity_combined,
    is_severity_task=False 
)


# Split into train and test sets
train_size_combined = int(0.8 * len(resampled_dataset_combined))
test_size_combined = len(resampled_dataset_combined) - train_size_combined
train_dataset_combined, test_dataset_combined = random_split(resampled_dataset_combined, [train_size_combined, test_size_combined])

# DataLoaders for training and testing combined model
train_loader_combined = DataLoader(train_dataset_combined, batch_size=8, shuffle=True, pin_memory=True)
test_loader_combined = DataLoader(test_dataset_combined, batch_size=8, shuffle=False, pin_memory=True)

# Define the InceptionResNetV2 model
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.conv1 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv2d(32, 32, kernel_size=3)
        self.conv3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.branch0 = nn.MaxPool2d(3, stride=2)
        self.branch1 = BasicConv2d(64, 96, kernel_size=3, stride=2)
        self.branch2 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 64, kernel_size=(7,1), padding=(3,0)),
            BasicConv2d(64, 64, kernel_size=(1,7), padding=(0,3)),
            BasicConv2d(64, 96, kernel_size=3)
        )
        self.branch4 = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.branch5 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x = torch.cat([x0, x1], dim=1)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat([x2, x3], dim=1)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x = torch.cat([x4, x5], dim=1)
        return x

class InceptionResNetA(nn.Module):
    def __init__(self, scale=0.17):
        super(InceptionResNetA, self).__init__()
        self.scale = scale
        self.branch1 = BasicConv2d(384, 32, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(384, 32, kernel_size=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(384, 32, kernel_size=1),
            BasicConv2d(32, 48, kernel_size=3, padding=1),
            BasicConv2d(48, 64, kernel_size=3, padding=1)
        )
        self.conv = nn.Conv2d(128, 384, kernel_size=1)
        self.bn = nn.BatchNorm2d(384, eps=0.001)

    def forward(self, x):
        branch1x1 = self.branch1(x)
        branch2x1 = self.branch2(x)
        branch3x1 = self.branch3(x)
        outputs = [branch1x1, branch2x1, branch3x1]
        x = torch.cat(outputs, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = x * self.scale
        return x + x

class ReductionA(nn.Module):
    def __init__(self):
        super(ReductionA, self).__init__()
        self.branch1 = nn.MaxPool2d(3, stride=2)
        self.branch2 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.branch3 = nn.Sequential(
            BasicConv2d(384, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=3, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

    def forward(self, x):
        x0 = self.branch1(x)
        x1 = self.branch2(x)
        x2 = self.branch3(x)
        x = torch.cat([x0, x1, x2], 1)
        return x

class InceptionResNetB(nn.Module):
    def __init__(self, scale=0.1):
        super(InceptionResNetB, self).__init__()
        self.scale = scale
        self.branch1 = nn.Sequential(
            BasicConv2d(1152, 192, kernel_size=1),
            BasicConv2d(192, 160, kernel_size=(1,7), padding=(0,3)),
            BasicConv2d(160, 192, kernel_size=(7,1), padding=(3,0))
        )
        self.branch2 = BasicConv2d(1152, 192, kernel_size=1)
        self.conv = nn.Conv2d(384, 1152, kernel_size=1)
        self.bn = nn.BatchNorm2d(1152, eps=0.001)

    def forward(self, x):
        branch1x1 = self.branch1(x)
        branch2x1 = self.branch2(x)
        outputs = [branch1x1, branch2x1]
        x = torch.cat(outputs, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = x * self.scale
        return x + x

class ReductionB(nn.Module):
    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch1 = nn.MaxPool2d(3, stride=2)
        self.branch2 = nn.Sequential(
            BasicConv2d(1152, 256, kernel_size=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(1152, 256, kernel_size=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(1152, 256, kernel_size=1),
            BasicConv2d(256, 288, kernel_size=3, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

    def forward(self, x):
        x0 = self.branch1(x)
        x1 = self.branch2(x)
        x2 = self.branch3(x)
        x3 = self.branch4(x)
        x = torch.cat([x0, x1, x2, x3], 1)
        return x

class InceptionResNetC(nn.Module):
    def __init__(self, scale=0.2):
        super(InceptionResNetC, self).__init__()
        self.scale = scale
        self.branch1 = nn.Sequential(
            BasicConv2d(2144, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1,3), padding=(0,1)),
            BasicConv2d(224, 256, kernel_size=(3,1), padding=(1,0))
        )
        self.branch2 = BasicConv2d(2144, 192, kernel_size=1)
        self.conv = nn.Conv2d(448, 2144, kernel_size=1)
        self.bn = nn.BatchNorm2d(2144, eps=0.001)

    def forward(self, x):
        branch1x1 = self.branch1(x)
        branch2x1 = self.branch2(x)
        outputs = [branch1x1, branch2x1]
        x = torch.cat(outputs, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = x * self.scale
        return x + x

class InceptionResNetV2(nn.Module):
    def __init__(self, num_classes=2):  # 2 classes: non-DR and DR
        super(InceptionResNetV2, self).__init__()
        self.stem = Stem()
        self.inception_resnet_a = nn.Sequential(
            InceptionResNetA(),
            InceptionResNetA(),
            InceptionResNetA()
        )
        self.reduction_a = ReductionA()
        self.inception_resnet_b = nn.Sequential(
            InceptionResNetB(),
            InceptionResNetB(),
            InceptionResNetB()
        )
        self.reduction_b = ReductionB()
        self.inception_resnet_c = nn.Sequential(
            InceptionResNetC(),
            InceptionResNetC(),
            InceptionResNetC()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2144, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_resnet_a(x)
        x = self.reduction_a(x)
        x = self.inception_resnet_b(x)
        x = self.reduction_b(x)
        x = self.inception_resnet_c(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#Model Densenet 121
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, drop_rate=0.0):
        super(Bottleneck, self).__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        bottleneck_output = self.conv1(self.relu(self.bn1(x)))
        bottleneck_output = self.conv2(self.relu(self.bn2(bottleneck_output)))
        if self.drop_rate > 0:
            bottleneck_output = F.dropout(bottleneck_output, p=self.drop_rate, training=self.training)
        return torch.cat([x, bottleneck_output], 1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate, drop_rate=0.0):
        super(DenseBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(Bottleneck(in_channels, growth_rate, drop_rate))
            in_channels += growth_rate
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.relu(self.bn(x))
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNet121(nn.Module):
    def __init__(self, num_classes=4):  # 4 classes: severity levels
        super(DenseNet121, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            DenseBlock(64, 6, 32),
            Transition(64 + 6 * 32, 128),
            DenseBlock(128, 12, 32),
            Transition(128 + 12 * 32, 256),
            DenseBlock(256, 24, 32),
            Transition(256 + 24 * 32, 512),
            DenseBlock(512, 16, 32)
        )
        self.fc = nn.Linear(512 + 16 * 32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Define criterion, optimizer, and other hyperparameters
criterion = nn.CrossEntropyLoss()
num_epochs = 10
patience = 5

# Train the binary model (InceptionResNetV2) for DR detection
def train_binary_model(model, dataloader, num_epochs,patience , device=device):
    model.train()
    dataloader=train_loader_inception
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    start_time = time.time()  # Start timing
    print ("Training binary (inception resnetv2) start")

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for i, batch in enumerate(dataloader):
            # Unpack batch; ignore label_severity if present
            for inputs, label_dr, _ in dataloader:
                inputs, label_dr = inputs.to(device), label_dr.to(device)
                optimizer.zero_grad()
                
            # with autocast():
                # Forward pass for binary classification (Non-DR vs DR)
            dr_pred = model(inputs)
                # print("Model Output:", dr_pred)
            #     with autocast(dtype=torch.float32):
            loss = criterion(dr_pred, label_dr)
            # print("Loss:", loss.item())
            # Check if loss is NaN
            if torch.isnan(loss).any():
                print(f"NaN loss encountered at epoch {epoch + 1}, step {i + 1}. Investigating...")
                return
            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, preds = torch.max(dr_pred, 1)
            running_corrects += (preds == label_dr).sum().item()
            total_samples += label_dr.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects / total_samples
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        
        # Early stopping check
        early_stopping(epoch_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if epoch_loss < early_stopping.best_score:
            torch.save(model.state_dict(), "binary_model_best.pth")
    
    end_time = time.time()  # End timing
    print(f'Training completed in {(end_time - start_time) / 60:.2f} minutes') 
    print('Binary model training complete.')

# Prepare data for severity classification
def prepare_data_for_severity(binary_model, dataloader, device):
    """
    Filter data dari binary model dan kembalikan data yang relevan untuk model severity.
    """
    binary_model.eval()  # Set binary model ke evaluasi
    severity_data = []
    
    with torch.no_grad():
        for inputs, label_dr, label_severity in dataloader:
            inputs = inputs.to(device)

            # Prediksi dengan model binary
            dr_preds = torch.argmax(binary_model(inputs), dim=1)

            # Ambil hanya sampel dengan prediksi DR (label 1)
            dr_indices = (dr_preds == 1).nonzero(as_tuple=True)[0]

            if len(dr_indices) > 0:
                filtered_inputs = inputs[dr_indices].cpu()
                filtered_severity_label = label_severity[dr_indices].cpu()
                # Filter valid severity labels (assuming [1, 4] is valid range before adjustment)
                valid_indices = (filtered_severity_label >= 1) & (filtered_severity_label <= 4)
                filtered_inputs = filtered_inputs[valid_indices]
                filtered_severity_label = filtered_severity_label[valid_indices]
                severity_data.extend(zip(filtered_inputs, filtered_severity_label))
    
    print(f"Total severity samples prepared: {len(severity_data)}")
    return severity_data

# Train the severity model (DenseNet121) for DR severity classification
def train_severity_model(model, dataloader, num_epochs,patience, device=device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.1)   
    scaler = GradScaler()
    class_weights = torch.tensor([ 0.0213, 0.0167, 0.0930, 0.2186]).to(device)  # Adjust based on your dataset
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    start_time = time.time()  # Start timing
    print ("Training severity (Densenet 121) start")
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        for i, (inputs, _, label_severity) in  enumerate(dataloader):
            inputs, label_severity = inputs.to(device), label_severity.to(device)
            label_severity -= 1  # Adjust labels to [0, 3] range for CrossEntropyLoss
            # assert torch.all(label_saverity > 0),f"Found label 0 in batch {i + 1}"
            # print(f"label: min={label.min().item()}, max={label.max().item()}")
            # assert torch.all(label >= 1) and torch.all(label <= 4), "label berada di luar rentang [1, 4]!"
            assert torch.all(label_severity >= 0) and torch.all(label_severity < 4), f"Invalid label: {label_severity}"
            optimizer.zero_grad()
            # print(np.unique(label.cpu().numpy()))  # Convert tensor to NumPy array before using np.unique
            # print(f"label: {label}")  # Check the label in each batch
            with autocast(), torch.autograd.detect_anomaly():
                # Forward pass for severity classification (classes 1 to 4)
                severity_pred = model(inputs)
                loss = criterion(severity_pred, label_severity)  # label are already in [1, 2, 3, 4]
            if torch.isnan(severity_pred).any() or torch.isinf(severity_pred).any():
                print(f"NaN or Inf detected in model output!")
                break
            if torch.isnan(loss).any():
                print(f"NaN loss encountered at epoch {epoch + 1}, step {i + 1}. Investigating...")
                return
            # Backward pass and optimization
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            _, preds = torch.max(severity_pred, 1)
            # preds += 1 
            running_corrects += (preds == label_severity).sum().item()
            total_samples += label_severity.size(0)
        epoch_loss = running_loss / len(severity_data)
        epoch_acc = running_corrects.double() / total_samples
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        # Scheduler step
        scheduler.step()    
        # Early stopping check
        early_stopping(epoch_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if epoch_loss < early_stopping.best_score:
            torch.save(model.state_dict(), "severity_model_best.pth")

    end_time = time.time()  # End timing
    print(f'Training completed in {(end_time - start_time) / 60:.2f} minutes')      
    print('Severity model training complete.')

class CombinedInferenceModel(nn.Module):
    def __init__(self, binary_model, severity_model):
        """
        Menggabungkan model binary dan severity ke dalam pipeline inference.
        
        Parameters:
            binary_model: Model untuk klasifikasi DR/Non-DR.
            severity_model: Model untuk klasifikasi tingkat keparahan DR.
        """
        super(CombinedInferenceModel, self).__init__()
        self.binary_model = binary_model
        self.severity_model = severity_model

    def forward(self, x):
        # Step 1: Prediksi DR/Non-DR
        dr_pred = self.binary_model(x)
        dr_class = torch.argmax(dr_pred, dim=1)  # Prediksi kelas DR/Non-DR

        # Step 2: Filter DR-positif gambar untuk klasifikasi severity
        mask = (dr_class > 0)  # Ambil hanya DR-positif
        
        dr_images = x[mask]

        severity_pred = None
        if dr_images.size(0) > 0:
            # Step 3: Prediksi tingkat keparahan untuk gambar DR-positif
            severity_pred = self.severity_model(dr_images)
        else:
            severity_pred = torch.tensor([], device=x.device)  # Kosongkan tensor untuk konsistensi

        return dr_pred, severity_pred


def train_combined_model(model, dataloader, num_epochs, patience=5, device='cuda'):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # LR lebih stabil
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct_dr = 0
        epoch_correct_severity = 0
        total_samples_dr = 0
        total_samples_severity = 0

        for i, (inputs, label_dr, label_severity) in enumerate(dataloader):
            inputs, label_dr, label_severity = (
                inputs.to(device),
                label_dr.to(device),
                label_severity.to(device),
            )

            optimizer.zero_grad()

            with autocast():
                dr_pred, severity_pred = model(inputs)

                # DR Loss and Accuracy
                loss_dr = criterion(dr_pred, label_dr)
                _, dr_preds = torch.max(dr_pred, dim=1)
                correct_dr = torch.sum(dr_preds == label_dr).item()
                total_samples_dr += label_dr.size(0)

                # Severity Loss and Accuracy
                loss_severity = 0.0
                if severity_pred is not None and severity_pred.size(0) > 0:
                    mask = (label_dr > 0)
                    severity_label = label_severity[mask]
                    severity_pred = severity_pred[mask]

                    if severity_label.size(0) > 0:
                        loss_severity = criterion(severity_pred, severity_label)
                        _, severity_preds = torch.max(severity_pred, dim=1)
                        correct_severity = torch.sum(severity_preds == severity_label).item()
                        epoch_correct_severity += correct_severity
                        total_samples_severity += severity_label.size(0)

                # Total Loss
                total_loss = loss_dr + loss_severity

            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            epoch_correct_dr += correct_dr

        # Epoch Metrics
        avg_loss = epoch_loss / len(dataloader)
        dr_accuracy = epoch_correct_dr / total_samples_dr if total_samples_dr > 0 else 0.0
        severity_accuracy = epoch_correct_severity / total_samples_severity if total_samples_severity > 0 else 0.0

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, "
            f"DR Accuracy: {dr_accuracy:.4f}, Severity Accuracy: {severity_accuracy:.4f}"
        )

        early_stopping(avg_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("Training completed.")


def evaluate_combined_model(model, dataloader, device='cuda'):
    model.eval()
    true_label_dr, pred_label_dr = [], []
    true_label_severity, pred_label_severity = [], []

    with torch.no_grad():
        for inputs, label_dr, label_severity in dataloader:
            inputs, label_dr, label_severity = (
                inputs.to(device),
                label_dr.to(device),
                label_severity.to(device),
            )

            dr_pred, severity_pred = model(inputs)
            dr_preds = torch.argmax(dr_pred, dim=1)

            true_label_dr.extend(label_dr.cpu().numpy())
            pred_label_dr.extend(dr_preds.cpu().numpy())

            if severity_pred is not None and severity_pred.size(0) > 0:
                mask = (dr_preds == 1)
                severity_preds = torch.argmax(severity_pred, dim=1)
                true_label_severity.extend(label_severity[mask].cpu().numpy())
                pred_label_severity.extend(severity_preds.cpu().numpy())

    # Calculate Metrics
    dr_accuracy = accuracy_score(true_label_dr, pred_label_dr)
    dr_f1 = f1_score(true_label_dr, pred_label_dr, average='weighted')

    if true_label_severity:
        severity_accuracy = accuracy_score(true_label_severity, pred_label_severity)
        severity_f1 = f1_score(true_label_severity, pred_label_severity, average='weighted')
        print(f"Severity Accuracy: {severity_accuracy:.4f}, Severity F1: {severity_f1:.4f}")

    print(f"DR Accuracy: {dr_accuracy:.4f}, DR F1: {dr_f1:.4f}")
    if not true_label_severity:
        print("No DR-positive samples found for severity evaluation.")

# Train both models
binary_model = InceptionResNetV2(num_classes=2).to(device)
severity_model = DenseNet121(num_classes=4).to(device)

# Latih binary model
train_binary_model(binary_model, train_loader_inception, num_epochs=num_epochs, patience=patience)
torch.save(binary_model.state_dict(), "binary_model_best.pth")  # Simpan model binary
# Filter data untuk model severity
severity_data = prepare_data_for_severity(binary_model, train_loader_inception, device)
severity_dataloader = DataLoader(
    severity_data, batch_size=8, shuffle=True, pin_memory=True
)
# Latih severity model
train_severity_model(severity_model, severity_dataloader, num_epochs=num_epochs, patience=patience)
torch.save(severity_model.state_dict(), "severity_model_best.pth")  # Simpan model severity
# binary_model.eval()  # Set model ke mode evaluasi
# severity_model.eval()
# Gabungkan kedua model untuk inferensi
combined_model = CombinedInferenceModel(binary_model, severity_model).to(device)

# Evaluasi gabungan model
evaluate_combined_model(combined_model, test_loader_combined, device=device)

# Evaluation functions can similarly be split to test each model separately.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluate Binary DR Classification Model
def evaluate_binary_model(model, dataloader, device=device):
    model.eval()
    true_label = []
    pred_label = []
    
    with torch.no_grad():
        for inputs, label, _ in dataloader:
            inputs, label = inputs.to(device), label.to(device)
            
            # Binary DR classification prediction
            dr_pred = model.inception_resnet_v2(inputs)
            dr_preds = torch.argmax(dr_pred, dim=1)  # Get the predicted DR class (0 or 1)
            
            true_label.extend(label.cpu().numpy())
            pred_label.extend(dr_preds.cpu().numpy())
    
    # Calculate metrics for the binary classification
    accuracy = accuracy_score(true_label, pred_label)
    precision = precision_score(true_label, pred_label, average='weighted')
    recall = recall_score(true_label, pred_label, average='weighted')
    f1 = f1_score(true_label, pred_label, average='weighted')
    
    print(f'Binary DR Classification - Accuracy: {accuracy:.4f}')
    print(f'Binary DR Classification - Precision: {precision:.4f}')
    print(f'Binary DR Classification - Recall: {recall:.4f}')
    print(f'Binary DR Classification - F1 Score: {f1:.4f}')


# Evaluate Severity Classification Model
def evaluate_severity_model(model, dataloader, device=device):
    model.eval()
    true_label = []
    pred_label = []
    
    with torch.no_grad():
        for inputs, label, _ in dataloader:
            inputs, label = inputs.to(device), label.to(device)
            
            # Binary DR classification first to get DR predictions
            dr_pred = model.inception_resnet_v2(inputs)
            dr_preds = torch.argmax(dr_pred, dim=1)  # DR or Non-DR
            
            # Only evaluate severity for DR images
            mask = (dr_preds == 1)  # DR images only (class 1)
            dr_images = inputs[mask]
            severity_label = label[mask]
            
            if dr_images.size(0) > 0:
                # Severity classification only on DR images
                severity_pred = model.densenet121(dr_images)
                severity_preds = torch.argmax(severity_pred, dim=1)
                
                true_label.extend(severity_label.cpu().numpy())
                pred_label.extend(severity_preds.cpu().numpy())
    
    if len(true_label) > 0:
        # Calculate metrics for the severity classification
        accuracy = accuracy_score(true_label, pred_label)
        precision = precision_score(true_label, pred_label, average='weighted')
        recall = recall_score(true_label, pred_label, average='weighted')
        f1 = f1_score(true_label, pred_label, average='weighted')
        
        print(f'Severity Classification - Accuracy: {accuracy:.4f}')
        print(f'Severity Classification - Precision: {precision:.4f}')
        print(f'Severity Classification - Recall: {recall:.4f}')
        print(f'Severity Classification - F1 Score: {f1:.4f}')
    else:
        print('No DR images were present in the batch for severity evaluation.')


# Save model
torch.save(combined_model.state_dict(), 'combined_model.pth')

# Load model
model = CombinedInferenceModel(num_classes_dr=2, num_classes_severity=4)
model.load_state_dict(torch.load('combined_model.pth'))
model.to(device)