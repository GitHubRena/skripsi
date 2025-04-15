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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report

# Import EarlyStopping from a separate file
from early_stopping import EarlyStopping

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset Class
class FundusDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, debug=False, max_debug_images=5):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.debug = debug
        self.max_debug_images = max_debug_images  # Limit to display only a few images for debugging
        self.debug_count = 0

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
       
        # Extract the green channel first ("terlihat hijau" 
        # (tetapi sebenarnya seluruh citra akan terlihat abu-abu karena semua saluran memiliki nilai yang sama))
        green_channel = image_rgb[:, :, 1]
        green_channel_image = np.stack((green_channel, green_channel, green_channel), axis=-1)
        
        # Apply CLAHE and sharpening
        image_clahe = self._apply_clahe(green_channel_image)
        image_sharpened = self._apply_sharpening(image_clahe)
        
        # Apply super-resolution and transformation
        # image_super_res = self._apply_super_resolution(image_sharpened)
        # Debug: Display the image before applying the transform (after prapemrosesan)
        if self.debug and self.debug_count < self.max_debug_images:
            # Convert tensor to numpy array and display it
            image_debug = image_sharpened.astype(np.uint8)  # Convert to uint8 if necessary
            plt.imshow(image_debug)
            plt.title(f"Debug Image {self.debug_count + 1}")
            plt.axis("off")  # Hide axes
            plt.show()
            self.debug_count += 1

        if self.transform:
            image_sharpened = self.transform(image_rgb)
        
        label = self.labels_df.iloc[idx, 1]
        filename = self.labels_df.iloc[idx, 0]

        return image_sharpened, label, filename

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

    # def _apply_super_resolution(self, img):
    #     return cv2.resize(img, (299, 299), interpolation=cv2.INTER_CUBIC)

# Data Preparation
def calculate_mean_std(dataset):
    mean = 0.0
    std = 0.0
    num_samples = len(dataset)
    
    for img, _, _ in dataset:
        img = img.numpy()  # Convert tensor to numpy array
        mean += img.mean(axis=(1, 2))
        std += img.std(axis=(1, 2))

    mean /= num_samples
    std /= num_samples
    return mean, std

# Initialize dataset with a placeholder transform
temp_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])

dataset = FundusDataset(
    csv_file = r'C:\Users\renat\OneDrive\Documents\skripsi\Code\images_id_kelas.csv',  # Adjust with your CSV file path
    img_dir = r'C:\Users\renat\OneDrive\Documents\skripsi\Code\HasilCrop299',  # Adjust with your image directory
    transform=temp_transform
)

mean, std = calculate_mean_std(dataset)

# Define actual transform with calculated mean and std
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
    
])

# Initialize dataset again with updated transform
dataset = FundusDataset(
    csv_file = r'C:\Users\renat\OneDrive\Documents\skripsi\Code\images_id_kelas.csv',  
    img_dir = r'C:\Users\renat\OneDrive\Documents\skripsi\Code\HasilCrop299',  
    transform=transform,
    debug=True
)

# Extract features and labels
features, labels = [], []
for img, label, _ in dataset:
    features.append(img.numpy().flatten())  # Flatten the image to a 1D array
    labels.append(label)

X = np.array(features)
y = np.array(labels)

# Apply RandomOverSampler
ros = RandomOverSampler(random_state=42)
# X_subset = X[:100]  # Gunakan 1000 sampel pertama
# y_subset = y[:100]
X_resampled, y_resampled = ros.fit_resample(X, y)

# Reconstruct the resampled dataset
resampled_features = [x.reshape(3, 299, 299) for x in X_resampled]
resampled_dataset = list(zip(resampled_features, y_resampled))

# DataLoader for resampled dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Retrieve the image and label from the dataset
        img, label = self.data[index]  # Assuming each entry in data is a (img, label) tuple
        
        # Check if img is already a tensor
        if isinstance(img, np.ndarray):
            img_tensor = torch.from_numpy(img).float().clone().detach()
        else:
            img_tensor = img.float().clone().detach()
        
        # Directly clone and detach the label if it’s already a tensor
        if isinstance(label, torch.Tensor):
            label_tensor = label.clone().detach().long()
        else:
            label_tensor = torch.tensor(label).long()

        return img_tensor, label_tensor
    
# Now resampled_dataset should be a list of (image, label) tuples
resampled_dataset = CustomDataset(resampled_dataset)
# Split dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    resampled_features, y_resampled, test_size=0.2, random_state=42
    )
# Create DataLoaders for training and testing sets
train_dataset = CustomDataset(list(zip(train_data, train_labels)))
test_dataset = CustomDataset(list(zip(test_data, test_labels)))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=0)
from collections import Counter

# # Cek distribusi label setelah oversampling
# print("Distribusi setelah oversampling:", Counter(y_resampled))

# # Cek distribusi label pada train dan test set
# print("Distribusi training set:", Counter(train_labels))
# print("Distribusi testing set:", Counter(test_labels))

# # Total data
# print("Total data training:", len(train_labels))
# print("Total data testing:", len(test_labels))


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
    def __init__(self,debug = False):
        super(Stem, self).__init__()
        self.debug = debug
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
        if self.debug:
            self._debug_display(x, "Input to Stem")
        x = self.conv1(x)
        if self.debug:
            self._debug_display(x, "After Conv1")
        x = self.conv2(x)
        if self.debug:
            self._debug_display(x, "After Conv2")
        x = self.conv3(x)
        if self.debug:
            self._debug_display(x, "After Conv3")

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x = torch.cat([x0, x1], dim=1)
        if self.debug:
            self._debug_display(x, "After First Branch Merge")

        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat([x2, x3], dim=1)
        if self.debug:
            self._debug_display(x, "After Second Branch Merge")

        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x = torch.cat([x4, x5], dim=1)
        if self.debug:
            self._debug_display(x, "Output of Stem")

        return x
    
    def _debug_display(self, x, title):
        """Utility function to display tensors as images."""
        # Convert the tensor to numpy array
        if self.debug:
            img = x[0].cpu().detach().numpy()  # Select the first image in the batch
        # Rearrange the dimensions (C, H, W) to (H, W, C)
        # If the image is in (C, H, W) format, we need to convert it to (H, W, C)
        if len(img.shape) == 3 and img.shape[2] > 1:  # RGB Image
            img = np.mean(img, axis=2)  # Convert from (C, H, W) to (H, W, C)

        # If the image has more than 1 channel (RGB), normalize it
        img = np.clip(img, 0, 1)
        
        plt.imshow(img,cmap='gray')
        plt.title(title)
        shape_info = f"Shape: {img.shape}\nChannels: {img.shape[0] if img.ndim == 3 else 1}"
        plt.gcf().text(0.75, 0.25, shape_info, fontsize=12, va='center', ha='left', transform=plt.gcf().transFigure)

        plt.show()
        input("Tekan Enter untuk keluar...")


class InceptionResNetA(nn.Module):
    def __init__(self, debug=False, scale=0.17):
        super(InceptionResNetA, self).__init__()
        self.scale = scale
        self.debug = debug
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
        if self.debug:
            self._debug_display(x, "Input to InceptionResNetA")
        
        residual = x
        branch1x1 = self.branch1(x)
        if self.debug:
            self._debug_display(branch1x1, "Branch 1x1 Output")
        
        branch2x1 = self.branch2(x)
        if self.debug:
            self._debug_display(branch2x1, "Branch 3x3 Output")
        
        branch3x1 = self.branch3(x)
        if self.debug:
            self._debug_display(branch3x1, "Branch 3x3x3 Output")
        
        outputs = [branch1x1, branch2x1, branch3x1]
        x = torch.cat(outputs, 1)
        if self.debug:
            self._debug_display(x, "After Concatenation")
        
        x = self.conv(x)
        x = self.bn(x)
        x = x * self.scale
        if self.debug:
            self._debug_display(x, "After Conv and Scaling")
        
        return x + residual

    def _debug_display(self, x, title):
        """Utility function to display tensors as images."""
        # Convert the tensor to numpy array
        if self.debug:
            img = x[0].cpu().detach().numpy()  # Select the first image in the batch
        # Rearrange the dimensions (C, H, W) to (H, W, C)
        # If the image is in (C, H, W) format, we need to convert it to (H, W, C)
        if len(img.shape) == 3 and img.shape[2] > 1:  # RGB Image
            img = np.mean(img, axis=2)  # Convert from (C, H, W) to (H, W, C)

        # If the image has more than 1 channel (RGB), normalize it
        img = np.clip(img, 0, 1)
        
        plt.imshow(img,cmap='gray')
        plt.title(title)
        shape_info = f"Shape: {img.shape}\nChannels: {img.shape[0] if img.ndim == 3 else 1}"
        plt.gcf().text(0.75, 0.25, shape_info, fontsize=12, va='center', ha='left', transform=plt.gcf().transFigure)

        plt.show()
        input("Tekan Enter untuk keluar...")


class ReductionA(nn.Module):
    def __init__(self, debug=False):
        super(ReductionA, self).__init__()
        self.debug = debug
        self.branch1 = nn.MaxPool2d(3, stride=2)
        self.branch2 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.branch3 = nn.Sequential(
            BasicConv2d(384, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=3, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

    def forward(self, x):
        if self.debug:
            self._debug_display(x, "Input to ReductionA")
        
        x0 = self.branch1(x)
        if self.debug:
            self._debug_display(x0, "Branch 1 Output")
        
        x1 = self.branch2(x)
        if self.debug:
            self._debug_display(x1, "Branch 2 Output")
        
        x2 = self.branch3(x)
        if self.debug:
            self._debug_display(x2, "Branch 3 Output")
        
        x = torch.cat([x0, x1, x2], 1)
        if self.debug:
            self._debug_display(x, "After Concatenation in ReductionA")
        
        return x

    def _debug_display(self, x, title):
        """Utility function to display tensors as images."""
        # Convert the tensor to numpy array
        if self.debug:
            img = x[0].cpu().detach().numpy()  # Select the first image in the batch
        # Rearrange the dimensions (C, H, W) to (H, W, C)
        # If the image is in (C, H, W) format, we need to convert it to (H, W, C)
        if len(img.shape) == 3 and img.shape[2] > 1:  # RGB Image
            img = np.mean(img, axis=2)  # Convert from (C, H, W) to (H, W, C)

        # If the image has more than 1 channel (RGB), normalize it
        img = np.clip(img, 0, 1)
        
        plt.imshow(img,cmap='gray')
        plt.title(title)
        shape_info = f"Shape: {img.shape}\nChannels: {img.shape[0] if img.ndim == 3 else 1}"
        plt.gcf().text(0.75, 0.25, shape_info, fontsize=12, va='center', ha='left', transform=plt.gcf().transFigure)

        plt.show()
        input("Tekan Enter untuk keluar...")

class InceptionResNetB(nn.Module):
    def __init__(self, scale=0.1, debug=False):
        super(InceptionResNetB, self).__init__()
        self.scale = scale
        self.debug = debug

        self.branch1 = nn.Sequential(
            BasicConv2d(1152, 192, kernel_size=1),
            BasicConv2d(192, 160, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), padding=(3, 0))
        )
        self.branch2 = BasicConv2d(1152, 192, kernel_size=1)
        self.conv = nn.Conv2d(384, 1152, kernel_size=1)
        self.bn = nn.BatchNorm2d(1152, eps=0.001)

    def forward(self, x):
        if self.debug:
            self._debug_display(x, "Input to InceptionResNetB")

        residual = x
        branch1x1 = self.branch1(x)
        if self.debug:
            self._debug_display(branch1x1, "Branch1 Output")

        branch2x1 = self.branch2(x)
        if self.debug:
            self._debug_display(branch2x1, "Branch2 Output")

        outputs = [branch1x1, branch2x1]
        x = torch.cat(outputs, 1)
        if self.debug:
            self._debug_display(x, "Concatenated Output")

        x = self.conv(x)
        x = self.bn(x)
        x = x * self.scale

        if self.debug:
            self._debug_display(x, "Final Output")
        
        return x + residual

    def _debug_display(self, x, title):
        """Utility function to display tensors as images."""
        # Convert the tensor to numpy array
        if self.debug:
            img = x[0].cpu().detach().numpy()  # Select the first image in the batch
        # Rearrange the dimensions (C, H, W) to (H, W, C)
        # If the image is in (C, H, W) format, we need to convert it to (H, W, C)
        if len(img.shape) == 3 and img.shape[2] > 1:  # RGB Image
            img = np.mean(img, axis=2)  # Convert from (C, H, W) to (H, W, C)

        # If the image has more than 1 channel (RGB), normalize it
        img = np.clip(img, 0, 1)
        
        plt.imshow(img,cmap='gray')
        plt.title(title)
        shape_info = f"Shape: {img.shape}\nChannels: {img.shape[0] if img.ndim == 3 else 1}"
        plt.gcf().text(0.75, 0.25, shape_info, fontsize=12, va='center', ha='left', transform=plt.gcf().transFigure)

        plt.show()
        input("Tekan Enter untuk keluar...")

class ReductionB(nn.Module):
    def __init__(self, debug=False):
        super(ReductionB, self).__init__()
        self.debug = debug

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
        if self.debug:
            self._debug_display(x, "Input to ReductionB")

        x0 = self.branch1(x)
        x1 = self.branch2(x)
        x2 = self.branch3(x)
        x3 = self.branch4(x)

        if self.debug:
            self._debug_display(x0, "Branch1 Output")
            self._debug_display(x1, "Branch2 Output")
            self._debug_display(x2, "Branch3 Output")
            self._debug_display(x3, "Branch4 Output")

        x = torch.cat([x0, x1, x2, x3], 1)
        if self.debug:
            self._debug_display(x, "Concatenated Output")

        return x

    def _debug_display(self, x, title):
        """Utility function to display tensors as images."""
        # Convert the tensor to numpy array
        if self.debug:
            img = x[0].cpu().detach().numpy()  # Select the first image in the batch
        # Rearrange the dimensions (C, H, W) to (H, W, C)
        # If the image is in (C, H, W) format, we need to convert it to (H, W, C)
        if len(img.shape) == 3 and img.shape[2] > 1:  # RGB Image
            img = np.mean(img, axis=2)  # Convert from (C, H, W) to (H, W, C)

        # If the image has more than 1 channel (RGB), normalize it
        img = np.clip(img, 0, 1)
        
        plt.imshow(img,cmap='gray')
        plt.title(title)
        shape_info = f"Shape: {img.shape}\nChannels: {img.shape[0] if img.ndim == 3 else 1}"
        plt.gcf().text(0.75, 0.25, shape_info, fontsize=12, va='center', ha='left', transform=plt.gcf().transFigure)

        plt.show()
        input("Tekan Enter untuk keluar...")

class InceptionResNetC(nn.Module):
    def __init__(self, scale=0.2, debug=False):
        super(InceptionResNetC, self).__init__()
        self.scale = scale
        self.debug = debug

        self.branch1 = nn.Sequential(
            BasicConv2d(2144, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), padding=(1, 0))
        )
        self.branch2 = BasicConv2d(2144, 192, kernel_size=1)
        self.conv = nn.Conv2d(448, 2144, kernel_size=1)
        self.bn = nn.BatchNorm2d(2144, eps=0.001)

    def forward(self, x):
        if self.debug:
            self._debug_display(x, "Input to InceptionResNetC")

        residual = x
        branch1x1 = self.branch1(x)
        if self.debug:
            self._debug_display(branch1x1, "Branch1 Output")

        branch2x1 = self.branch2(x)
        if self.debug:
            self._debug_display(branch2x1, "Branch2 Output")

        outputs = [branch1x1, branch2x1]
        x = torch.cat(outputs, 1)
        if self.debug:
            self._debug_display(x, "Concatenated Output")

        x = self.conv(x)
        x = self.bn(x)
        x = x * self.scale

        if self.debug:
            self._debug_display(x, "Final Output")

        return x + residual

    def _debug_display(self, x, title):
        """Utility function to display tensors as images."""
        # Convert the tensor to numpy array
        if self.debug:
            img = x[0].cpu().detach().numpy()  # Select the first image in the batch
        # Rearrange the dimensions (C, H, W) to (H, W, C)
        # If the image is in (C, H, W) format, we need to convert it to (H, W, C)
        if len(img.shape) == 3 and img.shape[2] > 1:  # RGB Image
            img = np.mean(img, axis=2)  # Convert from (C, H, W) to (H, W, C)

        # If the image has more than 1 channel (RGB), normalize it
        img = np.clip(img, 0, 1)
        
        plt.imshow(img,cmap='gray')
        plt.title(title)
        shape_info = f"Shape: {img.shape}\nChannels: {img.shape[0] if img.ndim == 3 else 1}"
        plt.gcf().text(0.75, 0.25, shape_info, fontsize=12, va='center', ha='left', transform=plt.gcf().transFigure)

        plt.show()
        input("Tekan Enter untuk keluar...")
class InceptionResNetV2(nn.Module):
    def __init__(self, num_classes=5,debug=False):
        super(InceptionResNetV2, self).__init__()
        self.debug = debug 
        self.stem = Stem()
        self.inception_a = nn.Sequential(*[InceptionResNetA () for _ in range(5)])
        self.reduction_a = ReductionA()
        self.inception_b = nn.Sequential(*[InceptionResNetB()for _ in range(10)])
        self.reduction_b = ReductionB()
        self.inception_c = nn.Sequential(*[InceptionResNetC() for _ in range(5)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2144, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        # Debug: Show the output before average pooling and dropout
        if self.debug:
            self._debug_display(x, "Before Average Pooling and Dropout")
        
        # Average pooling and dropout
        x = self.avgpool(x)
        if self.debug:
            self._debug_display(x, "After Average Pooling")

        x = self.dropout(x)
        if self.debug:
            self._debug_display(x, "After Dropout")

        x = torch.flatten(x, 1)
        
        # Debug: Show the output after flattening
        if self.debug:
            self._debug_display(x, "After Flattening")
        
        x = self.fc(x)
        
        # Debug: Show the final output before returning
        if self.debug:
            self._debug_display(x, "Final Output Before FC Layer")
        return x
    def _debug_display(self, x, title):
        """Utility function to display tensors."""
        if x.dim() == 4:  # Batch tensors (N, C, H, W)
            img = x[0].cpu().detach().numpy()  # Pilih gambar pertama di batch
            if img.shape[0] > 3:  # Jika jumlah channel lebih dari 3
                img = img[0]  # Pilih channel pertama
            elif img.shape[0] == 3:  # Jika 3 channel (RGB)
                img = np.transpose(img, (1, 2, 0))  # Convert (C, H, W) ke (H, W, C)
            elif img.shape[0] == 1:  # Jika single-channel (grayscale)
                img = img[0]
            img = (img - img.min()) / (img.max() - img.min())  # Normalisasi ke [0, 1]
            plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
            shape_info = f"Shape: {img.shape}\nChannels: {img.shape[0] if img.ndim == 3 else 1}"
            plt.gcf().text(0.75, 0.25, shape_info, fontsize=12, va='center', ha='left', transform=plt.gcf().transFigure)
            plt.title(title)
            plt.show()
        elif x.dim() == 3:  # Tensor dengan dimensi (C, H, W)
            img = x.cpu().detach().numpy()
            if img.shape[0] > 3:  # Jika jumlah channel lebih dari 3
                img = img[0]  # Pilih channel pertama
            elif img.shape[0] == 3:  # Jika RGB
                img = np.transpose(img, (1, 2, 0))
            elif img.shape[0] == 1:  # Jika grayscale
                img = img[0]
            img = (img - img.min()) / (img.max() - img.min())  # Normalisasi
            plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
            shape_info = f"Shape: {img.shape}\nChannels: {img.shape[0] if img.ndim == 3 else 1}"
            plt.gcf().text(0.75, 0.25, shape_info, fontsize=12, va='center', ha='left', transform=plt.gcf().transFigure)
            plt.title(title)
            plt.show()
        elif x.dim() == 2:  # Tensor 2D
            img = x.cpu().detach().numpy()
            img = (img - img.min()) / (img.max() - img.min())  # Normalisasi
            plt.imshow(img, cmap='gray')
            shape_info = f"Shape: {img.shape}\nChannels: {img.shape[0] if img.ndim == 3 else 1}"
            plt.gcf().text(0.75, 0.25, shape_info, fontsize=12, va='center', ha='left', transform=plt.gcf().transFigure)
            plt.title(title)
            plt.show()
        elif x.dim() == 1:  # Tensor 1D
            plt.figure()
            plt.plot(x.cpu().detach().numpy())
            plt.title(f"{title} (1D Tensor)")
            plt.xlabel("Index")
            plt.ylabel("Value")
            shape_info = f"Shape: {img.shape}\nChannels: {img.shape[0] if img.ndim == 3 else 1}"
            plt.gcf().text(0.75, 0.25, shape_info, fontsize=12, va='center', ha='left', transform=plt.gcf().transFigure)
            plt.show()
        else:
            print(f"{title}: Cannot visualize tensor with shape {x.shape}.")

# Instantiate and train the model
num_classes = len(np.unique(y_resampled))  # Number of unique classes in your dataset
model = InceptionResNetV2(num_classes=num_classes,debug=False).to(device)
# model.stem.debug = True
# Define criterion, optimizer, and other hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 25
patience = 5

# Define training and evaluation functions
def train_model(model, train_loader, criterion, optimizer, num_epochs,patience):
    model.train()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    #scaler = GradScaler(init_scale=1024)  # Initialize GradScaler for mixed precision training
    start_time = time.time()  # Start timing
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            if torch.isnan(loss):
                print(f"Loss is NaN at epoch {epoch+1}. Investigate the issue.")
                print("Labels:", labels)
                print("Model output:", outputs)
                return
            # Scales the loss, calls backward(), and updates the weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    
        epoch_loss = running_loss / len(resampled_dataset)
        epoch_acc = running_corrects.double() / len(resampled_dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
        # Check early stopping condition
        early_stopping(epoch_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    end_time = time.time()  # End timing
    print(f'Training completed in {(end_time - start_time) / 60:.2f} minutes')

def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    all_outputs = []
    start_time = time.time()  # Start timing

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    
    # Binarisasi label untuk perhitungan ROC AUC
    all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4])
    # Calculate metrics
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    specificity = {}
    for i in range(cm.shape[0]):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificity[i] = tn / (tn + fp)
    # Report per kelas (precision, recall, F1)
    report = classification_report(all_labels, all_preds, 
                                   target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
                                   )
    print(report)
    try:
        auc_roc = roc_auc_score(all_labels_bin, all_outputs, multi_class='ovr')
    except:
        auc_roc = "N/A"  # Not available for some cases
    end_time = time.time()  # End timing
    elapsed_time = (end_time - start_time) / 60  # Calculate elapsed time in minutes

    # Print metrics and timing
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Specificity per kelas: {specificity}')
    print(f'Confusion Matrix:\n{cm}')
    print(f'Evaluation completed in {elapsed_time:.2f} minutes')
    print(f'Roc AUC score: {auc_roc}')


if __name__ == "__main__":
    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs, patience=patience)

    # Evaluate the model on the training set
    print("Evaluating on the training set:")
    evaluate_model(model, train_loader)  # Evaluasi pada set pelatihan

    # Evaluate the model on the test set
    print("Evaluating on the test set:")
    evaluate_model(model, test_loader)  # Evaluasi pada set pengujian 