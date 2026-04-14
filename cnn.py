from torch.utils.data import random_split
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm as tqdm
from data_processing import train_set, val_set, test_set, full_dataset, get_balanced_sampler, ApplyTransform


# Data augmentation CNN : 

train_transform_cnn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data_cnn = ApplyTransform(train_set, transform=train_transform_cnn)
val_data_cnn   = ApplyTransform(val_set, transform=train_transform_cnn)
test_data_cnn  = ApplyTransform(test_set, transform=train_transform_cnn)

batch_size = 32

train_loader_cnn = DataLoader(
    train_data_cnn, 
    batch_size=batch_size, 
    sampler=get_balanced_sampler(train_set), 
    shuffle=False
)
val_loader_cnn   = DataLoader(val_data_cnn, batch_size=batch_size, shuffle=False)
test_loader_cnn  = DataLoader(test_data_cnn, batch_size=batch_size, shuffle=False)

num_classes = len(full_dataset.class_to_idx)

class PlainCNN(nn.Module):
    def __init__(self, num_classes=num_classes):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),   # after 3 pools : 224 -> 112 -> 56 -> 28
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.fc_block(x)
        return x


# Replace mps -> cuda if nvidia
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Device:", device)


model_cnn = PlainCNN(num_classes=5).to(device)

criterion_cnn = nn.CrossEntropyLoss()

optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=1e-3)



