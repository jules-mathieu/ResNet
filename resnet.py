from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models, transforms
import torch
from data_processing import train_set, val_set, test_set, full_dataset, get_balanced_sampler, ApplyTransform

# Define Transforms first
train_transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data_res = ApplyTransform(train_set, transform=train_transform_resnet)
val_data_res   = ApplyTransform(val_set, transform=train_transform_resnet)
test_data_res  = ApplyTransform(test_set, transform=train_transform_resnet)


batch_size = 32

train_loader_resnet = DataLoader(
    train_data_res, # wrapped data
    batch_size=batch_size, 
    sampler=get_balanced_sampler(train_set), 
    shuffle=False
)

val_loader_resnet   = DataLoader(val_data_res, batch_size=batch_size, shuffle=False)
test_loader_resnet  = DataLoader(test_data_res, batch_size=batch_size, shuffle=False)

# Model
class ResNet18Gray(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

device = torch.device("mps")
num_classes = len(full_dataset.class_to_idx)
model_res = ResNet18Gray(num_classes).to(device)

criterion_res = nn.CrossEntropyLoss()
optimizer_res = torch.optim.Adam(model_res.parameters(), lr=1e-4, weight_decay=1e-4)

