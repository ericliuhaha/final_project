import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
import torchvision.transforms as transforms
from PIL import Image


class EyeCNN(nn.Module):
    def __init__(self):
        super(EyeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


'''
Label of 'Close' : 0
Label of 'Open' : 1
'''


# 影像轉換 (Data Augmentation)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_dataloaders(data_dir, batch_size):
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
epochs = 5
learning_rate = 0.0001
data_dir = "./blink_detect_CNN/eye_dataset/data"
model_path = "eye_state_cnn.pth"
print(device)


'''
model_path="eye_state_cnn.pth"
model = EyeCNN().to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
#model.load_state_dict(torch.load(model_path))
model.eval()'
'''