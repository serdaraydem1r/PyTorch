#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import  transforms
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
train_csv_path = '/Users/fsa/Desktop/data/pytorch/Training_Set/RFMiD_Training_Labels.csv'
test_csv_path = '/Users/fsa/Desktop/data/pytorch/Test_Set/RFMiD_Testing_Labels.csv'
train_images_path = '/Users/fsa/Desktop/data/pytorch/Training_Set/Training'
test_images_path = '/Users/fsa/Desktop/data/pytorch/Test_Set/Test'

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
print(train_df.shape)
print(test_df.shape)

#%%
class RDC_Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
            Args:
                csv_file (string): Path to the csv file with annotations.
                img_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Ensure idx is an integer and properly convert file name to string
        img_name = os.path.join(self.img_dir, str(self.annotations.iloc[idx, 0]))

        # Open image file
        image = Image.open(img_name).convert('RGB')

        # Get the label
        label = int(self.annotations.iloc[idx, 1])

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label
#%%
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Görüntü boyutunu yeniden boyutlandır
    transforms.ToTensor(),        # Görüntüyü tensöre çevir
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize et
])

train_dataset = RDC_Dataset(csv_file=train_csv_path, img_dir=train_images_path, transform= transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = RDC_Dataset(csv_file=test_csv_path, img_dir=test_images_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#%% CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Evrişim katmanları
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout katmanı (%50 dropout oranı)
        self.dropout = nn.Dropout(p=0.5)

        # Flatten katmanı
        self.flatten = nn.Flatten()

        # Fully connected katmanlar
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 46)  # Çıkış katmanı (46 sınıf)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 1. evrişim + ReLU
        x = F.relu(self.conv2(x))  # 2. evrişim + ReLU
        x = self.pool1(x)  # 1. Max pooling

        x = F.relu(self.conv3(x))  # 3. evrişim + ReLU
        x = F.relu(self.conv4(x))  # 4. evrişim + ReLU
        x = self.pool2(x)  # 2. Max pooling

        x = F.relu(self.conv5(x))  # 5. evrişim + ReLU
        x = F.relu(self.conv6(x))  # 6. evrişim + ReLU
        x = self.pool3(x)  # 3. Max pooling

        x = self.flatten(x)  # Tensörü vektöre dönüştür

        x = self.dropout(x)  # Dropout uygulaması
        x = F.relu(self.fc1(x))  # 1. fully connected katman + ReLU
        x = self.dropout(x)  # Dropout uygulaması
        x = self.fc2(x)  # 2. fully connected katman (Çıkış)

        x = F.softmax(x, dim=1)  # Softmax aktivasyonu
        return x
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#%%
epochs = 10
for i in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{i + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
#%%
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total} %')
