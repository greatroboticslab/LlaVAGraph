# this script has _heavy_ ChatGPT help - thanks!
import torchvision
import torchvision.models as models
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2

resnet = models.resnet18(pretrained=True)

# Modify the last fully connected layer for 3 classes
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 3)  # Replace FC layer to match your 3 classes

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)
print(resnet)
transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
print("Loading...")
train_data = ImageFolder(root="../data/subset/trainData", transform = transforms)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
print("Loaded dataset!")

# Class names
print(train_data.classes)

criterion = nn.CrossEntropyLoss()  # Because you're doing classification
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    resnet.train()
    running_loss = 0.0

    for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

print("Training Complete!")

torch.save(resnet, "resnet_finetuned.pth")

