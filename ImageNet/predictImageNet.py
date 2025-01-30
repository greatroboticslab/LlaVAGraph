from PIL import Image
# this script has _heavy_ ChatGPT help - thanks!
import torchvision
import torchvision.models as models
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2

resnet = torch.load("resnet_finetuned.pth")

resnet.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = ImageFolder(root="../data/subset/testData", transform = transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Make prediction
with torch.no_grad():
   
    predictions = []

    for images, labels in train_loader:
        predictions = []
        images = images.to(device)

        outputs = resnet(images)
        _, predicted = torch.max(outputs, 1)
        predictions.append(predicted)
        print(f"Predicted class: {predictions}")
        print(f"Actual labels: {labels}")
