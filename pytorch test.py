import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

# Load model, use pretrained ResNet 50
model = models.resnet50(pretrained=True)

print(model)

# Use CPU if no GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modify only on the last layer
for param in model.parameters():
    # freeze weights
    # requires_grad is an attribute if a tensor requires gradients
    param.requires_grad = False
    
# Set up fully connected layer
# nn.Sequential() apply new model/layer in sequence to previous model
model.fc = nn.Sequential(nn.Linear(2048, 512), # Linear(input dim, hidden dim)
                                 nn.ReLU(), # Activation function
                                 nn.Dropout(0.2), #
                                 nn.Linear(512, 10), # Linear(hidden dim, output dim)
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)

print("test")