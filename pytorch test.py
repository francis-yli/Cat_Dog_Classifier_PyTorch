import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

m = nn.Dropout(p=0.2)
input = torch.randn(2,3)
output = m(input)
print(input)
print(input.size())
print(output)
print(output.size())