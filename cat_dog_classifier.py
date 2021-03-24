import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
# data file
data_dir = 'C:/Users/franc/Documents/Dataset/cats_and_dogs'

# Load model, use pretrained ResNet 50
model = models.resnet50(pretrained=True)

#print(model)

# Use CPU if no GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modify only on the last layer
for param in model.parameters():
    # freeze weights
    # requires_grad is an attribute if a tensor requires gradients
    param.requires_grad = False
    
# Set up fully connected layer
# nn.Sequential() apply new model/layer in sequence to previous model
# Linear(input dim, hidden dim, bias)
# Dropout(p), randomly zeros some input tensor elements with probability p. Default p=0.5
#   PyTorch mannual Dropout():
#   "proven to be an effective technique for regularization and preventing the co-adaptation of neurons" 
# ReLu() and LogSoftmax() are activation functions
#   LogSoftmax() computes the normalized exponential function of all the elements in the layer
#   describes the probability of the neural network that a particular sample belongs to a certain class


# Tensor size varies for layer input
# 1d: [batch size], for labels and predictions
# 2d: [batch size, num_features], Linear() input
# 3d: [batch size, channels, num_features], Conv1d() input; [seq_len, batch_size, num_features], RNN input
# 4d: [batch_size, channels, height, width], Conv2d() input
# 5d: [batch_size, channels, depth, height, width], Con3d() input
#   batch_size: num of samples through the network at each run.
#   ususally < num of all samples, for less memeory usage and frequent update in parameters at each run
#   batch_size == 1 is considered as stochastic
#   channels: 1 for B&W; 3 for RGB; 4 for transparency
#       out_channels, number of channels produced by the convolution
# view(), resize tensor, use -1 if use PyTorch to figure out

model.fc = nn.Sequential(nn.Linear(2048, 512), # for ResNet50, use 2048 as input dim
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 2),
                                 nn.LogSoftmax(dim=1))

# Negative Log Likelihood Loss
# at high probability (closer to 1), the loss is low and vice versa
# Often used after LogSoftmax()
# See: https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/
criterion = nn.NLLLoss()

# DL Optimizers - reduce loss via changing attributes like weights and learning rates
# - Gradient Descent
#   uses the whole data to updates weights, slow and computationally expensive
# - SGD, Stochastic Gradient Descent with momentum
#   uses single data point to updates weights, noisy
#   momentum, present gradient depends on previous gradient
# - Adagrad, Adaptive Gradient Algorithm
#   lr continuously decreases as summation of gradients increase
# - Adadelta
#   lr changes based on summation of some of previous gradients (not all of)
# - Adam, Adaptive Momemt Estimation
#   SGD w/ momentum + Adadelta
#   computationally efficient and requires less memory
#   See: https://towardsdatascience.com/deep-learning-optimizers-436171c9e23f
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

model.to(device)
#print(model)
#print(device)

# Load and preprocess data
train_transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])