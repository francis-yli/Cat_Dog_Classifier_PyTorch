# Cat Dog Classfier ussing pre-trained ResNet50

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

#########################
# ---- Set up Data ---- #
#########################

# data file
# for image classification, put images from different categories into corresponding folders
data_dir = 'C:/Users/franc/Documents/Dataset/cats_and_dogs/test'

# Define data-preprocessing function
# resize image and convert to tensor
# add randomness would help network generalizes features
image_train_trans = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])

image_test_trans = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.ToTensor()])

# Load data and normalize with transform function
image_train_set = datasets.ImageFolder(root=data_dir, transform=image_train_trans)
image_test_set = datasets.ImageFolder(root=data_dir, transform=image_train_trans)

# --TEST-- #
#print(image_train_set.classes) #show class names sorted alphabetically
#print(image_train_set.class_to_idx) #show dict(class_name, class_index)
#print(image_train_set.imgs) #show dict(image_path, class_index)


# Forward normalized data into data loader
image_train_loader = torch.utils.data.DataLoader(image_train_set, batch_size=32, shuffle=True)
image_test_loader = torch.utils.data.DataLoader(image_test_set, batch_size=32, shuffle=True)

# iteration on data
data_iter = iter(image_train_loader)
images, labels = next(data_iter)

# --TEST-- #
# show the first image in image_loader
# permute() rearranges the dimensions of the tensor
# in this case, rearranges C*H*W into H*W*C
#for i in range(4):
#    plt.imshow(images[i].permute(1,2,0))
#    plt.axis('off')
#    plt.show()


##########################
# ---- Set up Model ---- #
##########################

# Load model, use pretrained ResNet 50
model = models.resnet50(pretrained=True)

# --TEST-- #
#print(model)

# Use CPU if no GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modify only on the last layer
for param in model.parameters():
    # freeze weights
    # requires_grad, if a tensor requires gradients
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

# --TEST-- #
#print(model)
#print(device)

################################
# ---- FC Layer Learning  ---- #
################################

# epoch, forward and backward run of all data
# batch size, number of data in one run
# iteration, forward and backward run of [batch] number of data
epoches = 1
steps = 0

train_losses, test_losses = [], []

for e in range(epoches):
    running_loss = 0
    
    for images, labels in image_train_loader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(images)
        
        loss = criterion(logits, labels)
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        
        train_losses.append(running_loss)
        
        if steps % 5 == 0:
            test_loss, accuracy = 0, 0
        
            with torch.no_grad():
                model.eval()

                for images, labels, in image_test_loader:
                    images, labels = images.to(device), labels.to(device)

                    logits = model(images)

                    test_loss += criterion(logits, labels)

                    ps = torch.exp(logits)

                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


            print(f"Epoch: {e+1}/{epoches};"
                  f"Train_loss: {running_loss};"
                  f"Test_loss: {test_loss/len(image_test_loader)};"
                  f"Accuracy: {accuracy/len(image_test_loader)}")
            model.train()
            running_loss = 0

print('Finished Training')
