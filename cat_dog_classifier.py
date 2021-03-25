# Cat Dog Classfier ussing pre-trained ResNet50

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

#################################
# ---- Variables For Tweak ---- #
#################################
# data file
# for image classification, put images from different categories into corresponding folders
train_dir = 'C:/Users/franc/Documents/Dataset/cats_and_dogs/test'
test_dir = 'C:/Users/franc/Documents/Dataset/cats_and_dogs/test'

# categories
classes = ('cat', 'dog')

BATCH_SIZE = 32
EPOCHES = 1



#########################
# ---- Set up Data ---- #
#########################

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
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor()])

# Load data and normalize with transform function
image_train_set = datasets.ImageFolder(root=train_dir, transform=image_train_trans)
image_test_set = datasets.ImageFolder(root=test_dir, transform=image_train_trans)

# --TEST-- #
#print(image_train_set.classes) #show class names sorted alphabetically
#print(image_train_set.class_to_idx) #show dict(class_name, class_index)
#print(image_train_set.imgs) #show dict(image_path, class_index)


# Forward normalized data into data loader
image_train_loader = torch.utils.data.DataLoader(image_train_set, batch_size=BATCH_SIZE, shuffle=True)
image_test_loader = torch.utils.data.DataLoader(image_test_set, batch_size=BATCH_SIZE, shuffle=True)

# --TEST-- #
# show the first image in image_loader
# permute() rearranges the dimensions of the tensor
# in this case, rearranges C*H*W into H*W*C
#data_iter = iter(image_train_loader)
#images, labels = next(data_iter)
#print(images.shape)
#for i in range(4):
#plt.imshow(images[i].permute(1,2,0))
#    plt.axis('off')
#plt.show()


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

for e in range(EPOCHES):  #loop over our data iterator

    running_loss = 0.0

    # for count, value in enumerate(iterable, start = 0)
    for i, data in enumerate(image_train_loader):
        # data, a list of [inputs, labels]
        inputs, labels = data
        # assure data and model are on the same device
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        # PyTorch accumulates the gradients on subsequent backward passes
        optimizer.zero_grad()

        # forward run
        outputs = model(inputs)
        # compute loss
        loss = criterion(outputs, labels)
        # backward run
        loss.backward()

        # optimize, updates the parameters
        optimizer.step()

        # print stats
        # item() converts loss tensor to python float32
        running_loss += loss.item()

        if i % 20 == 19:    # print avg running_loss every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (e + 1, i + 1, running_loss / 20)) # +1 for easy reading
            
            # reset running_loss
            running_loss = 0.0

#print('Training Complete')

# save model

#save_dir = ''
#torch.save(net.state_dict(), save_dir)


#########################################
# ---- Test Model with Test Image  ---- #
#########################################

image_test_iter = iter(image_test_loader)
images, labels = next(image_test_iter)
images = images.to(device)

test_outputs = model(images)
max_value, predict_value = torch.max(test_outputs, 1)

print(images.shape)

# print images
#plt.figure(figsize=[, height])
imglist = [images[0], images[1], images[2], images[3]]
plt.imshow(np.transpose(torchvision.utils.make_grid(imglist).numpy(), (1,2,0)))
#plt.axis('off')
plt.show()

# print results
print('Labels: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

print('Predicted: ', ' '.join('%5s' % classes[predict_value[k]] for k in range(4))) 
