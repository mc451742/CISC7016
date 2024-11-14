"""
CISC7016 Advanced Topics in Computer Science
Multi-layer Perceptron (MLP)
Skeleton code is sourced from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
Modified by Yumu Xie
"""

# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms

# import matplotlib.pyplot as plt
# import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

# from data_augmentation import SaltAndPepper, addGaussianNoise, darker, brighter, rotate, flip

# function to download and transform datasets
def initialization():
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize
    ])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck') # 10 classifications = 10 labels
    
    return trainloader, testloader, classes

# functions to show an image
# def imshow(img):
#     img = img / 2 + 0.5 # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
# def showimage(trainloader, classes, batch_size):
#     # get some random training images
#     dataiter = iter(trainloader)
#     images, labels = next(dataiter)
#     # show images
#     imshow(torchvision.utils.make_grid(images))
#     # print labels
#     print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# Multi-layer Perceptron (MLP)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 512) # color channels = 3, width = height = 32, 3 * 32 * 32 = 3072
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32, 10) # 10 labels

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.bn2(F.relu(self.fc2(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.bn3(F.relu(self.fc3(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.bn4(F.relu(self.fc4(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.bn5(F.relu(self.fc5(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc6(x)
        return x

# weight initialization function (Xavier (Glorot))  
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

# main function    
def main_mlp():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using {device} device")

    trainloader, testloader, classes = initialization()

    net = Net().to(device)

    net.apply(weights_init)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    mlp_training_loss_list = []
    mlp_testing_loss_list = []

    # main loop
    for epoch in range(36): # loop over the dataset multiple times
        net.train() # set model into training mode
        # training loop
        training_loss = 0.0
        for _, data in enumerate(trainloader, 0):
        # for i, data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad() # reset gradients back to zero
            # forward + backward + optimize
            outputs = net(inputs) # get prediction
            loss = criterion(outputs, labels) # compute loss
            loss.backward() # compute gradients
            optimizer.step() # update parameters
            # print statistics
            training_loss += loss.item()
        epoch_training_loss = training_loss / len(trainloader)
        mlp_training_loss_list.append(epoch_training_loss)
        print(f"epoch {epoch+1}/{36}, training loss: {epoch_training_loss:.4f}")
        net.eval() # set model into evaluation mode
        # testing loop
        testing_loss = 0.0
        # for i, data in trainloader:
        with torch.no_grad():
            for _, data in enumerate(testloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                # forward + backward + optimize
                outputs = net(inputs) # get prediction
                loss = criterion(outputs, labels) # compute loss
                # print statistics
                testing_loss += loss.item()
            epoch_testing_loss = testing_loss / len(testloader)
            mlp_testing_loss_list.append(epoch_testing_loss)
            print(f"epoch {epoch+1}/{36}, testing loss: {epoch_testing_loss:.4f}")

    # print('Finished Training')

    # save model
    PATH = './path/cifar_mlp.pth'
    torch.save(net.state_dict(), PATH)

    # dataiter = iter(testloader)
    # images, labels = next(dataiter)
    # # print images
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    # net = Net() # there is no need to put model into cuda
    # net.load_state_dict(torch.load(PATH, weights_only=True))

    # outputs = net(images)

    # _, predicted = torch.max(outputs, 1)

    # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
    #                           for j in range(4)))

    # load weight
    net = Net().to(device)
    net.load_state_dict(torch.load(PATH, weights_only=True))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        # for _, data in enumerate(testloader, 0):
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        # for _, data in enumerate(testloader, 0):
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    # print('Finished Testing')

    return mlp_training_loss_list, mlp_testing_loss_list

if __name__ == "__main__":
    _, _ = main_mlp()
