import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt

# Download two kind of data. 
train = datasets.MNIST(
    "example data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

test = datasets.MNIST(
    "example data",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

# Data resume
print("DATA RESUME:\n")
print(train)
print(test)
print()

# Load data into tensor
trainset = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=2, shuffle=True)


for data in trainset:
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(data[1])
    break


