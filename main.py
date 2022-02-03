import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt


# CHECK
print(f"CUDA is available: {torch.cuda.is_available()}")
x = torch.rand(1, 1)
print(x)


def data_loader():
    training_data = datasets.FashionMNIST(
        root="fashion data", train=True, download=True, transform=None
    )
    return training_data


training_data = data_loader()

print(len(training_data))
print(type(training_data))

feature = training_data[8][0]
label = training_data[8][1]

print(label)
print(feature.size)
