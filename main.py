import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

def data_loader():
    training_data = datasets.FashionMNIST(
        root="fashion data", train=True, download=True, transform=None
    )
    return training_data


training_data = data_loader()
