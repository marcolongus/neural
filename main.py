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


training_data = datasets.MNIST(root="data", train=True, download=True, transform=None)

print(len(training_data))
