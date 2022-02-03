# Import standard libraries
import matplotlib.pyplot as plt
import numpy as np

# Import PyTorch modules
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

# Import torchvision module to handle image manipulation
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torchvision.transforms import ToTensor
from torchvision.transforms import Lambda
from torchvision.transforms import Compose


def data_loader():
    """
    Data is a dict over tuples, object[idx] = (image, target).
    Image is a (28,28) grayscaled image --> needs to be transformed to an array.
    Target range from 0 to 9. Indicates the label of the image.
    """
    training_data = datasets.FashionMNIST(
        root="fashion data", train=True, download=True, transform=None
    )
    return training_data


training_data = data_loader()
