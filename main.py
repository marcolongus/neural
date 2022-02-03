##############################################################################
# Import standard libraries
import matplotlib.pyplot as plt
import numpy as np

# Import PyTorch modules
import torch
from torch import nn  # Neural networks builder
from torch import optim  # optimazers SGD, ADAM, etc
from torch.utils.data import DataLoader

# from torch.utils.tensorboard import SummaryWriter # TensorBoard support

# Import torchvision module to handle image manipulation
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torchvision.transforms import ToTensor
from torchvision.transforms import Lambda
from torchvision.transforms import Compose

##############################################################################


def data_loader():

    """
    Data is a dict over tuples, object[idx] = (image, target).
    Image is a (28,28) grayscaled image --> needs to be transformed to an array.
    Target range from 0 to 9. Indicates the label of the image.
    """
    training_data = datasets.FashionMNIST(
        root="fashion data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    return training_data


def data_viz(image):
    plt.imshow(image, cmap="gray")
    plt.show()


##############################################################################
# Main
##############################################################################

training_data = data_loader()
# data_viz(training_data[0][0])

x = training_data[0][0].reshape(28,28)
print(x.shape)
print(training_data[0][0].shape)


y = torch.rand(2,2)

print(y.shape)