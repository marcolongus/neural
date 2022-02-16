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

# Load data into tensor, and shuffle
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

if __name__ == "__main__":
    # Data resume
    print("DATA RESUME:\n")
    print(train)
    print(test)
    print()

    for data in trainset:
        images, target = data
        for i, image in enumerate(images):
            plt.title(target[i])
            plt.imshow(image.view(28,28))
            plt.show()
        break


