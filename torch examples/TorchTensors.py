import torch

array = [1, 2, 3]

# Basic declaration
x = torch.Tensor(array)
y = torch.zeros([3,5])
z = torch.rand([5,3])

#reshape and assing
x = x.view([1,3])

print(x)
print(y)
print(z)
print()

print(x.shape)
print(y.shape)
print(z.shape)

###########################
# Data
###########################
import torchvision
from torchvision import transforms, datasets