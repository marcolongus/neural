import torch
import torch.nn as nn
import torch.nn.functional as F


m = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)

x = torch.rand(1, 50, 50)

print(x)

output = m(x)


print(output.shape)

print(output)