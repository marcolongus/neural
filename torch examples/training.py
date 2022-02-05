import torch.optim as optim
from data import *
from convolution import *

net = Convolution()

optimizer = optim.Adam(net.parameters(), lr=0.001)