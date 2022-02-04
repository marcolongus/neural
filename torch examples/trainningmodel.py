from neuralnetwork import *
from data import * 
import torch.optim as optim

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3
for epoch in range(EPOCHS):
	print(f'EPOCH {epoch}:')
	for data in trainset:
		# Data is a batch of featuresets and labels
		images, target = data
		net.zero_grad()
		output = net(images.view(-1, 28*28))
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
	print(loss)

correct = 0
total = 0

# For test the data
with torch.no_grad():
	 