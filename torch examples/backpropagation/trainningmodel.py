from neuralnetwork import *
from data import * 
import torch.optim as optim

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 5
for epoch in range(EPOCHS):
	print(f'EPOCH {epoch}:')
	for data in trainset:
		images, target = data
		# Forward pass
		net.zero_grad()
		output = net(images.view(-1, 28*28))
		# Calculate loss and backward pass
		loss = F.nll_loss(output, target)
		loss.backward()

		optimizer.step()
	
	print(loss)

correct = 0
counter = 0
total = 0.1

# For test the data
with torch.no_grad():
	for data in trainset:
		images, target = data
		output = net(images.view(-1, 28*28))
		for idx, tensor_output in enumerate(output):
			if torch.argmax(tensor_output) == target[idx]:
				correct += 1
			else:
				counter+=1
				if counter < 10: 
					plt.title(f'Predict:{torch.argmax(tensor_output)}, Target:{target[idx]} ')
					plt.imshow(images[idx].view(28,28))
					plt.show()
			total +=1

print('Accuracy: ', round(correct/total,2)) 

with torch.no_grad():
	for data in trainset:
		images, target = data
		output = net(images.view(-1, 28*28))
		for idx, tensor_output in enumerate(output):
			plt.title(f'Predict:{torch.argmax(tensor_output)}, Target:{target[idx]} ')
			plt.imshow(images[idx].view(28,28))
			plt.show()
		break

