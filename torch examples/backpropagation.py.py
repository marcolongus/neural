import torch 
from torch.autograd import Variable

dtype = torch.FloatTensor
# dtypy = torch.cuda.FloatTensor 

N = 64 # batch size 
D_in = 1000 #input dimension 
H = 100 # hidden dimension
D_out = 10 #output dimension

# Create random Tensors to hold input and outputs, and wrap them in variables
x = Variable(torch.rand(N, D_in).type(dtype), requires_grad = False)
y = Variable(torch.rand(N, D_out).type(dtype), requires_grad = False)

# Create random Tensors for wieghts, and wrap the in Variables
w1 = Variable(torch.rand(D_in, H).type(dtype), requires_grad = True)
w2 = Variable(torch.rand(H, D_out).type(dtype), requires_grad = True)

print(type(w1))
learning_rate = 1e-6

for t in range(500):
	# Foward pass: compute predicted y using operations on variables
	# x.mm matrix multiplication, clamp clamps all the ell min< values < max
	y_pred = x.mm(w1).clamp(min=0).mm(w2)

	loss = (y_pred - y).pow(2).sum()
	print(t, loss.data[0])

	w1.grad.data.zero_()
	w2.grad.data.zero_()

	loss.backward()

	w1.data -= learning_rate * w1.grad.data
	w2.data -= learning_rate * w1.grad.data
