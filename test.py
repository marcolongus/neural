import torch



# Check torch working well
try:
	x = torch.rand(1, 1)
	print("-->",x)
except:
	print("-->Torch.rand() not working")
finally:
	print(f"--> CUDA is available: {torch.cuda.is_available()}")

