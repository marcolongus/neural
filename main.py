import torch
import numpy as np


def fashion_loader():
    return 0


# CHECK
print(f"CUDA is available: {torch.cuda.is_available()}")
x = torch.rand(5, 3)
print(x)
