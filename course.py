import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#print(torch.__version__)

a = torch.tensor([[1,3], [3,4], [3,4]])

#print(a.shape)
#print(a.ndim)

r = torch.rand(3,4, 3)
print(r)