import torch
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
array  = torch.tensor([[[0.3,0.3], [0.5,0.6], [0.3,1.0] ], [[0.3,0.3], [0.5,0.6], [0.3,1.0] ]])

for data in array:
    scaler.partial_fit(data)
array2 = [scaler.transform(data) for data in array]

print(array2)