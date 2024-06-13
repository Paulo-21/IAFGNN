import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#scaler = StandardScaler()
#array  = torch.tensor([[0.3,0.3], [0.5,0.6], [0.3,1.0] ])
scaler = StandardScaler()
array  = torch.tensor([[0.3,0.3], [0.3,0.3]])
out = scaler.fit_transform(array)
print(out)