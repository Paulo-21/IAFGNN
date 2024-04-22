import sys
import time
import af_reader_py
"""
1.94 sec before import optimisation
25 ms after
"""
file = sys.argv[1]
task = sys.argv[2]
argId = sys.argv[3]
device = "cpu"
raw_features, nb_el, arg_pos, acceptance = af_reader_py.special_gs(file, argId)
if acceptance != 2:
    if acceptance == 1:
        print("YES")
        exit(0)
    elif acceptance == 0:
        print("NO")
        exit(0)
    else:
        print("ERROR")
        exit(1)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_FEATURES = 7

class MultiLinear(nn.Module):
    def __init__(self):
        super(MultiLinear, self).__init__()
        self.layer1 = nn.Linear(INPUT_FEATURES, 49)
        self.layer2 = nn.Linear(49, 49)
        self.layer3 = nn.Linear(49, INPUT_FEATURES)
        self.layer4 = nn.Linear(INPUT_FEATURES, 1)
    def forward(self, inputs):
        h = self.layer1(inputs)
        h = F.leaky_relu(h)
        h = self.layer2(h)
        h = F.leaky_relu(h)
        h = self.layer3(h)
        h = F.leaky_relu(h)
        h = self.layer4(h)
        h = F.sigmoid(h)
        return h  # Remove the last dimension

model = MultiLinear()
model_dir = "model_save/"
model_path = model_dir + "linear_"+task+".pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

#tic = time.perf_counter()
inputs = torch.Tensor(raw_features[arg_pos])
#print(time.perf_counter()-tic)
with torch.no_grad():
    out = model(inputs)
    predicted = (out.squeeze()>0.5).float()
    if predicted == True:
        print("YES")
    else:
        print("NO")
