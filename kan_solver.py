import sys
import af_reader_py

file = sys.argv[1]
task = sys.argv[2]
argId = sys.argv[3]
device = "cuda"

raw_features , nb_el, arg_pos, acceptance = af_reader_py.special_gs(file, argId)
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

import torch
from kan import KAN
model = KAN(width=[8,2,2], grid=7, k=5, device=device)
model.load_state_dict(torch.load("kan_model/kan.pth"))
l = []
l.append(raw_features[arg_pos])
with torch.no_grad():
    pred = model.forward(torch.tensor(l, device=device))
    acce = torch.argmax(pred, dim=1)
    if acce == 0:
        print("NO")
    elif acce == 1:
        print("YES")
    else:
        print("Error")
    