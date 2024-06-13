import torch
from efficient_kan import KAN
from torch.utils.data import DataLoader
import DatasetLinear
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics
import DatasetLinear
import schedulefree

af_data_root = "../af_dataset/"
result_root = "../af_dataset/all_result/"
task = "DC-CO"
print(task)
MAX_ARG = 200000
v = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
print("PYTORCH_CUDA_ALLOC_CONF : ", v)
INPUT_FEATURES = 9
HIDDEN_FEATURES = (INPUT_FEATURES*INPUT_FEATURES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = "cpu"
print("runtime : ", device)
model = KAN([INPUT_FEATURES, HIDDEN_FEATURES, HIDDEN_FEATURES, INPUT_FEATURES, 1] ).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("total parameters : ", total_params)
#loss = nn.BCELoss().cuda()
loss = nn.CrossEntropyLoss().cuda()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = schedulefree.AdamWScheduleFree(model.parameters())
optimizer = torch.optim.LBFGS(model.parameters(), lr=1., history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)
print("Loading Data...")
#af_dataset = DatasetDGL.TrainingGraphDataset(af_data_root+"dataset_af/", af_data_root+"result/", task=task, device=device)
tic = time.perf_counter()
dataset = DatasetLinear.DatasetEffKan(task=task, device=device)
dataloader = DataLoader(dataset, shuffle=True)
print(time.perf_counter()-tic)
print("Start training")
model.train()
#optimizer.train()
for epoch in range(500):
    tot_loss = []
    tot_loss_v = 0
    for (inputs, label) in dataloader:
        optimizer.zero_grad()
        out = model(inputs)
        losse = loss(out, label)
        losse.backward()
        optimizer.step()
        tot_loss.append(losse.item())
        tot_loss_v += losse.item()
    print("Epoch : ", epoch," Mean : " , statistics.fmean(tot_loss), " Median : ", statistics.median(tot_loss), "loss : ", tot_loss_v)

print("final test start")
#TEST the Model
model.eval()
optimizer.eval()
#SCRIPT

DatasetLinear.test(model=model, task=task, device=device)


