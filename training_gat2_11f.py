import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim.lr_scheduler as lr_scheduler
from dgl.data import DGLDataset
import af_reader_py
import statistics
import dgl
from sklearn.preprocessing import StandardScaler
from dgl.nn import GATv2Conv
import time
import DatasetDGL

af_data_root = "../af_dataset/"
result_root = "../af_dataset/all_result/"
task = "DS-PR"
print(task)
MAX_ARG = 200000
v = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
print(v)

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # three-layer GAT
        self.gat_layers.append(
            GATv2Conv(in_size, hid_size, heads[0], activation=F.elu)
        )
        self.gat_layers.append(
            GATv2Conv(
                hid_size * heads[0],
                hid_size,
                heads[1],
                residual=True,
                activation=F.elu,
            )
        )
        self.gat_layers.append(
            GATv2Conv(
                hid_size * heads[1],
                out_size,
                heads[2],
                residual=True,
                activation=None,
            )
        )
    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 2:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h
device1 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
#device = "cpu"
print("runtime : ", device)
print("runtime : ", device1)
torch.backends.cudnn.benchmark = True

model = GAT(11, 5, 1, heads=[5, 3, 3]).to(device1)
model_path = "v3-"+task+"-11-gatv2.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
#total_params = sum(p.numel() for p in model.parameters())
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("total parameters : ", total_params)

loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
print("Loading Data...")
tic = time.perf_counter()
af_dataset = DatasetDGL.TrainingGraphDataset(af_data_root+"dataset_af/", af_data_root+"result/", task=task, device=device)
#af_dataset = DatasetDGL.LarsMalmDataset(task=task, device=device)
data_loader = dgl.dataloading.GraphDataLoader(af_dataset, batch_size=4, shuffle=True)
print(time.perf_counter()-tic)
print("Start training")
#scaler = GradScaler()
model.train()
for epoch in range(400):
    tot_loss = []
    tot_loss_v = 0
    i=0
    tic = time.perf_counter()
    for graph in data_loader:
        torch.cuda.empty_cache()
        inputs = graph.ndata["feat"].to(device1)
        label = graph.ndata["label"].to(device1)
        graph_cdn = graph.to(device1)
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        out = model(graph_cdn, inputs)
        predicted = (torch.sigmoid(out.squeeze())).float()
        del inputs
        del graph_cdn
        torch.cuda.empty_cache()
        losse = loss(predicted, label)
        del label
        torch.cuda.empty_cache()
        losse.backward()
        torch.cuda.empty_cache()
        tot_loss.append(losse.item())
        tot_loss_v += losse.item()
        i+=1
        #if i%4 == 0:
        optimizer.step()
    print("compute : ", time.perf_counter()-tic)
    if epoch == 150:
        for g in optimizer.param_groups:
            g['lr'] = 0.001
    if epoch == 380:
        for g in optimizer.param_groups:
            g['lr'] = 0.0001
    print(i, "Epoch : ", epoch," Mean : " , statistics.fmean(tot_loss), " Median : ", statistics.median(tot_loss), "loss : ", tot_loss_v)

torch.save(model.state_dict(), model_path)

print("final test start")
DatasetDGL.test(model, task=task, device=device)
#torch.save(model.state_dict(), model_path)
#print("F1 Score : ", f1_score/len(af_dataset))