import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import statistics
import dgl
from dgl.nn import GATv2Conv
import time
import DatasetDGL
#import DatasetDGL2
from sklearn.preprocessing import StandardScaler
import schedulefree

af_data_root = "../af_dataset/"
result_root = "../af_dataset/all_result/"
task = "DC-CO"
print(task)
MAX_ARG = 200000
v = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
print(v)

class GAT(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.num_head = 2
        self.outgat = in_size
        self.gat_att1 = GATv2Conv( in_size, self.outgat, self.num_head, residual=True, activation=F.elu)

        self.gat_def1 = GATv2Conv( in_size, in_size, self.num_head, residual=True, activation=F.elu)
        self.gat_def2 = GATv2Conv( in_size*self.num_head, self.outgat, self.num_head, residual=True, activation=F.elu)
        """self.gat_att_def1 = GATv2Conv( in_size, in_size, self.num_head, residual=True, activation=F.elu)
        self.gat_att_def2 = GATv2Conv( in_size*self.num_head, self.outgat, self.num_head, residual=True, activation=F.elu)
        self.gat_att_def3 = GATv2Conv( in_size*self.num_head, self.outgat, self.num_head, residual=True, activation=F.elu)
        """
        self.input_features = self.outgat*self.num_head  + self.outgat*self.num_head + in_size #+ self.outgat*self.num_head 
        self.hidden_features = self.input_features**2
        self.layer1 = nn.Linear(self.input_features, self.hidden_features)
        self.layer2 = nn.Linear(self.hidden_features, self.hidden_features)
        self.layer3 = nn.Linear(self.hidden_features, self.input_features)
        self.layer4 = nn.Linear(self.input_features, 1)

    def forward(self, g, inputs):
        h_att = self.gat_att1(g, inputs).flatten(1)

        h_def = self.gat_def1(g, inputs).flatten(1)
        h_def = self.gat_def2(g, h_def).flatten(1)

        h_att_def = self.gat_att_def1(g, inputs).flatten(1)
        h_att_def = self.gat_att_def2(g, h_att_def).flatten(1)
        h_att_def = self.gat_att_def3(g, h_att_def).flatten(1)

        inputs = torch.cat((h_att, h_def, h_att_def, inputs), 1)
        h = self.layer1(inputs)
        h = F.leaky_relu(h)
        h = self.layer2(h)
        h = F.leaky_relu(h)
        h = self.layer3(h)
        h = F.leaky_relu(h)
        h = self.layer4(h)
        h = F.sigmoid(h)
        return h
device1 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = device1
print("runtime : ", device)
print("runtime : ", device1)
torch.backends.cudnn.benchmark = True
scaler = StandardScaler()
print("Loading Data...")

model = GAT(8).to(device1)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_path = "model_save/"+task+"-14-gatv2_idea.pth"
"""
#if os.path.exists(model_path):
    #loading_model = torch.load(model_path)
    #model.load_state_dict(loading_model["model"])
    scaler.mean_ = loading_model["mean"]
    scaler.var_ = loading_model["var"]
    scaler.scale_ = loading_model["scale"]
    """

print("total parameters : ", total_params)

loss_fn = nn.BCELoss().to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#optimizer = torch.optim.AdamW(model.parameters())
#optimizer = torch.optim.RAdam(model.parameters())
#optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)
optimizer = schedulefree.AdamWScheduleFree(model.parameters())
tic = time.perf_counter()
af_dataset = DatasetDGL.LarsMalmDataset(task=task, device=device)
#af_dataset = DatasetDGL2.LarsMalmDataset(task=task, scaler=scaler, device=device)
data_loader = dgl.dataloading.GraphDataLoader(af_dataset, batch_size=3, shuffle=True)
print(time.perf_counter()-tic)

model.train()
optimizer.train()

print("Start training")

model.train()
epoch=-1
#for epoch in range(500):
while True:
    epoch+=1
    tot_loss = []
    tot_loss_v = 0
    i=0
    for graph in data_loader:
        inputs = graph.ndata["feat"]
        label = graph.ndata["label"]
        optimizer.zero_grad()
        pred = model(graph, inputs)
        loss = loss_fn(pred, label) 
        loss.backward()
        optimizer.step()
        tot_loss.append(loss.item())
        tot_loss_v += loss.item()
        i+=1
    print(i, "Epoch : ", epoch," Mean : " , statistics.fmean(tot_loss), " Median : ", statistics.median(tot_loss), "loss : ", tot_loss_v)
    #print("epoch : ", epoch)
data_save = {"mean" : scaler.mean_, "var": scaler.var_, "scale": scaler.scale_, "model": model.state_dict()}
torch.save(data_save, model_path)

print("final test start")
optimizer.eval()
DatasetDGL.test(model, task=task, scaler=scaler, device=device1)
#DatasetDGL2.test(model, task=task, scaler=scaler, device=device1)

"""def closure():
          if torch.is_grad_enabled():
            optimizer.zero_grad()
          output = model(graph, inputs)
          loss = loss_fn(output, label)
          if loss.requires_grad:
            loss.backward()
            tot_loss.append(loss.item())
            global tot_loss_v
            tot_loss_v += loss.item()
          return loss"""