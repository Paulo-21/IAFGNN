import sys
import af_reader_py

file = sys.argv[1]
task = sys.argv[2]
argId = sys.argv[3]
device = "cpu"
raw_features ,att1, att2, nb_el, arg_pos, acceptance = af_reader_py.special_gs_for_gat(file, argId)

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
import dgl
from dgl.nn import GATv2Conv
from sklearn.preprocessing import StandardScaler

class GAT(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.num_head = 2
        self.gat_att1 = GATv2Conv( in_size, in_size, self.num_head, residual=True, activation=F.elu)
        self.gat_def1 = GATv2Conv( in_size, in_size, self.num_head, residual=True, activation=F.elu)
        self.gat_def2 = GATv2Conv( in_size*self.num_head, in_size, self.num_head, residual=True, activation=F.elu)
        self.input_features = in_size*self.num_head  + in_size*self.num_head + in_size
        self.hidden_features = self.input_features**2
        self.layer1 = nn.Linear(self.input_features, self.hidden_features)
        self.layer2 = nn.Linear(self.hidden_features, self.hidden_features)
        self.layer3 = nn.Linear(self.hidden_features, self.input_features)
        self.layer4 = nn.Linear(self.input_features, 1)

    def forward(self, g, inputs):
        h_att1 = self.gat_att1(g, inputs).flatten(1)
        h_def1 = self.gat_def1(g, inputs)
        h_def1 = h_def1.flatten(1)
        h_def2 = self.gat_def2(g, h_def1).flatten(1)
        inputs = torch.cat((h_att1, h_def2, inputs), 1)
        h = self.layer1(inputs)
        h = F.leaky_relu(h)
        h = self.layer2(h)
        h = F.leaky_relu(h)
        h = self.layer3(h)
        h = F.leaky_relu(h)
        h = self.layer4(h)
        h = F.sigmoid(h)
        return h
    
graph = dgl.graph((att1,att2), num_nodes=nb_el, device=device)
graph = dgl.add_self_loop(graph)

model_path = "model_save/"+task+"-14-gatv2_idea.pth"
save = torch.load(model_path, map_location=device)
scaler = StandardScaler()
scaler.mean_ = save["mean"]
scaler.var_ = save["var"]
scaler.scale_ = save["scale"]
features = scaler.transform(raw_features)
inputs = torch.tensor(features, dtype=torch.float32)

model = GAT(8).to(device)
model.eval()
model.load_state_dict(save["model"])

with torch.no_grad():
    pred = model(graph, inputs)
    predicted = (pred.squeeze()>0.5).float()
    if predicted[arg_pos] == True:
        print("YES")
    else:
        print("NO")
