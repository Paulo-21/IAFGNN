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
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATv2Conv
from sklearn.preprocessing import StandardScaler

class GAT(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        in_size = in_size-3
        self.num_head = 3
        self.outgat_att = 1#in_size
        self.outgat_def = 2#in_size
        self.dropout = nn.Dropout(0.2)
        self.gat_att1 = GATv2Conv( in_size, self.outgat_att, self.num_head, residual=True, activation=F.elu, allow_zero_in_degree=True)

        self.gat_def1 = GATv2Conv( in_size, in_size, self.num_head, residual=True, activation=F.elu, allow_zero_in_degree=True)
        self.gat_def2 = GATv2Conv( in_size*self.num_head, self.outgat_def, self.num_head, residual=True, activation=F.elu, allow_zero_in_degree=True)
        #self.input_features = self.outgat*self.num_head  + self.outgat*self.num_head + in_size #+ self.outgat*self.num_head
        self.input_features = self.outgat_att*self.num_head  + self.outgat_def*self.num_head + in_size+3 #+ self.outgat*self.num_head 
        self.hidden_features = self.input_features**2
        self.layer1 = nn.Linear(self.input_features, self.hidden_features)
        self.layer2 = nn.Linear(self.hidden_features, self.hidden_features)
        self.layer3 = nn.Linear(self.hidden_features, self.input_features)
        self.layer4 = nn.Linear(self.input_features, 1)

    def forward(self, g, inputs):
        gat_inputs = inputs[:,:5]
        h_att = self.gat_att1(g, gat_inputs).flatten(1)
        h_def = self.gat_def1(g, gat_inputs).flatten(1)
        del gat_inputs
        h_def = self.gat_def2(g, h_def).flatten(1)
        inputs = torch.cat((h_att, h_def, inputs), 1)
        h = self.layer1(inputs)
        h = F.leaky_relu(h)
        h = self.layer2(h)
        h = F.leaky_relu(h)
        h = self.layer3(h)
        h = F.leaky_relu(h)
        h = self.dropout(h)
        h = self.layer4(h)
        h = F.sigmoid(h)
        return h
graph = dgl.graph((att1,att2), num_nodes=nb_el, device=device)
#graph = dgl.add_self_loop(graph)

model_path = "model_save/"+task+"-14-gatv2_idea.pth"
save = torch.load(model_path, map_location=device)
scaler = StandardScaler()
features = scaler.fit_transform(raw_features)
"""scaler.mean_ = save["mean"]
scaler.var_ = save["var"]
scaler.scale_ = save["scale"]
features = scaler.transform(raw_features)"""
inputs = torch.tensor(raw_features, dtype=torch.float32)

model = GAT(8).to(device)
model.eval()
model.load_state_dict(save)

with torch.no_grad():
    pred = model(graph, inputs)
    predicted = (pred.squeeze()>0.5).float()
    if predicted[arg_pos] == True:
        print("YES")
    else:
        print("NO")
