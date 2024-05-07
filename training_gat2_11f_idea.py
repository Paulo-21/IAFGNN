import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import statistics
import dgl
from dgl.nn import GATv2Conv
import time
import DatasetDGL
import schedulefree

af_data_root = "../af_dataset/"
result_root = "../af_dataset/all_result/"
task = "DS-ST"
print(task)
MAX_ARG = 200000
v = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
print(v)

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
device1 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = device1
print("runtime : ", device)
print("runtime : ", device1)
torch.backends.cudnn.benchmark = True

model = GAT(8).to(device1)
model_path = "model_save/"+task+"-14-gatv2_idea.pth"
#if os.path.exists(model_path):
#    model.load_state_dict(torch.load(model_path))
#total_params = sum(p.numel() for p in model.parameters())
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("total parameters : ", total_params)

loss = nn.BCELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#optimizer = torch.optim.AdamW(model.parameters())
optimizer = schedulefree.AdamWScheduleFree(model.parameters())

model.train()
optimizer.train()
print("Loading Data...")
tic = time.perf_counter()
#af_dataset = DatasetDGL.TrainingGraphDataset(af_data_root+"dataset_af/", af_data_root+"result/", task=task, device=device)
af_dataset = DatasetDGL.LarsMalmDataset(task=task, device=device)
data_loader = dgl.dataloading.GraphDataLoader(af_dataset, batch_size=7, shuffle=True)
print(time.perf_counter()-tic)
print("Start training")

model.train()
for epoch in range(500):
    tot_loss = []
    tot_loss_v = 0
    i=0
    for graph in data_loader:
        inputs = graph.ndata["feat"]#.to(device1)
        label = graph.ndata["label"]#.to(device1)
        graph_cdn = graph#.to(device1)
        optimizer.zero_grad()
        pred = model(graph_cdn, inputs)
        losse = loss(pred, label)
        losse.backward()

        tot_loss.append(losse.item())
        tot_loss_v += losse.item()
        i+=1
        optimizer.step()
    #if epoch == 8:
    #    for g in optimizer.param_groups:
    #        g['lr'] = 0.001
    print(i, "Epoch : ", epoch," Mean : " , statistics.fmean(tot_loss), " Median : ", statistics.median(tot_loss), "loss : ", tot_loss_v)
torch.save(model.state_dict(), model_path)

print("final test start")
optimizer.eval()
DatasetDGL.test(model, task=task, device=device1)
#torch.save(model.state_dict(), model_path)
#print("F1 Score : ", f1_score/len(af_dataset))
