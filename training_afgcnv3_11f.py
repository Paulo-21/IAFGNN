import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import statistics
import dgl
from sklearn.preprocessing import StandardScaler
from dgl.nn import GraphConv
import DatasetDGL

af_data_root = "../af_dataset/"
result_root = "../af_dataset/all_result/"
task = "DS-PR"
print(task)
MAX_ARG = 200000
v = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
print("PYTORCH_CUDA_ALLOC_CONF : ", v)

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, fc_features, num_classes, dropout=0.5):
        super(GCN, self).__init__()
        self.layer1 = GraphConv(in_features, hidden_features)
        self.layer2 = GraphConv(hidden_features, hidden_features)
        self.layer3 = GraphConv(hidden_features, hidden_features)
        self.layer4 = GraphConv(hidden_features, hidden_features)
        #self.layer5 = GraphConv(hidden_features, fc_features)
        self.fc = nn.Linear(fc_features, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        h = self.layer1(g, inputs)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layer2(g, h + inputs)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layer3(g, h + inputs)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layer4(g, h + inputs)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.fc(h)
        return h.squeeze()  # Remove the last dimension

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = "cpu"
print("runtime : ", device)
torch.backends.cudnn.benchmark = True
model = GCN(11, 11, 11, 1).to(device)
model_path = "model_w_lars_data/"+"v3-"+task+"-11.pth"
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("total parameters : ", total_params)
#if os.path.exists(model_path):
#    model.load_state_dict(torch.load(model_path))
loss = nn.BCELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
print("Loading Data...")
#af_dataset = DatasetDGL.TrainingGraphDataset(af_data_root+"dataset_af/", af_data_root+"result/", task=task, device=device)
af_dataset = DatasetDGL.LarsMalmDataset(task=task, device=device)
data_loader = dgl.dataloading.GraphDataLoader(af_dataset, batch_size=64, shuffle=True)

print("Start training")
model.train()
for epoch in range(400):
    tot_loss = []
    tot_loss_v = 0
    for graph in data_loader:
        inputs = graph.ndata["feat"]
        label = graph.ndata["label"]
        optimizer.zero_grad()
        out = model(graph, inputs)
        predicted = (torch.sigmoid(out.squeeze())).float()
        losse = loss(predicted, label)
        losse.backward()
        optimizer.step()
        tot_loss.append(losse.item())
        tot_loss_v += losse.item()
    if epoch == 150:
        for g in optimizer.param_groups:
            g['lr'] = 0.001
    
    print("Epoch : ", epoch," Mean : " , statistics.fmean(tot_loss), " Median : ", statistics.median(tot_loss), "loss : ", tot_loss_v)

print("final test start")
#TEST the Model 
DatasetDGL.test(model, task=task, device=device)

torch.save(model.state_dict(), model_path)