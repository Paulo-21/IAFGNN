import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import DGLDataset
import af_reader_py
import statistics
import platform
import dgl
from sklearn.preprocessing import StandardScaler
from dgl.nn import GraphConv

af_data_root = "../af_dataset/"
result_root = "../af_data/"

def transfom_to_graph(label_path, n):
    f = open(label_path, 'r')
    data = f.read()
    target = [0.]*n
    for n in data.split(','):
        if n == '':
            continue
        target[int(n)] = 1.0
    return torch.tensor(target, requires_grad=False).to(device)

def get_item(af_name, af_dir, label_dir):
    af_path = os.path.join(af_dir,af_name)
    label_path = os.path.join(label_dir,af_name)
    att1, att2, nb_el = af_reader_py.reading_cnf_for_dgl(af_path)
    target = transfom_to_graph(label_path, nb_el)
    graph = dgl.graph((torch.tensor(att1),torch.tensor(att2)), device=device).to(device)
    raw_features = af_reader_py.compute_features(af_path, 10000, 0.00001 )
    scaler = StandardScaler()
    features = scaler.fit_transform(raw_features)
    features_tensor = torch.tensor(features, dtype=torch.float32, requires_grad=False).to(device)
    #    torch.save(features_tensor, af_data_root+"features_tensor/" + "" + af_name+".pt")
    if graph.number_of_nodes() < nb_el:
        graph.add_nodes(nb_el - graph.number_of_nodes())
    
    graph = dgl.add_self_loop(graph)
    num_rows_to_overwrite = features_tensor.size(0)
    num_columns_in_features = features_tensor.size(1)
    inputs = torch.randn(graph.number_of_nodes(), 128 , dtype=torch.float32, requires_grad=False).to(device)
    inputs_to_overwrite = inputs.narrow(0, 0, num_rows_to_overwrite).narrow(1, 0, num_columns_in_features)
    inputs_to_overwrite.copy_(features_tensor)
    
    return graph, inputs, target, nb_el

class CustumGraphDataset(DGLDataset):
    def __init__(self, af_dir, label_dir):
        super().__init__(name="Af dataset")
        self.label_dir = label_dir
        self.af_dir = af_dir
    def __len__(self):
        return len(self.graphs)
    def process(self):
        self.af_dir = af_data_root+"dataset_af_2023/"
        self.label_dir = result_root+"result_DC-CO/"
        self.graphs = []
        self.labels = []
        for file in os.listdir(self.label_dir):
            graph, features, label, nb_el = get_item(file, self.af_dir, self.label_dir)
            if nb_el < 25000:
                graph.ndata["feat"] = features
                graph.ndata["label"] = label
                self.graphs.append(graph)

    def __getitem__(self, idx:int):
        return self.graphs[idx]

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
        #h = self.layer5(g, h + inputs)
        #h = F.relu(h)
        #h = self.dropout(h)
        h = self.fc(h)
        return h.squeeze()  # Remove the last dimension

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("runtime : ", device)
model = GCN(128, 128, 128, 1).to(device)
if platform.system() == "Linux" and torch.cuda.is_available():
    model = torch.compile(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss = nn.BCELoss()
model.train()
print("Loading Data...")
af_dataset = CustumGraphDataset(af_data_root+"dataset_af/", af_data_root+"result/")
data_loader = dgl.dataloading.GraphDataLoader(af_dataset, batch_size=64, shuffle=True)
test_data_loader = dgl.dataloading.GraphDataLoader(af_dataset, batch_size=64, shuffle=False)
#train_dataloader = DataLoader(af_dataset, batch_size=64)
print("Start training")
for epoch in range(200):
    tot_loss = []
    tot_loss_v = 0
    model.train()
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
    print("Epoch : ", epoch," Mean : " , statistics.fmean(tot_loss), " Median : ", statistics.median(tot_loss), "loss : ", tot_loss_v)
    #print("acc : ", (acc_yes+acc_no)/(tot_el_no+tot_el_yes) ,"acc yes : ", acc_yes/tot_el_yes, "acc no : ", acc_no/tot_el_no )
"""
    model.eval()
    tot_el_yes = 0
    tot_el_no = 0
    acc_yes = 0
    acc_no = 0

    with torch.no_grad():
        for graph in test_data_loader:
            inputs = graph.ndata["feat"]
            label = graph.ndata["label"]
            out = model(graph, inputs)
            predicted = (torch.sigmoid(out.squeeze())>0.9).float()
            acc_yes += sum(element1 == element2 == 1.0  for element1, element2 in zip(predicted, label)).item()
            acc_no += sum(element1 == element2 == 0.0   for element1, element2 in zip(predicted, label)).item()
            tot_el_yes += sum(element1 == 1.0  for element1 in label).item()
            tot_el_no += sum(element1 == 0.0   for element1 in label).item()
"""
print("final test")

model.eval()
acc_yes = 0
acc_no = 0
tot_el_yes = 0
tot_el_no = 0
with torch.no_grad():
    for graph in af_dataset:
        inputs = graph.ndata["feat"]
        label = graph.ndata["label"]
        out = model(graph, inputs)
        predicted = (torch.sigmoid(out.squeeze())>0.9).float()
        acc_yes += sum(element1 == element2 == 1.0  for element1, element2 in zip(predicted, label)).item()
        acc_no += sum(element1 == element2 == 0.0   for element1, element2 in zip(predicted, label)).item()
        tot_el_yes += sum(element1 == 1.0  for element1 in label).item()
        tot_el_no += sum(element1 == 0.0   for element1 in label).item()

print("acc : ", (acc_yes+acc_no)/(tot_el_no+tot_el_yes) ,"acc yes : ", acc_yes/tot_el_yes, "acc no : ", acc_no/tot_el_no )