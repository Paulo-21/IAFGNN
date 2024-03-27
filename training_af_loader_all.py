#from itertools import chain
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from dgl.data import DGLDataset
import af_reader_py
import statistics
import platform
import dgl
from sklearn.preprocessing import StandardScaler
from dgl.nn import GraphConv

af_data_root = "../af_dataset/"
result_root = "../af_dataset/all_result/"
task = "DS-ST"
MAX_ARG = 90000
v = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
print(v)
#= "expandable_segments:True"

def transfom_to_graph(label_path, n):
    f = open(label_path, 'r')
    data = f.read()
    target = [0.]*n
    for n in data.split(','):
        if n == '':
            continue
        target[int(n)] = 1.0
    return torch.tensor(target, requires_grad=False, device=device)
def light_get_item(af_name, af_dir, label_dir, year):
    af_path = os.path.join(af_dir,af_name)
    label_path = os.path.join(label_dir,af_name)
    att1, att2, nb_el = af_reader_py.reading_file_for_dgl(af_path)
    if nb_el > MAX_ARG:
        return None, None, None, 1000000
    target = transfom_to_graph(label_path, nb_el)
    graph = dgl.graph((torch.tensor(att1),torch.tensor(att2)), num_nodes = nb_el, device=device)
    print("L : ",nb_el, " ", graph.number_of_nodes())
    if graph.number_of_nodes() < nb_el:
        print("DIFFFFFFFFFFFFFFFFFFFFFFFFFFF : ", af_name)
        #graph.add_nodes(nb_el - graph.number_of_nodes())
    graph = dgl.add_self_loop(graph)
    features_tensor = torch.load(af_data_root+"all_features/"+year+"/"+af_name+".pt", map_location=device)
    num_rows_to_overwrite = features_tensor.size(0)
    num_columns_in_features = features_tensor.size(1)
    inputs = torch.randn(graph.number_of_nodes(), 128 , dtype=torch.float32, requires_grad=False).to(device)
    inputs_to_overwrite = inputs.narrow(0, 0, num_rows_to_overwrite).narrow(1, 0, num_columns_in_features)
    inputs_to_overwrite.copy_(features_tensor)
    return graph, inputs, target, nb_el
 
def get_item(af_name, af_dir, label_dir, year):
    print(af_name," ",  af_dir, " ", label_dir)
    af_path = os.path.join(af_dir,af_name)
    label_path = os.path.join(label_dir,af_name)
    att1, att2, nb_el = af_reader_py.reading_file_for_dgl(af_path)
    if nb_el > MAX_ARG:
        return None, None, None, 1000000
    target = transfom_to_graph(label_path, nb_el)
    graph = dgl.graph((torch.tensor(att1),torch.tensor(att2)), num_nodes = nb_el, device=device)#.to(device)
    raw_features = af_reader_py.compute_features(af_path, 10000, 0.00001 )
    scaler = StandardScaler()
    features = scaler.fit_transform(raw_features)
    features_tensor = torch.tensor(features, dtype=torch.float32, requires_grad=False).to(device)
    torch.save(features_tensor, af_data_root+"all_features/"+year+"/"+af_name+".pt")
    print("N : ",nb_el," ", graph.number_of_nodes())
    if graph.number_of_nodes() < nb_el:
        print("---------------------------------------------------------------------------------")
        print("DIFFFFFFFFFFFFFFFFFFFFFFFFFFF : ", af_name)
        #graph.add_nodes(nb_el - graph.number_of_nodes())
    
    graph = dgl.add_self_loop(graph)
    num_rows_to_overwrite = features_tensor.size(0)
    num_columns_in_features = features_tensor.size(1)
    inputs = torch.randn(graph.number_of_nodes(), 128 , dtype=torch.float32, requires_grad=False).to(device)
    inputs_to_overwrite = inputs.narrow(0, 0, num_rows_to_overwrite).narrow(1, 0, num_columns_in_features)
    inputs_to_overwrite.copy_(features_tensor)
    
    return graph, inputs, target, nb_el

class TrainingGraphDataset(DGLDataset):
    def __init__(self, af_dir, label_dir):
        super().__init__(name="Af dataset")
        self.label_dir = label_dir
        self.af_dir = af_dir
    def __len__(self):
        return len(self.graphs)
    def process(self):
        #list_year_dir = ["2017", "2023"]
        list_year_dir = ["2017"]
        self.af_dir = af_data_root+"dataset_af"
        self.label_dir = result_root+"result_"+task
        self.graphs = []
        self.labels = []
        list_unique_file = []
        for year in list_year_dir:
            iter = os.listdir(self.label_dir +"_"+ year)
            for f in iter:
                true_name = f.replace(".apx", "")
                true_name = true_name.replace(".af", "")
                true_name = true_name.replace(".tgf", "")
                true_name = true_name.replace(".old", "")
                #if f != "afinput_exp_acyclic_indvary3_step7_batch_yyy09.apx":
                #    continue
                #if True:
                if true_name not in list_unique_file:
                    list_unique_file.append(true_name)
                    #print(f," ",  self.label_dir+"_"+year )
                    
                    if os.path.exists(af_data_root+"all_features/"+year+"/"+f+".pt"):
                        graph, features, label, nb_el = light_get_item(f, self.af_dir+"_"+year+"/", self.label_dir+"_"+year+"/", year)
                    else:
                        graph, features, label, nb_el = get_item(f, self.af_dir+"_"+year+"/", self.label_dir+"_"+year+"/", year)
                    if nb_el < MAX_ARG:
                        graph.ndata["feat"] = features
                        graph.ndata["label"] = label
                        #self.labels.append(label)
                        self.graphs.append(graph)

    def __getitem__(self, idx:int):
        return self.graphs[idx]
class ValisationDataset(DGLDataset):
    def __init__(self, af_dir, label_dir):
        super().__init__(name="Validation Dataset")
        self.label_dir = label_dir
        self.af_dir = af_dir
    def __len__(self):
        return len(self.graphs)
    def process(self):
        #list_year_dir = ["2017", "2023"]
        list_year_dir = ["2023"]
        self.af_dir = af_data_root+"dataset_af"
        self.label_dir = result_root+"result_"+task
        self.graphs = []
        self.labels = []
        list_unique_file = []
        for year in list_year_dir:
            iter = os.listdir(self.label_dir +"_"+ year)
            for f in iter:
                true_name = f.replace(".apx", "")
                true_name = true_name.replace(".af", "")
                true_name = true_name.replace(".tgf", "")
                true_name = true_name.replace(".old", "")
                if true_name not in list_unique_file:
                    list_unique_file.append(true_name)
                    #print(f," ",  self.label_dir+"_"+year )
                    if os.path.exists(af_data_root+"all_features/"+year+"/"):
                        graph, features, label, nb_el = light_get_item(f, self.af_dir+"_"+year+"/", self.label_dir+"_"+year+"/", year)
                    else:
                        graph, features, label, nb_el = get_item(f, self.af_dir+"_"+year+"/", self.label_dir+"_"+year+"/", year)
                    if nb_el > MAX_ARG:
                        continue
                    graph.ndata["feat"] = features
                    #self.labels.append(label)
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
        #h = self.layer5(g, h )
        #h = F.relu(h)
        #h = self.dropout(h)
        h = self.fc(h)
        return h.squeeze()  # Remove the last dimension

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = "cpu"
print("runtime : ", device)
model = GCN(128, 128, 128, 1).to(device)
model_path = "v3-"+task+".pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
#if platform.system() == "Linux" and torch.cuda.is_available():
#    model = torch.compile(model)
loss = nn.BCELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)
model.train()
print("Loading Data...")
af_dataset = TrainingGraphDataset(af_data_root+"dataset_af/", af_data_root+"result/")
data_loader = dgl.dataloading.GraphDataLoader(af_dataset, batch_size=64, shuffle=True)
#print("FINAL TEST LOADING...")
#test_data_loader = dgl.dataloading.GraphDataLoader(af_dataset, batch_size=64, shuffle=False)
#train_dataloader = DataLoader(af_dataset, batch_size=64)
print("Start training")
model.train()
for epoch in range(20000):
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
    #if epoch > 120:
    #    scheduler.step()
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
print("final test start")
af_dataset = ValisationDataset(af_data_root+"dataset_af/", af_data_root+"result/")
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
        predicted = (torch.sigmoid(out.squeeze())>0.5).float()
        acc_yes += sum(element1 == element2 == 1.0  for element1, element2 in zip(predicted, label)).item()
        acc_no += sum(element1 == element2 == 0.0   for element1, element2 in zip(predicted, label)).item()
        tot_el_yes += sum(element1 == 1.0  for element1 in label).item()
        tot_el_no += sum(element1 == 0.0   for element1 in label).item()

print("acc : ", (acc_yes+acc_no)/(tot_el_no+tot_el_yes) ,"acc yes : ", acc_yes/tot_el_yes, "acc no : ", acc_no/tot_el_no )

torch.save(model.state_dict(), "v3-"+task+".pth")
