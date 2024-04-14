import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim.lr_scheduler as lr_scheduler
from dgl.data import DGLDataset
import af_reader_py
import statistics
#from torch.cuda.amp import autocast, GradScaler
import dgl
from sklearn.preprocessing import StandardScaler
from dgl.nn import GATConv

af_data_root = "../af_dataset/"
result_root = "../af_dataset/all_result/"
task = "DC-CO"
MAX_ARG = 200000
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
    #print("L : ",nb_el, " ", graph.number_of_nodes())
    graph = dgl.add_self_loop(graph)
    inputs = torch.load(af_data_root+"all_features/"+year+"/"+af_name+".pt", map_location=device)
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
    inputs = torch.tensor(features, dtype=torch.float16, requires_grad=False).to(device)
    torch.save(inputs, af_data_root+"all_features/"+year+"/"+af_name+".pt")
    #print("N : ",nb_el," ", graph.number_of_nodes())
    graph = dgl.add_self_loop(graph)
    
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
class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # three-layer GAT
        self.gat_layers.append(
            GATConv(in_size, hid_size, heads[0], activation=F.elu)
        )
        self.gat_layers.append(
            GATConv(
                hid_size * heads[0],
                hid_size,
                heads[1],
                residual=True,
                activation=F.elu,
            )
        )
        self.gat_layers.append(
            GATConv(
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = "cpu"
print("runtime : ", device)
#model = GCN(11, 11, 11, 1).to(device)
use_fp16 = True
model = GAT(11, 5, 1, heads=[4, 4, 6]).to(device)

model_path = "v3-"+task+"-11-gat.pth"
#model = GATConv(11, 11, 1, 1).to(device)

#if platform.system() == "Linux" and torch.cuda.is_available():
#    model = torch.compile(model)
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)
model.train()
print("Loading Data...")
af_dataset = TrainingGraphDataset(af_data_root+"dataset_af/", af_data_root+"result/")
data_loader = dgl.dataloading.GraphDataLoader(af_dataset, batch_size=32, shuffle=True)
#print("FINAL TEST LOADING...")
#test_data_loader = dgl.dataloading.GraphDataLoader(af_dataset, batch_size=64, shuffle=False)
#train_dataloader = DataLoader(af_dataset, batch_size=64)
print("Start training")
#scaler = GradScaler()
model.train()
for epoch in range(400):
    tot_loss = []
    tot_loss_v = 0
    for graph in data_loader:
        inputs = graph.ndata["feat"]
        label = graph.ndata["label"]
        optimizer.zero_grad()
        #with autocast(enabled=use_fp16):
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
    torch.cuda.empty_cache()
    #if epoch > 120:
    #    scheduler.step()
    print("Epoch : ", epoch," Mean : " , statistics.fmean(tot_loss), " Median : ", statistics.median(tot_loss), "loss : ", tot_loss_v)

print("final test start")
af_dataset = ValisationDataset(af_data_root+"dataset_af/", af_data_root+"result/")
model.eval()
acc_yes = 0
acc_no = 0
tot_el_yes = 0
tot_el_no = 0
mean_acc = 0
with torch.no_grad():
    for graph in af_dataset:
        inputs = graph.ndata["feat"]
        label = graph.ndata["label"]
        out = model(graph, inputs)
        predicted = (torch.sigmoid(out.squeeze())>0.5).float()
        one_acc_yes = sum(element1 == element2 == 1.0  for element1, element2 in zip(predicted, label)).item()
        one_acc_no = sum(element1 == element2 == 0.0   for element1, element2 in zip(predicted, label)).item()
        acc_yes += one_acc_yes
        acc_no += one_acc_no
        tot_yes = sum(element1 == 1.0  for element1 in label).item()
        tot_no = sum(element1 == 0.0   for element1 in label).item()
        tot_el_yes += tot_yes
        tot_el_no += tot_no
        mean_acc += ((one_acc_yes+one_acc_no)/(tot_yes+tot_no))

print("acc : ", (acc_yes+acc_no)/(tot_el_no+tot_el_yes) ,"acc yes : ", acc_yes/tot_el_yes, "acc no : ", acc_no/tot_el_no )
print("acc mean : ", mean_acc/len(af_dataset))
torch.save(model.state_dict(), model_path)
