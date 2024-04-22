import dgl
import torch 
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from dgl.nn import GraphConv
from sklearn.preprocessing import StandardScaler
#from dgl.data import DGLDataset
from torch.utils.data import Dataset
import json
import os
import af_reader_py

af_data_root = "../af_dataset/"
result_root = "../af_dataset/all_result/"
MAX_ARG = 200000

def read_af_input(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    attacks = []
    args = []
    for line in lines:
        # Ignore comment lines
        if line.startswith('#'):
            continue

        # Split the line into parts
        parts = line.split()

        # If it's a p-line, extract the number of arguments and create args array
        if len(parts) > 0 and parts[0] == 'p' and parts[1] == 'af':
            num_args = int(parts[2])
            args = list([str(s) for s in range(1, num_args + 1)])

        # If it's an attack line, add the attack to the list of attacks
        elif len(parts) == 2:
            i, j = parts[0], parts[1]
            attacks.append([i, j])

    return args, attacks

def transfom_to_graph(label_path, n, device="cpu"):
    f = open(label_path, 'r')
    data = f.read()
    target = [0.]*n
    for n in data.split(','):
        if n == '':
            continue
        target[int(n)] = 1.0
    return torch.tensor(target, requires_grad=False, device=device)

def light_get_item(af_path, features_path, device= "cpu"):
    #nxg, mapping = reindex_nodes(nxg)
    att1, att2, nb_el = af_reader_py.reading_file_for_dgl(af_path)
    graph = dgl.graph((torch.tensor(att1),torch.tensor(att2)), num_nodes = nb_el, device=device)
    graph = dgl.add_self_loop(graph)
    inputs = torch.load(features_path, map_location=device)
    return graph, inputs, graph.number_of_nodes()

def get_item(af_path, features_path, device="cpu", max_arg=MAX_ARG):
    args, atts = af_reader_py.reading_cnf(af_path)
    nxg = nx.DiGraph()
    nxg.add_nodes_from(args)
    nxg.add_edges_from(atts)
    #nxg, mapping = reindex_nodes(nxg)
    #raw_features = calculate_node_features(nxg)
    raw_features = af_reader_py.compute_features_wo_gs(af_path, 10000, 0.000001)
    att1, att2, nb_el = af_reader_py.reading_file_for_dgl(af_path)
    graph = dgl.graph((torch.tensor(att1),torch.tensor(att2)), num_nodes = nb_el, device=device)
    #graph = dgl.from_networkx(nxg, device=device)
    graph = dgl.add_self_loop(graph)
    inputs = torch.tensor(np.array([raw_features[node] for node in nxg.nodes()]),device=device , dtype=torch.float)
    del nxg
    torch.save(inputs, features_path)
    
    return graph, inputs, graph.number_of_nodes()


class ValisationDataset(Dataset):
    def __init__(self,  task, device = "cpu"):
        self.task = task
        self.device = device
        #list_year_dir = ["2017", "2023"]
        list_year_dir = ["2023"]
        self.af_dir = af_data_root+"dataset_af"
        self.label_dir = result_root+"result_"+self.task
        self.graphs = []
        self.labels = []
        self.labels_path = []
        self.features_paths = []
        self.af_paths = []
        print("dataset device  :", self.device)
        for year in list_year_dir:
            iter = os.listdir(self.label_dir +"_"+ year)
            for f in iter:
                af_path = self.af_dir+"_"+year+"/"+f
                args, atts = af_reader_py.reading_cnf(af_path)
                if len(args) >= 200000:
                    continue
                self.af_paths.append(af_path)
                self.labels_path.append(self.label_dir+"_"+year+"/"+f)
                self.features_paths.append(af_data_root+"all_features_old/"+year+"/"+f+".pt")
                
    def __len__(self):
        return len(self.af_paths)
    def __getitem__(self, idx:int):
        print("ID : ", idx)
        if os.path.exists(self.features_paths[idx]):
            print("Exist : ", self.af_paths[idx])
            graph, features, nb_el = light_get_item(self.af_paths[idx], self.features_paths[idx], device=self.device)
        else:
            print("No Exist : ", self.af_paths[idx])
            graph, features, nb_el = get_item(self.af_paths[idx], self.features_paths[idx], device=self.device)
        graph.ndata["feat"] = features
        graph.ndata["label"] = transfom_to_graph(self.labels_path[idx], nb_el, device=self.device)
            
        return graph
    
def test(model, task, device="cpu", rand=False):
    print("Start Loading")
    af_dataset = ValisationDataset(task=task, device=device)
    model.eval()
    acc_yes = 0
    acc_no = 0
    tot_el_yes = 0
    tot_el_no = 0
    mean_acc = 0
    mean_acc_yes = 0
    mean_acc_no = 0
    tot_yes_count = 0
    tot_no_count = 0
    threshold_path = "../AFGCNv2/thresholds.json"
    thresholds = load_thresholds(threshold_path)
    threshold = thresholds[task]
    print("Start Testing ...")
    print(task)
    with torch.no_grad():
        i = 0
        for graph in af_dataset:
            i+=1
            inputs = graph.ndata["feat"]
            if rand == True:
                inputs_rand = torch.randn(graph.number_of_nodes(), 128 , dtype=torch.float, device=device)
                num_rows_to_overwrite = inputs.size(0)
                num_columns_in_features = inputs.size(1)
                inputs_to_overwrite = inputs_rand.narrow(0, 0, num_rows_to_overwrite).narrow(1, 0, num_columns_in_features)
                inputs_to_overwrite.copy_(inputs)
                inputs = inputs_rand
            label = graph.ndata["label"]
            out = model(graph, inputs)
            predicted = (torch.sigmoid(out.squeeze())>threshold).float()
            one_acc_yes = sum(element1 == element2 == 1.0  for element1, element2 in zip(predicted, label)).item()
            one_acc_no = sum(element1 == element2 == 0.0   for element1, element2 in zip(predicted, label)).item()
            acc_yes += one_acc_yes
            acc_no += one_acc_no
            tot_yes = sum(element1 == 1.0  for element1 in label).item()
            tot_no = sum(element1 == 0.0   for element1 in label).item()
            tot_el_yes += tot_yes
            tot_el_no += tot_no
            mean_acc += ((one_acc_yes+one_acc_no)/(tot_yes+tot_no))
            if tot_yes != 0:
                mean_acc_yes += ((one_acc_yes)/(tot_yes))
                tot_yes_count +=1
            if tot_no != 0:
                mean_acc_no += ((one_acc_no)/(tot_no))
                tot_no_count +=1

    print("acc : ", (acc_yes+acc_no)/(tot_el_no+tot_el_yes) ,"acc yes : ", acc_yes/tot_el_yes, "acc no : ", acc_no/tot_el_no )
    print("acc mean : ", mean_acc/len(af_dataset), " acc mean y : ", mean_acc_yes/tot_yes_count, " acc mean no : ", mean_acc_no/tot_no_count)
    print(task)

class AFGCNModel(nn.Module):
    def __init__(self, in_features, hidden_features, fc_features, num_classes, dropout=0.5):
        super(AFGCNModel, self).__init__()
        self.layer1 = GraphConv(in_features, hidden_features)
        self.layer2 = GraphConv(hidden_features, hidden_features)
        self.layer3 = GraphConv(hidden_features, hidden_features)
        self.layer4 = GraphConv(hidden_features, fc_features)
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
def graph_coloring(nx_G):
    coloring = nx.algorithms.coloring.greedy_color(nx_G, strategy='largest_first')
    return coloring

def calculate_node_features(nx_G):
    coloring = graph_coloring(nx_G)
    page_rank = nx.pagerank(nx_G)
    closeness_centrality = nx.degree_centrality(nx_G)
    eigenvector_centrality = nx.eigenvector_centrality(nx_G, max_iter=10000)
    in_degrees = nx_G.in_degree()
    out_degrees = nx_G.out_degree()

    raw_features = {}
    for node in nx_G.nodes():
        raw_features[node] = [
            coloring[node],
            page_rank[node],
            closeness_centrality[node],
            eigenvector_centrality[node],
            in_degrees[node],
            out_degrees[node],
        ]

    # Normalize the features
    scaler = StandardScaler()
    nodes = list(nx_G.nodes())
    feature_matrix = scaler.fit_transform([raw_features[node] for node in nodes])

    # Create the normalized features dictionary
    normalized_features = {node: feature_matrix[i] for i, node in enumerate(nodes)}

    return normalized_features

def reindex_nodes(graph):
    mapping = {node.strip(): index for index, node in enumerate(graph.nodes())}
    return nx.relabel_nodes(graph, mapping), mapping

def load_thresholds(file_path):
    with open(file_path, 'r') as f:
        thresholds = json.load(f)
    return thresholds
# Define a function to load the model checkpoint and retrieve the associated loss
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
 
task = "DC-ST"
device = "cuda"
net = AFGCNModel(128, 128, 128, 1).to(device)
checkpoint_path = "/home/paul/AFGCNv2/"+ task + ".pth"

load_checkpoint(net, checkpoint_path)

test(net, task, rand=True, device=device)