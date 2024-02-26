import copy
import os
import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import ctypes
import multiprocessing as mp
import numpy as np
import networkx as nx
import time
import af_reader_py
import statistics
import platform
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.utils import *

#import dgl
from sklearn.preprocessing import StandardScaler
#from dgl.nn import GraphConv
af_data_root = "../af_data/"
def graph_coloring(nx_G):
    coloring = nx.algorithms.coloring.greedy_color(nx_G, strategy='largest_first')
    return coloring

def calculate_node_features(nx_G, hcat, card, noselfatt, maxb, gr):
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
            hcat[node], 
            card[node], 
            noselfatt[node], 
            maxb[node],
            gr[node]
        ]

    # Normalize the features
    scaler = StandardScaler()
    nodes = list(nx_G.nodes())
    feature_matrix = scaler.fit_transform([raw_features[node] for node in nodes])

    # Create the normalized features dictionary
    normalized_features = {node: feature_matrix[i] for i, node in enumerate(nodes)}

    return normalized_features

def transfom_to_graph(label_path, n):
    f = open(label_path, 'r')
    data = f.read()
    target = [0.]*n
    for n in data.split(','):
        if n == '':
            continue
        target[int(n)] = 1.0
    return torch.tensor(target).to(device)


def reading_cnf(path):
    file = open(path).read().splitlines(True)
    first = file.pop(0).split(' ')
    nb_arg = int(first[2])
    args = list([s for s in range(0, nb_arg)])
    att = []
    for l in file:
        s = l.split(' ')
        att.append([int(s[0]), int(s[1])])

    return args, att
    
def read_apx(path):
    i = 0
    map = {}
    att = []
    file = open(path).read().splitlines(True)
    for line in file:
        if line[:3] == "arg":
            line = line[4:-3]
            map[line] = i
            i+=1
        elif line[:3] == "att":
            line = line[4:-3]
            s = line.split(",")
            att.append([map[s[0]], map[s[1]]])
    arg = list([s for s in range(0, i)])
    return att, arg

def get_item(af_name, af_dir, label_dir):
    
    af_path = os.path.join(af_dir,af_name)
    label_path = os.path.join(label_dir,af_name)
    tic = time.perf_counter()
    #att1, att2, nb_el = af_reader_py.reading_cnf_for_dgl(af_path+".af")
    att1, att2, nb_el, hcat, card, noselfatt, maxb, gr = af_reader_py.reading_cnf_for_dgl_with_semantics(af_path+".af")
    if nb_el > 10000:
        #print("pop")
        return [[], [], [], nb_el]
    
    toc = time.perf_counter()
    #print(toc-tic , " seconds for RUST ")
    target = transfom_to_graph(label_path, nb_el)
    
    graph = dgl.graph((torch.tensor(att1),torch.tensor(att2))).to(device)
    #print("Graph build in ", toc-tic , " sec")
    features_tensor = torch.Tensor(3).to(device)
    if os.path.isfile(af_data_root+"features_tensor/" + "" + af_name+".pt"):
        features_tensor = torch.load(af_data_root+"features_tensor/" + "" + af_name+".pt").to(device)
        #print("loaded in ", toc-tic , " sec")
    else:
        nxg = nx.DiGraph()
        nodes = list([s for s in range(0, nb_el)])
        att = list([([s, att2[i]]) for i, s in enumerate(att1)])
        nxg.add_nodes_from(nodes)
        nxg.add_edges_from(att)
        features  = calculate_node_features(nxg, hcat, card, noselfatt, maxb, gr)
        features_tensor = torch.tensor(np.array([features[node] for node in nxg.nodes()]), dtype=torch.float32).to(device)
        torch.save(features_tensor, af_data_root+"features_tensor/" + "" + af_name+".pt")
    
    if graph.number_of_nodes() < nb_el:
        graph.add_nodes(nb_el - graph.number_of_nodes())
    
    graph = dgl.add_self_loop(graph)
    num_rows_to_overwrite = features_tensor.size(0)
    num_columns_in_features = features_tensor.size(1)
    inputs = torch.randn(graph.number_of_nodes(), 128 , dtype=torch.float32).to(device)
    inputs_to_overwrite = inputs.narrow(0, 0, num_rows_to_overwrite).narrow(1, 0, num_columns_in_features)
    inputs_to_overwrite.copy_(features_tensor)
    return graph, inputs, target, nb_el

class CustumGraphDataset(Dataset):
    def __init__(self, af_dir, label_dir):
        self.graph = os.listdir(label_dir)
        self.af_dir = af_dir
        self.label_dir = label_dir
        self.cache_length = len(self.graph) # number of samples you want to cache
        self.cache = [None]*len(self.graph)
    def __len__(self):
        return len(self.graph)

    def __getitem__(self, idx:int):
        if idx <= len(self.cache) :
            if self.cache[idx] is None:
                x = get_item(self.graph[idx], self.af_dir, self.label_dir)
                self.cache[idx] = (copy.deepcopy(x))
                return x
        # hack to see if cache slot has changed since initialisation
        x = self.cache[idx] # get float32 image tensor from cache
        return x

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, fc_features, num_classes, dropout=0.5):
        super(GCN, self).__init__()
        self.layer1 = GraphConv(in_features, hidden_features)
        self.layer2 = GraphConv(hidden_features, hidden_features)
        self.layer3 = GraphConv(hidden_features, hidden_features)
        self.layer4 = GraphConv(hidden_features, hidden_features)
        self.layer5 = GraphConv(hidden_features, fc_features)
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
        h = self.layer5(g, h + inputs)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.fc(h)
        return h.squeeze()  # Remove the last dimension

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = GCN(128, 128, 128, 1).to(device)
if platform.system() == "Linux":
    model = torch.compile(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss = nn.BCELoss()
model.train()
af_dataset = CustumGraphDataset(af_data_root+"dataset_af/", af_data_root+"result/")

#train_dataloader = DataLoader(af_dataset, batch_size=64)
for epoch in range(200):
    tot_loss = [0]*len(af_dataset)
    tot_loss_v = 0
    for i, item in enumerate(af_dataset):
        
        if item[3] > 10000:
            #print("plp")
            continue
        #tic = time.perf_counter()
        graph = item[0]
        inputs = item[1]
        optimizer.zero_grad()
        out = model(graph, inputs)
        predicted = (torch.sigmoid(out.squeeze())).float()
        losse = loss(predicted, item[2])
        losse.backward()
        optimizer.step()
        #toc = time.perf_counter()
        tot_loss[i] = losse.item()
        tot_loss_v += losse.item()
        if i % 50 == 49:
            #print("Finished in ", toc-tic , " sec")
            print("Epoch : ", epoch, " iter : ", i, losse.item())
                
    print("Epoch : ", epoch," Mean : " , statistics.fmean(tot_loss), " Median : ", statistics.median(tot_loss), " ", tot_loss_v )

