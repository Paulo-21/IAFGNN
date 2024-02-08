import os
import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import ctypes
import multiprocessing as mp
import numpy as np
import networkx as nx
import time
import af_reader_py
import statistics
"""
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.utils import *
"""
import dgl
from sklearn.preprocessing import StandardScaler
from dgl.nn import GraphConv

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


def transfom_to_graph(label_path, n):
    f = open(label_path, 'r')
    data = f.read()
    target = [0.]*n
    for n in data.split(','):
        if n == '':
            continue
        target[int(n)] = 1.0
    return torch.tensor(target)


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
    #G = nx.Graph()
    i = 0
    map = {}
    att = []
    file = open(path).read().splitlines(True)
    for line in file:
        if line[:3] == "arg":
            line = line[4:-3]
            map[line] = i
            #G.add_node(i)
            i+=1
        elif line[:3] == "att":
            line = line[4:-3]
            s = line.split(",")
            att.append([map[s[0]], map[s[1]]])
            #G.add_edge(map[s[0]], map[s[1]])
    arg = list([s for s in range(0, i)])
    return att, arg

class CustumGraphDataset(Dataset):
    def __init__(self, af_dir, label_dir, use_cache = False):
        self.graph = iter(os.listdir(label_dir))
        self.af_dir = af_dir
        self.label_dir = label_dir
        
    def __len__(self):
        return len(self.graph)

    def __getitem__(self, idx):
        af_name = next(self.graph)
        af_path = os.path.join(self.af_dir,af_name)
        label_path = os.path.join(self.label_dir,af_name)
        tic = time.perf_counter()
        #arg, att = af_reader_py.reading_apx(af_path)
        #arg, att = af_reader_py.reading_cnf(af_path+".af")
        att1, att2, nb_el = af_reader_py.reading_cnf_for_dgl(af_path+".af")
        """
        arg , att = af_reader_py.reading_cnf(af_path+".af")
        print(af_name)
        
        print(att1)
        print(att2)
        print("1_________")       
        print(arg)
        print(att)        
        print("2----------")       

        print("finish")
        """
        toc = time.perf_counter()
        print(toc-tic , " seconds for RUST ")
        #tic = time.perf_counter()
        #arg, att = reading_cnf(af_path+".af")
        #toc = time.perf_counter()
        #print(toc-tic , " seconds PYTHON")
        target = transfom_to_graph(label_path, nb_el)
        #toc = time.perf_counter()
        #print(toc-tic , " seconds PYTHON")
        return att1, att2, target, af_name, nb_el
    

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, fc_features, num_classes, dropout=0.5):
        super(GCN, self).__init__()
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(128, 128, 128, 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss = nn.BCELoss()
sys.stdout.flush()
model.train()

for epoch in range(200):
    tot_loss = []
    af_dataset = CustumGraphDataset("../../../Documents/dataset_af/", "../../../Documents/result/")
    for i, item in enumerate(af_dataset):
        """
        nxg = nx.DiGraph()
        nxg.add_nodes_from(item[1])
        nxg.add_edges_from(item[0])
        #print("number of nodes : ", nxg.number_of_nodes())
        #graph = dgl.from_networkx(nxg)
        """
        if item[4] > 10000:
            continue
        tic = time.perf_counter()
        graph = dgl.graph((torch.tensor(item[0]),torch.tensor(item[1])))
        #print("Graph build in ", toc-tic , " sec")
        features_tensor = torch.Tensor(3)
        if os.path.isfile("features_tensor/" + "" + item[3]+".pt"):
            features_tensor = torch.load("features_tensor/" + "" + item[3]+".pt")
            #print("loaded in ", toc-tic , " sec")
        else:
            nxg = nx.DiGraph()
            nodes = list([s for s in range(0, item[4])])
            att = list([([s, item[1][i]]) for i, s in enumerate(item[0])])
            nxg.add_nodes_from(nodes)
            nxg.add_edges_from(att)
            #print("number of nodes : ", nxg.number_of_nodes())
            #graph = dgl.from_networkx(nxg)
            features  = calculate_node_features(nxg)
            features_tensor = torch.tensor(np.array([features[node] for node in nxg.nodes()]), dtype=torch.float32)
            torch.save(features_tensor, "features_tensor/" + "" + item[3]+".pt")
        
        if graph.number_of_nodes() < item[4]:
            graph.add_nodes(item[4] - graph.number_of_nodes())
        graph = dgl.add_self_loop(graph)
        num_rows_to_overwrite = features_tensor.size(0)
        num_columns_in_features = features_tensor.size(1)
        inputs = torch.randn(graph.number_of_nodes(), 128 , dtype=torch.float32)
        inputs_to_overwrite = inputs.narrow(0, 0, num_rows_to_overwrite).narrow(1, 0, num_columns_in_features)
        inputs_to_overwrite.copy_(features_tensor)
        toc = time.perf_counter()
        print("Finished in ", toc-tic , " sec")
        optimizer.zero_grad()
        out = model(graph, inputs)
        predicted = (torch.sigmoid(out.squeeze())).float()
        
        losse = loss(predicted, item[2])
        losse.backward()
        optimizer.step()
        tot_loss.append(losse.item())

        print("Epoch : ", epoch, " iter : ", i, losse.item())
        #if i % 100 == 99:
    
    print("Epoch : ", epoch," Mean : " , statistics.fmean(tot_loss), " Median : ", statistics.median(tot_loss))

