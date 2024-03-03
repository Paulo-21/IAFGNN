import copy
import os
import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import DGLDataset
#from torch.utils.data import Dataset
#from torch.utils.data import DataLoader
import ctypes
import rustworkx as rx
import multiprocessing as mp
import numpy as np
import networkx as nx
import time
import af_reader_py
import dgl
from sklearn.preprocessing import StandardScaler
from dgl.nn import GraphConv
af_data_root = "../af_data/"

def graph_coloring(nx_G):
    coloring = rx.algorithms.coloring.greedy_color(nx_G, strategy='largest_first')
    return coloring

def calculate_node_features(nx_G, hcat, card, noselfatt, maxb, gr):
    coloring = graph_coloring(nx_G)
    page_rank = rx.pagerank(nx_G)
    closeness_centrality = rx.degree_centrality(nx_G)
    eigenvector_centrality = rx.eigenvector_centrality(nx_G, max_iter=10000)
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
    return torch.tensor(target, requires_grad=False).to(device)

def get_item(af_path):
    
    tic = time.perf_counter()
    #att1, att2, nb_el = af_reader_py.reading_cnf_for_dgl(af_path+".af")
    att1, att2, nb_el, hcat, card, noselfatt, maxb, gr = af_reader_py.reading_cnf_for_dgl_with_semantics(af_path)
    toc = time.perf_counter()
    #print(toc-tic , " seconds for RUST ")
    
    graph = dgl.graph((torch.tensor(att1),torch.tensor(att2)), device=device)#.to(device)
    #print("Graph build in ", toc-tic , " sec")
    features_tensor = torch.Tensor(3).to(device)
    #if os.path.isfile(af_data_root+"features_tensor/" + "" + af_name+".pt"):
        #features_tensor = torch.load(af_data_root+"features_tensor/" + "" + af_name+".pt").to(device)
        #print("loaded in ", toc-tic , " sec")
    #else:
    nxg = rx.DiGraph()
    nodes = list([s for s in range(0, nb_el)])
    att = list([([s, att2[i]]) for i, s in enumerate(att1)])
    nxg.add_nodes_from(nodes)
    nxg.add_edges_from(att)
    features  = calculate_node_features(nxg, hcat, card, noselfatt, maxb, gr)
    features_tensor = torch.tensor(np.array([features[node] for node in nxg.nodes()]), dtype=torch.float32).to(device)
    #    torch.save(features_tensor, af_data_root+"features_tensor/" + "" + af_name+".pt")
    
    if graph.number_of_nodes() < nb_el:
        graph.add_nodes(nb_el - graph.number_of_nodes())
    
    graph = dgl.add_self_loop(graph)
    num_rows_to_overwrite = features_tensor.size(0)
    num_columns_in_features = features_tensor.size(1)
    inputs = torch.randn(graph.number_of_nodes(), 128 , dtype=torch.float32, requires_grad=False).to(device)
    inputs_to_overwrite = inputs.narrow(0, 0, num_rows_to_overwrite).narrow(1, 0, num_columns_in_features)
    inputs_to_overwrite.copy_(features_tensor)
    return graph, inputs

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

file = sys.argv[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(128, 128, 128, 1).to(device)

graph, inputs = get_item(af_path=file)
with torch.no_grad():
    out = model(graph, inputs)
    predicted = (torch.sigmoid(out.squeeze())>0.9).float()
