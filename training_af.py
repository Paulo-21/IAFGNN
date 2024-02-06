import os
import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import networkx as nx
import numpy as np
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


def transfom_to_graph(data, n):
    target = [0.]*n
    for n in data.split(','):
        if n == '':
            continue
        target[int(n)] = 1.0
    return torch.tensor(target)

def read_apx(path):
    #G = nx.Graph()
    i = 0
    map = {}
    att = []
    file = open(path).read().splitlines(True)
    for line in file:
        if line.startswith("arg"):
            line = line.strip()
            line = line[4:]
            line = line[:-2]
            map[line] = i
            #G.add_node(i)
            i+=1
        elif line.startswith("att"):
            line = line.strip()
            line = line.removeprefix("att(")
            line = line.removesuffix(").")
            s = line.split(",")
            att.append([map[s[0]], map[s[1]]])
            #G.add_edge(map[s[0]], map[s[1]])
    print("finish")
    arg = list([s for s in range(0, i)])
    return att, arg

class CustumGraphDataset(Dataset):
    def __init__(self, af_dir, label_dir):
        self.graph = iter(os.listdir(label_dir))
        self.af_dir = af_dir
        self.label_dir = label_dir
    def __len__(self):
        return len(self.graph)

    def __getitem__(self, idx):
        af_name = next(self.graph)
        af_path = os.path.join(self.af_dir,af_name)
        label_path = os.path.join(self.label_dir,af_name)
        att, arg = read_apx(af_path)
        f = open(label_path, 'r')
        label = f.read()
        target = transfom_to_graph(label, len(arg))
        return att, arg, target
    

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
    af_dataset = CustumGraphDataset("../../../Documents/dataset/", "../../../Documents/result/")
    for i, item in enumerate(af_dataset):
        print(epoch, " ", i)
        nxg = nx.DiGraph()
        nxg.add_edges_from(item[0])
        nxg.add_nodes_from(item[1])
        print("number of nodes : ", nxg.number_of_nodes())
        #if nxg.number_of_nodes() > 8000:
        #    continue
        graph = dgl.from_networkx(nxg)
        graph = dgl.add_self_loop(graph)
        features  = calculate_node_features(nxg)
        features_tensor = torch.tensor(np.array([features[node] for node in nxg.nodes()]), dtype=torch.float32)
        num_rows_to_overwrite = features_tensor.size(0)
        num_columns_in_features = features_tensor.size(1)
        inputs = torch.randn(graph.number_of_nodes(), 128 , dtype=torch.float32)
        inputs_to_overwrite = inputs.narrow(0, 0, num_rows_to_overwrite).narrow(1, 0, num_columns_in_features)
        inputs_to_overwrite.copy_(features_tensor)
        optimizer.zero_grad()
        out = model(graph, inputs)
        predicted = (torch.sigmoid(out.squeeze())).float()
        losse = loss(predicted, item[2])
        losse.backward()
        print(losse)
        optimizer.step()

