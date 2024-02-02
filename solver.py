import dgl
import torch 
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import argparse
from dgl.nn import GraphConv
from sklearn.preprocessing import StandardScaler
import json
import os
import sys

BLANK  = 0
IN     = 1
OUT    = 2

def solve(adj_matrix):
    
    #set up labelling
    labelling = np.zeros((adj_matrix.shape[0]), np.int8)

    #find all unattacked arguments
    a = np.sum(adj_matrix, axis=0) == 0
    unattacked_args = np.nonzero(a)[0]

    #label them in
    labelling[unattacked_args] = IN
    cascade = True
    while cascade:
        #find all outgoing attacks)
        new_attacks = np.unique(np.nonzero(adj_matrix[unattacked_args,:])[1])
        new_attacks_l = np.array([i for i in new_attacks if labelling[i] != OUT])
        
        #label those out
        if len(new_attacks_l) > 0:
            labelling[new_attacks_l] = OUT
            affected_idx = np.unique(np.nonzero(adj_matrix[new_attacks_l,:])[1])
        else:
            affected_idx = np.zeros((0), dtype='int64')

        #find any arguments that have all attackers labelled out        
        all_outs = []
        for idx in affected_idx:
            incoming_attacks = np.nonzero(adj_matrix[:,idx])[0]
            if(np.sum(labelling[incoming_attacks] == OUT) == len(incoming_attacks)):
                all_outs.append(idx)

        #label those in
        if len(all_outs) > 0:
            labelling[np.array(all_outs)] = IN
            unattacked_args = np.array(all_outs)
        else:
            cascade = False
    
    #print grounded extension     
    in_nodes = np.nonzero(labelling == IN)[0]
    return in_nodes

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
        if parts[0] == 'p' and parts[1] == 'af':
            num_args = int(parts[2])
            args = list([str(s) for s in range(1, num_args + 1)])

        # If it's an attack line, add the attack to the list of attacks
        elif len(parts) == 2:
            i, j = parts[0], parts[1]
            print(part)
            attacks.append([i, j])

    return args, attacks

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

def read_apx_input(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    attacks = []
    attacks_uncert = []
    attacks_uncert_node = []
    args = []
    args_uncert = []
    num_args = 0
    for line in lines:
        # Ignore comment lines
        if line.startswith('#'):
            continue

        # Split the line into parts
        parts = line.split()
        if line.startswith("arg"):
            num_args += 1
            args.append(str(num_args))
        elif line.startswith("?arg"):
            num_args += 1
            args_uncert.append(str(num_args))
        elif line.startswith("att") or line.startswith("?att"):
            if line.startswith("?att"):
                line = line.replace('?att(a', '')
                line = line.replace(').', '')
                line = line.replace(',a', ' ')
                part = line.split()
                if part[0] in args or part[1] in args:
                    attacks_uncert_node.append([part[0],part[1]])
                else:
                    attacks_uncert.append([part[0],part[1]])

            else:
                line = line.replace('att(a', '')
                line = line.replace(').', '')
                line = line.replace(',a', ' ')
                part = line.split()
                if part[0] in args or part[1] in args:
                    attacks_uncert_node.append([part[0],part[1]])
                else:
                    attacks_uncert.append([part[0],part[1]])


    return args, args_uncert, attacks, attacks_uncert_node, attacks_uncert
def main(cmd_args):
    
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))) 
    
    args, args_uncert ,atts, atts_uncer_nodes, atts_uncer = read_apx_input("A-inc/BA_60_60_3_10_att_inc.apx")
    nxg = nx.DiGraph()
    nxg.add_nodes_from(args, weight = 1)
    nxg.add_nodes_from(args_uncert, weight = 2)
    nxg.add_edges_from(atts, weight = 1)
    nxg.add_edges_from(atts_uncer_nodes, weight = 2)
    nxg.add_edges_from(atts_uncer, weight = 3)
    nxg, mapping = reindex_nodes(nxg)
    graph = dgl.from_networkx(nxg)
    graph = dgl.add_self_loop(graph)
    print(graph)

# Define a function to load the model checkpoint and retrieve the associated loss
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_features', type=int, default=128 , help='number of input features')
    parser.add_argument('--filepath', type=str, default='' , help='file')
    parser.add_argument('--task', type=str, default='' , help='task')
    parser.add_argument('--argument', type=str, default='' , help='argument')
    parser.add_argument('--thresholds_file', type=str, default='thresholds.json', help='path to the thresholds JSON file')

    args = parser.parse_args()
    main(args)