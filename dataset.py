import torch
from torch.utils.data import Dataset
from torchvision import datasets
import pandas as pd
import os
import networkx as nx

def transfom_to_graph(data, n):
    target = [0]*n
    for n in data.split(','):
        if n == '':
            continue
        target[int(n)] = 1
    return target

def read_apx(path):
    G = nx.Graph()
    i = 0
    map = {}
    file = open(path).read().splitlines(True)
    for line in file:
        if line.startswith("arg"):
            line = line.strip()
            line = line[4:]
            line = line[:-2]
            map[line] = i
            G.add_node(i)
            i+=1
        elif line.startswith("att"):
            line = line.strip()
            line = line.removeprefix("att(")
            line = line.removesuffix(").")
            s = line.split(",")
            G.add_edge(map[s[0]], map[s[1]])
    return G

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
        af = read_apx(af_path)
        f = open(label_path, 'r')
        label = f.read()
        target = transfom_to_graph(label, af.number_of_nodes())
        return af, target
    
af_dataset = CustumGraphDataset("../../../Documents/dataset/", "../../../Documents/result/")
for i, item in enumerate(af_dataset):
    z=2