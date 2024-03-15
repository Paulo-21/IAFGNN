import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
import af_reader_py
import dgl
from sklearn.preprocessing import StandardScaler
from dgl.nn import GraphConv

af_data_root = "../af_data/"

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
    att1, att2, nb_el = af_reader_py.reading_cnf_for_dgl(af_path)
    toc = time.perf_counter()
    print(toc-tic , " seconds for RUST ")
    tic = time.perf_counter()
    tic3 = time.perf_counter()
    graph = dgl.graph((att1,att2), num_nodes=nb_el, device=device)#.to(device)
    print("dgl ",time.perf_counter()-tic)

    tic = time.perf_counter()
    raw_features = af_reader_py.compute_features(af_path, 10000, 0.00001 )
    print("Python wrappe the function : ", time.perf_counter()-tic, " sec")

    #features  = calculate_node_features(nxg, hcat, card, noselfatt, maxb, gr, eig)
    tic = time.perf_counter()
    scaler = StandardScaler()
    features = scaler.fit_transform(raw_features)
    print("standscal : ", time.perf_counter()-tic)
    #features_tensor = torch.tensor(np.array([features[node] for node in nxg.nodes()]), dtype=torch.float32).to(device)
    tic = time.perf_counter()
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    #    torch.save(features_tensor, af_data_root+"features_tensor/" + "" + af_name+".pt")
    if graph.number_of_nodes() < nb_el:
        graph.add_nodes(nb_el - graph.number_of_nodes())
    
    graph = dgl.add_self_loop(graph)
    num_rows_to_overwrite = features_tensor.size(0)
    num_columns_in_features = features_tensor.size(1)
    inputs = torch.randn(graph.number_of_nodes(), 128 , dtype=torch.float32, requires_grad=False).to(device)
    inputs_to_overwrite = inputs.narrow(0, 0, num_rows_to_overwrite).narrow(1, 0, num_columns_in_features)
    inputs_to_overwrite.copy_(features_tensor)
    print("end ", time.perf_counter()-tic)
    
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
#model = GCN(128, 128, 128, 1).to(device)
tic = time.perf_counter()
graph, inputs = get_item(af_path=file)
print("get item ", time.perf_counter()-tic)
"""
with torch.no_grad():
    out = model(graph, inputs)
    predicted = (torch.sigmoid(out.squeeze())>0.9).float()
"""