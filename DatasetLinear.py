import os
import torch
import af_reader_py
from dgl.data import DGLDataset
#from sklearn.preprocessing import StandardScaler
#from torch.utils.data import DataLoader, Dataset
MAX_ARG = 200000
af_data_root = "../af_dataset/"
result_root = "../af_dataset/all_result/"

def transfom_to_graph(label_path, n, device="cpu"):
    f = open(label_path, 'r')
    data = f.read()
    target = [0.]*n
    for n in data.split(','):
        if n == '':
            continue
        target[int(n)] = 1.0
    return torch.tensor(target, requires_grad=False, device=device)

class TrainingLinearDataset(DGLDataset):
    def __init__(self, task, max_arg=MAX_ARG, device="cpu"):
        self.task = task
        self.max_arg = max_arg
        self.device=device
        super().__init__(name="Af dataset")
    def __len__(self):
        return len(self.instances)
    def process(self):
        list_year_dir = ["2017"]
        self.af_dir = af_data_root+"dataset_af"
        self.label_dir = result_root+"result_"+self.task
        self.instances = []
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
                    af_path = self.af_dir+"_"+year+"/"+f
                    label_path = self.label_dir+"_"+year+"/"+f
                    #gs = af_reader_py.compute_only_gs(af_path)                   
                    #gs = af_reader_py.compute_only_gs_w_gr(af_path)
                    gs = af_reader_py.compute_only_gs_w_gr_sa_ed(af_path)
                    label = transfom_to_graph(label_path, len(gs), device=self.device)
                    self.labels.append(label.unsqueeze(1))
                    self.instances.append(torch.tensor(gs,requires_grad=True ,device=self.device))

    def __getitem__(self, idx:int):
        #r = random.randint(0,len(self.instances[idx])-1)
        return (self.instances[idx], self.labels[idx])
        #return (self.instances[idx][r], torch.tensor([self.labels[idx][r]], device=self.device))
