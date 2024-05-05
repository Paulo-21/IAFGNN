import os
import torch
import af_reader_py
#from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
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
    return target

class TrainingLinearDataset(Dataset):
    def __init__(self, task, max_arg=MAX_ARG, device="cpu"):
        self.task = task
        self.max_arg = max_arg
        self.device=device
        self.process()
        #super().__init__(name="Af dataset")
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
                    #gs = af_reader_py.compute_only_gs_w_gr_sa_ed(af_path)
                    gs = af_reader_py.compute_only_gs_w_gr_sa_ed_eb(af_path)
                    label = torch.tensor(transfom_to_graph(label_path, len(gs), device=self.device), device=self.device)
                    self.labels.append(label.unsqueeze(1))
                    self.instances.append(torch.tensor(gs,requires_grad=True ,device=self.device))

    def __getitem__(self, idx:int):
        return (self.instances[idx], self.labels[idx])
    
def get_dataset_kan(task, max_arg=MAX_ARG, device="cpu"):
    task = task
    max_arg = max_arg
    device=device
    list_year_dir = ["2017"]
    af_dir = af_data_root+"dataset_af"
    label_dir = result_root+"result_"+task
    instances = []
    labels = []
    list_unique_file = []
    for year in list_year_dir:
        iter = os.listdir(label_dir +"_"+ year)
        for f in iter:
            true_name = f.replace(".apx", "")
            true_name = true_name.replace(".af", "")
            true_name = true_name.replace(".tgf", "")
            true_name = true_name.replace(".old", "")
    
            if true_name not in list_unique_file:
                list_unique_file.append(true_name)
                af_path = af_dir+"_"+year+"/"+f
                label_path = label_dir+"_"+year+"/"+f
                #gs = af_reader_py.compute_only_gs(af_path)                   
                #gs = af_reader_py.compute_only_gs_w_gr(af_path)
                gs = af_reader_py.compute_only_gs_w_gr_sa_ed(af_path)
                #gs = af_reader_py.compute_only_gs_w_gr_sa_ed_eb(af_path)
                if len(gs) > 1000:
                    continue
                label = transfom_to_graph(label_path, len(gs), device=device)
                labels.extend(label)
                instances.extend(gs)
                
    return (torch.tensor(instances, device=device), torch.tensor(labels, device=device, dtype=torch.long))


def get_dataset_kan_test(task, max_arg=MAX_ARG, device="cpu"):
    task = task
    max_arg = max_arg
    device=device
    list_year_dir = ["2023"]
    af_dir = af_data_root+"dataset_af"
    label_dir = result_root+"result_"+task
    instances = []
    labels = []
    list_unique_file = []
    for year in list_year_dir:
        iter = os.listdir(label_dir +"_"+ year)
        for f in iter:
            true_name = f.replace(".apx", "")
            true_name = true_name.replace(".af", "")
            true_name = true_name.replace(".tgf", "")
            true_name = true_name.replace(".old", "")

            if true_name not in list_unique_file:
                list_unique_file.append(true_name)
                af_path = af_dir+"_"+year+"/"+f
                label_path = label_dir+"_"+year+"/"+f
                #gs = af_reader_py.compute_only_gs(af_path)                   
                #gs = af_reader_py.compute_only_gs_w_gr(af_path)
                gs = af_reader_py.compute_only_gs_w_gr_sa_ed(af_path)
                if len(gs) > 1000:
                    continue
                #gs = af_reader_py.compute_only_gs_w_gr_sa_ed_eb(af_path)
                label = transfom_to_graph(label_path, len(gs), device=device)
                labels.extend(label)
                instances.extend(gs)
    return (torch.tensor(instances, device=device), torch.tensor(labels, device=device, dtype=torch.long))
