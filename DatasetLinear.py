import csv
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
                    print(f)
                    list_unique_file.append(true_name)
                    af_path = self.af_dir+"_"+year+"/"+f
                    label_path = self.label_dir+"_"+year+"/"+f
                    gs = get_features(af_path)

                    if len(gs) >= MAX_ARG:
                        continue
                    label = torch.tensor(transfom_to_graph(label_path, len(gs), device=self.device), dtype=torch.float32, device=self.device)
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
                gs = get_features(af_path)
                if len(gs) > MAX_ARG:
                    continue
                label = transfom_to_graph(label_path, len(gs), device=device)
                labels.extend(label)
                instances.extend(gs)

    return (torch.tensor(instances, device=device), torch.tensor(labels, device=device, dtype=torch.long))

class ValisationDataset(Dataset):
    def __init__(self, af_dir, label_dir, task, device = "cpu"):
        self.label_dir = label_dir
        self.af_dir = af_dir
        self.task = task
        self.device = device
        #list_year_dir = ["2017", "2023"]
        list_year_dir = ["2023"]
        self.af_dir = af_data_root+"dataset_af"
        self.label_dir = result_root+"result_"+self.task
        self.instances = []
        self.labels = []
        list_unique_file = []
        print("device  :", self.device)
        for year in list_year_dir:
            iter = os.listdir(self.label_dir +"_"+ year)
            for f in iter:
                true_name = f.replace(".apx", "")
                true_name = true_name.replace(".af", "")
                true_name = true_name.replace(".tgf", "")
                true_name = true_name.replace(".old", "")
                if true_name not in list_unique_file:
                    af_path = self.af_dir+"_"+year+"/"+f
                    label_path = self.label_dir+"_"+year+"/"+f
                    features_path = af_data_root+"all_features_14/"+year+"/"+f+".pt"
                    list_unique_file.append(true_name)
                    gs = get_features(af_path)
                if len(gs) > 10000:
                    continue
                label = transfom_to_graph(label_path, len(gs), device=device)
                self.labels.append(torch.tensor(label, device=self.device))
                self.instances.append(torch.tensor(gs, device=self.device))
                
    def __len__(self):
        return len(self.instances)
    def __getitem__(self, idx:int):
        return (self.instances[idx], self.labels[idx])

def test(model, task, device="cpu", rand=False):
    model.eval()
    af_dataset = ValisationDataset(af_data_root+"dataset_af/", af_data_root+"result/", task=task, device=device)
    acc_yes = 0
    acc_no = 0
    tot_el_yes = 0
    tot_el_no = 0
    mean_acc = 0
    mean_acc_yes = 0
    mean_acc_no = 0
    tot_yes_count = 0
    tot_no_count = 0
    with torch.no_grad():
        for (inputs, label) in af_dataset:
            out = model(inputs)
            predicted = (out.squeeze()>0.5).float()
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
                tot_yes_count += 1
            if tot_no != 0:
                mean_acc_no += ((one_acc_no)/(tot_no))
                tot_no_count += 1

    print("acc : ", (acc_yes+acc_no)/(tot_el_no+tot_el_yes) ,"acc yes : ", acc_yes/tot_el_yes, "acc no : ", acc_no/tot_el_no )
    print("acc mean : ", mean_acc/len(af_dataset), " acc mean y : ", mean_acc_yes/tot_yes_count, " acc mean no : ", mean_acc_no/tot_no_count)
    print(task)
    
    dir = "../benchmarks/main/"
    nb_correct = 0
    instances_answer = get_reponse(task=task)
    for names in instances_answer:
        instances_name, answer, arg_id = instances_answer[names]
        print(instances_name)
        filepath = os.path.join(dir, instances_name)
        feat = get_features(filepath)
        print("FEAT")
        inputs = torch.tensor(feat[arg_id-1], device=device)
        out = (model(inputs) > 0.5)
        print("FINISH")
        if out == answer:
            nb_correct += 1
        
    print(task, " score : ", nb_correct)

def get_features(af_path):
    #gs = af_reader_py.compute_only_gs_w_gr_sa_ed(af_path)
    gs = af_reader_py.compute_only_gs_w_gr_sa_ed_eb(af_path)
    #gs = af_reader_py.compute_only_gs_w_gr_sa_ed_perso(af_path)
    #gs = af_reader_py.compute_only_gs_w_gr_sa_ed_perso_mod(af_path)
    return gs
def get_reponse(task):
    reader = open("../reduce_results2023.csv", 'r')
    cr = csv.reader(reader, delimiter=';')
    instances_answer = {}
    for row in cr:
        row_task = row[0]
        instance_name = row[1]
        arg_id = row[2]
        truth_answer = eval(row[3])
        if row_task != task:
            continue
        instances_answer[instance_name] = (instance_name, truth_answer, int(arg_id))
    return instances_answer

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
                gs = get_features(af_path)
                if len(gs) > 1000:
                    continue
                label = transfom_to_graph(label_path, len(gs), device=device)
                labels.extend(label)
                instances.extend(gs)
    return (torch.tensor(instances, device=device), torch.tensor(labels, device=device, dtype=torch.long))
